"""
Level-1 Visual Navigation with Robust State Estimation
Pipeline: Pure Kinematics + Manhattan Geometric Constraints + VLAD Global Relocalization
"""

from vis_nav_game import Player, Action, Phase
import pygame
import cv2
import numpy as np
import os
import json
import heapq
import pickle
import networkx as nx
from sklearn.cluster import KMeans
from tqdm import tqdm
import time

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CACHE_DIR = "cache"
IMAGE_DIR = "data/images/"
DATA_INFO_PATH = "data/data_info.json"

# Graph construction
TEMPORAL_WEIGHT = 1.0       # edge weight for consecutive frames
VISUAL_WEIGHT_BASE = 2.0    # base weight for visual shortcut edges
VISUAL_WEIGHT_SCALE = 3.0   # weight += scale * vlad_distance
MIN_SHORTCUT_GAP = 50       # minimum trajectory index gap for shortcuts

os.makedirs(CACHE_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Math Models & Detectors
# ---------------------------------------------------------------------------
class PureOdometry:
    def __init__(self, initial_x=0.0, initial_y=0.0, initial_theta=0.0):
        self.x = initial_x
        self.y = initial_y
        self.theta = initial_theta

    def update(self, v, w, dt):
        if dt <= 0: return
        self.x += v * dt * np.cos(self.theta)
        self.y += v * dt * np.sin(self.theta)
        self.theta += w * dt
        self.theta = (self.theta + np.pi) % (2 * np.pi) - np.pi

class WallDetector:
    def __init__(self, K, true_camera_height=0.21, wall_height=0.30, max_depth=0.3):
        self.K = K
        self.cam_h = true_camera_height 
        self.wall_h = wall_height 
        self.max_depth = max_depth
        self.fx = K[0, 0]
        self.fy = K[1, 1]
        self.cx = K[0, 2]
        self.cy = K[1, 2]

    def extract_wall_points(self, fpv):
        h, w = fpv.shape[:2]
        gray = cv2.cvtColor(fpv, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        mask = np.zeros((h + 2, w + 2), np.uint8)
        seed_point = (w // 2, 5)
        cv2.floodFill(blurred, mask, seed_point, 255, 2, 2, cv2.FLOODFILL_MASK_ONLY)
        sky_mask = mask[1:-1, 1:-1]
        
        all_local_points = []
        step = max(1, w // 60)
        
        for u in range(step // 2, w, step):
            for v in range(5, int(self.cy) - 5):
                if sky_mask[v, u] == 0:
                    delta_h = self.wall_h - self.cam_h 
                    denominator = max((self.cy - v), 1)
                    Z_cam = (self.fy * delta_h) / denominator
                    
                    if Z_cam > self.max_depth or Z_cam <= 0.1: break
                    X_cam = ((u - self.cx) * Z_cam) / self.fx
                    X_robot, Y_robot = Z_cam, -X_cam
                    all_local_points.append((X_robot, Y_robot))
                    break 
                    
        return all_local_points

# ---------------------------------------------------------------------------
# 2. VLAD Feature Extraction
# ---------------------------------------------------------------------------
class VLADExtractor:
    """RootSIFT + VLAD with intra-normalization and power normalization."""

    def __init__(self, n_clusters: int = 128):
        self.n_clusters = n_clusters
        self.sift = cv2.SIFT_create()
        self.codebook = None
        self._sift_cache: dict[str, np.ndarray] = {}

    @property
    def dim(self) -> int:
        return self.n_clusters * 128

    @staticmethod
    def _root_sift(des: np.ndarray) -> np.ndarray:
        des = des / np.sum(des, axis=1, keepdims=True)
        return np.sqrt(des)

    def _des_to_vlad(self, des: np.ndarray) -> np.ndarray:
        labels = self.codebook.predict(des)
        centers = self.codebook.cluster_centers_
        k = self.codebook.n_clusters
        vlad = np.zeros((k, des.shape[1]))
        for i in range(k):
            mask = labels == i
            if np.any(mask):
                vlad[i] = np.sum(des[mask] - centers[i], axis=0)
                norm = np.linalg.norm(vlad[i])
                if norm > 0:
                    vlad[i] /= norm
        vlad = vlad.ravel()
        vlad = np.sign(vlad) * np.sqrt(np.abs(vlad))
        norm = np.linalg.norm(vlad)
        if norm > 0:
            vlad /= norm
        return vlad

    def load_sift_cache(self, file_list: list[str], subsample_rate: int):
        cache_file = os.path.join(CACHE_DIR, f"sift_ss{subsample_rate}.pkl")
        if os.path.exists(cache_file):
            print(f"Loading cached SIFT from {cache_file}")
            with open(cache_file, "rb") as f:
                self._sift_cache = pickle.load(f)
            if all(fname in self._sift_cache for fname in file_list):
                return
            print("  Cache incomplete, re-extracting...")

        self.frame_count = 0
        print(f"Extracting SIFT for {len(file_list)} images...")
        self._sift_cache = {}
        for fname in tqdm(file_list, desc="SIFT"):
            img = cv2.imread(os.path.join(IMAGE_DIR, fname))
            _, des = self.sift.detectAndCompute(img, None)
            if des is not None:
                self._sift_cache[fname] = self._root_sift(des)
        with open(cache_file, "wb") as f:
            pickle.dump(self._sift_cache, f)
        print(f"  Saved {len(self._sift_cache)} descriptors -> {cache_file}")

    def build_vocabulary(self, file_list: list[str]):
        cache_file = os.path.join(CACHE_DIR, f"codebook_k{self.n_clusters}.pkl")
        if os.path.exists(cache_file):
            print(f"Loading cached codebook from {cache_file}")
            with open(cache_file, "rb") as f:
                self.codebook = pickle.load(f)
            return

        all_des = np.vstack([self._sift_cache[f] for f in file_list if f in self._sift_cache])
        print(f"Fitting KMeans (k={self.n_clusters}) on {len(all_des)} descriptors...")
        self.codebook = KMeans(
            n_clusters=self.n_clusters, init='k-means++',
            n_init=3, max_iter=300, tol=1e-4, verbose=1, random_state=42,
        ).fit(all_des)
        print(f"  {self.codebook.n_iter_} iters, inertia={self.codebook.inertia_:.0f}")
        with open(cache_file, "wb") as f:
            pickle.dump(self.codebook, f)

    def extract(self, img: np.ndarray) -> np.ndarray:
        _, des = self.sift.detectAndCompute(img, None)
        if des is None or len(des) == 0:
            return np.zeros(self.dim)
        return self._des_to_vlad(self._root_sift(des))

    def extract_batch(self, file_list: list[str]) -> np.ndarray:
        vectors = []
        for fname in tqdm(file_list, desc="VLAD"):
            if fname in self._sift_cache and len(self._sift_cache[fname]) > 0:
                vectors.append(self._des_to_vlad(self._sift_cache[fname]))
            else:
                vectors.append(np.zeros(self.dim))
        return np.array(vectors)

# ---------------------------------------------------------------------------
# 3. Player / Main Agent
# ---------------------------------------------------------------------------
class KeyboardPlayerPyGame(Player):

    def __init__(self, n_clusters: int = 128, subsample_rate: int = 5, top_k_shortcuts: int = 30):
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None
        self.keymap = None
        self.occupancy_map = cv2.imread("cache/slam_map_walls_cleaned.png", cv2.IMREAD_GRAYSCALE)
        self.map_scale = 60.0 
        self.map_offset = 400 

        # --- A* PATH VISUALIZATION STATE ---
        self.global_path = []              # list of (world_x, world_y)
        self.goal_world_coords = None      # (gx, gy)
        self.is_autonomous = False
        self.lookahead_dist = 0.12
        self.goal_reach_dist = 0.10
        self.path_replan_dist = 0.35

        super().__init__()

        self.subsample_rate = subsample_rate
        self.top_k_shortcuts = top_k_shortcuts

        # Setup Camera Intrinsics and Wall Detector
        CAMERA_W, CAMERA_H = 320, 240
        CAMERA_F = np.round(CAMERA_W / 2.0 / np.tan(np.deg2rad(60)))
        self.K = np.array([[CAMERA_F, 0, CAMERA_W / 2.0],
                           [0, CAMERA_F, CAMERA_H / 2.0],
                           [0, 0, 1]])
        
        self.wall_detector = WallDetector(self.K)
        self.historic_v_walls = []
        self.historic_h_walls = []
        self.MATCH_THRESHOLD = 0.15

        # Load trajectory data
        self.motion_frames = []
        self.file_list = []
        if os.path.exists(DATA_INFO_PATH):
            with open(DATA_INFO_PATH) as f:
                raw = json.load(f)
            pure = {'FORWARD', 'LEFT', 'RIGHT', 'BACKWARD'}
            all_motion = [
                {'step': d['step'], 'image': d['image'], 'action': d['action'][0]}
                for d in raw if len(d['action']) == 1 and d['action'][0] in pure
            ]
            self.motion_frames = all_motion[::subsample_rate]
            self.file_list = [m['image'] for m in self.motion_frames]
            print(f"Frames: {len(all_motion)} total, {len(self.motion_frames)} after {subsample_rate}x subsample")

        self.extractor = VLADExtractor(n_clusters=n_clusters)
        self.database = None
        self.G = None
        self.goal_node = None

        # --- ODOMETRY & MAP STATE ---
        self.slam_map = None
        self.frame_poses = {}
        self.odom = None          
        self.last_time = None     
        self.frame_count = 0

        if os.path.exists("cache/slam_map_walls_cleaned.png") and os.path.exists("cache/frame_poses.json"):
            self.slam_map = cv2.imread("cache/slam_map_walls_cleaned.png")
            with open("cache/frame_poses.json") as f:
                self.frame_poses = json.load(f)
            print("Loaded SLAM map and world coordinates.")
        else:
            print("Warning: SLAM map/poses not found in cache. Run SLAM script first.")

    def reset(self):
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None
        self.odom = None
        self.last_time = None
        self.frame_count = 0
        self.historic_v_walls = []
        self.historic_h_walls = []
        self.global_path = []
        self.is_autonomous = False
        pygame.init()
        self.keymap = {
            pygame.K_LEFT: Action.LEFT,
            pygame.K_RIGHT: Action.RIGHT,
            pygame.K_UP: Action.FORWARD,
            pygame.K_DOWN: Action.BACKWARD,
            pygame.K_SPACE: Action.CHECKIN,
            pygame.K_ESCAPE: Action.QUIT,
        }

    def world_to_pixel(self, world_x, world_y):
        px = int(self.map_offset + (world_x * self.map_scale))
        py = int(self.map_offset - (world_y * self.map_scale))
        return px, py

    def pixel_to_world(self, px, py):
        x = (px - self.map_offset) / self.map_scale
        y = (self.map_offset - py) / self.map_scale
        return x, y

    def plan_astar_path(self):
        """
        Build one fixed geometric shortest path on the cleaned maze map
        from the initial localized pose to the goal position.
        """
        if self.odom is None or self.goal_world_coords is None or self.occupancy_map is None:
            self.global_path = []
            return

        start_px, start_py = self.world_to_pixel(self.odom.x, self.odom.y)
        goal_px, goal_py = self.world_to_pixel(self.goal_world_coords[0], self.goal_world_coords[1])

        # Build a strict free-space mask from the cleaned map:
        # only bright white corridor is drivable.
        free_mask = np.zeros_like(self.occupancy_map, dtype=np.uint8)
        free_mask[self.occupancy_map > 240] = 255

        # Add a bit more wall margin so the path sits away from walls.
        kernel = np.ones((5, 5), np.uint8)
        free_mask = cv2.erode(free_mask, kernel, iterations=1)

        # Distance-to-wall map: larger values = safer / more centered.
        clearance = cv2.distanceTransform(free_mask, cv2.DIST_L2, 5)

        h, w = free_mask.shape

        open_set = []
        heapq.heappush(open_set, (0, start_px, start_py))
        came_from = {}
        g_score = {(start_px, start_py): 0.0}

        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        directions = [(0,1),(0,-1),(1,0),(-1,0)]
        path_found = False

        while open_set:
            _, curr_x, curr_y = heapq.heappop(open_set)

            if heuristic((curr_x, curr_y), (goal_px, goal_py)) < 3:
                goal_px, goal_py = curr_x, curr_y
                path_found = True
                break

            for dx, dy in directions:
                nx_, ny_ = curr_x + dx, curr_y + dy
                if 0 <= nx_ < w and 0 <= ny_ < h and free_mask[ny_, nx_] > 0:
                    # Prefer cells farther from walls.
                    # Small clearance => larger penalty.
                    wall_penalty = 6.0 / max(clearance[ny_, nx_], 1.0)
                    step_cost = 1.0 + wall_penalty
                    tentative_g = g_score[(curr_x, curr_y)] + step_cost
                    if (nx_, ny_) not in g_score or tentative_g < g_score[(nx_, ny_)]:
                        g_score[(nx_, ny_)] = tentative_g
                        priority = tentative_g + heuristic((goal_px, goal_py), (nx_, ny_))
                        heapq.heappush(open_set, (priority, nx_, ny_))
                        came_from[(nx_, ny_)] = (curr_x, curr_y)

        if not path_found:
            self.global_path = []
            return

        curr = (goal_px, goal_py)
        path_pixels = []
        while curr in came_from:
            path_pixels.append(curr)
            curr = came_from[curr]
        path_pixels.reverse()

        # Keep the full path so it hugs the centered route instead of corner-cutting.
        self.global_path = [self.pixel_to_world(px, py) for (px, py) in path_pixels]

    def _wrap_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def _distance_to_path(self):
        if self.odom is None or not self.global_path:
            return np.inf
        robot = np.array([self.odom.x, self.odom.y], dtype=np.float32)
        return min(np.linalg.norm(np.array(p, dtype=np.float32) - robot) for p in self.global_path)

    def _is_forward_safe(self, samples=6, step_dt=0.01):
        """
        Short forward rollout in map space.
        If any predicted point lands in an occupied cell, do not move forward.
        """
        if self.odom is None:
            return False

        test_x = self.odom.x
        test_y = self.odom.y
        step = 2.9462 * step_dt

        for _ in range(samples):
            test_x += step * np.cos(self.odom.theta)
            test_y += step * np.sin(self.odom.theta)
            if not self.is_state_valid(test_x, test_y):
                return False
        return True

    def get_autonomous_action(self):
        """
        Minimal waypoint follower for the blue A* path:
        - prune passed waypoints
        - turn toward a lookahead point
        - only move forward if the short rollout is safe
        - replan if we drift too far from the path
        """
        if self.odom is None or self.goal_world_coords is None:
            return Action.IDLE

        if not self.global_path:
            self.plan_astar_path()
            if not self.global_path:
                return Action.IDLE

        robot = np.array([self.odom.x, self.odom.y], dtype=np.float32)
        goal = np.array(self.goal_world_coords, dtype=np.float32)

        # Auto-finish near the target
        if np.linalg.norm(goal - robot) < self.goal_reach_dist:
            print("[AUTO] Goal reached. CHECKIN.")
            self.is_autonomous = False
            return Action.CHECKIN

        # Replan if relocalization / drift moved us too far from the current path
        if self._distance_to_path() > self.path_replan_dist:
            self.plan_astar_path()
            if not self.global_path:
                return Action.IDLE

        # Prune waypoints we have already reached
        while len(self.global_path) > 1:
            wp0 = np.array(self.global_path[0], dtype=np.float32)
            if np.linalg.norm(wp0 - robot) < 0.08:
                self.global_path.pop(0)
            else:
                break

        if len(self.global_path) > 1 and np.linalg.norm(np.array(self.global_path[0], dtype=np.float32) - robot) < 0.12:
            target_pt = np.array(self.global_path[1], dtype=np.float32)
        else:
            target_pt = np.array(self.global_path[0], dtype=np.float32)

        dx = float(target_pt[0] - self.odom.x)
        dy = float(target_pt[1] - self.odom.y)
        target_heading = np.arctan2(dy, dx)
        heading_error = self._wrap_angle(target_heading - self.odom.theta)

        TURN_TOL = np.deg2rad(5)

        # Basic obstacle avoidance: if forward rollout is unsafe, rotate first
        if not self._is_forward_safe(samples=5, step_dt=0.01):
            return Action.LEFT if heading_error >= 0 else Action.RIGHT

        # Turn in place until reasonably aligned to the next path segment
        if heading_error > TURN_TOL:
            return Action.LEFT
        elif heading_error < -TURN_TOL:
            return Action.RIGHT
        else:
            return Action.FORWARD

    def act(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self.last_act = Action.QUIT
                return Action.QUIT
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_a:
                    self.is_autonomous = not self.is_autonomous
                    self.last_act = Action.IDLE
                    print(f"[AUTO] {'ON' if self.is_autonomous else 'OFF'}")
                    if self.is_autonomous and self.odom is not None:
                        self.plan_astar_path()
                elif event.key in self.keymap:
                    self.is_autonomous = False
                    self.last_act |= self.keymap[event.key]
                else:
                    self.show_target_images()
            if event.type == pygame.KEYUP:
                if event.key in self.keymap:
                    self.last_act ^= self.keymap[event.key]

        if self.is_autonomous:
            self.last_act = self.get_autonomous_action()

        return self.last_act

    def see(self, fpv):
        if fpv is None or len(fpv.shape) < 3:
            return
        self.fpv = fpv
        current_time = time.time()
        self.frame_count += 1

        if self.screen is None:
            h, w, _ = fpv.shape
            self.screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption("KeyboardPlayer:fpv")

        if self._state and self._state[1] == Phase.NAVIGATION:
            
            # --- 1. GLOBAL LOCALIZATION (Runs only once at start) ---
            if self.odom is None:
                feat = self.extractor.extract(self.fpv)
                best_idx = int(np.argmax(self.database @ feat))
                best_file = self.file_list[best_idx]
                
                if best_file in self.frame_poses:
                    wx, wy, wtheta = self.frame_poses[best_file]
                    self.odom = PureOdometry(wx, wy, wtheta)
                    print(f"\n[!] VLAD Initialized Position: X={wx:.2f}, Y={wy:.2f}, Theta={np.rad2deg(wtheta):.0f}deg")
                else:
                    self.odom = PureOdometry(0.0, 0.0, 0.0)
                self.last_time = current_time

                # Build the A* path once from the initial localized pose
                if self.goal_world_coords is not None and not self.global_path:
                    self.plan_astar_path() 

            # --- 2. KINEMATIC INTEGRATION (Runs every frame) ---
            elif self.last_time is not None:
                dt = 0.01 
                v, w = 0.0, 0.0
                base_v = 2.9462
                base_w = 4.27 
                
                is_back = bool(self.last_act & Action.BACKWARD)
                if self.last_act & Action.FORWARD: v += base_v
                if is_back: v -= base_v
                if self.last_act & Action.LEFT: w += -base_w if is_back else base_w
                if self.last_act & Action.RIGHT: w += base_w if is_back else -base_w
                
                next_x = self.odom.x + v * dt * np.cos(self.odom.theta)
                next_y = self.odom.y + v * dt * np.sin(self.odom.theta)
                
                # In AUTO mode, never slide into walls.
                # Either the full translational step is valid, or do not move.
                if self.is_autonomous and abs(v) > 0:
                    if self.is_state_valid(next_x, next_y):
                        self.odom.x = next_x
                        self.odom.y = next_y
                else:
                    # Manual mode keeps the old sliding behavior.
                    if self.is_state_valid(next_x, self.odom.y):
                        self.odom.x = next_x
                    if self.is_state_valid(self.odom.x, next_y):
                        self.odom.y = next_y
                self.odom.theta += w * dt
                self.odom.theta = (self.odom.theta + np.pi) % (2 * np.pi) - np.pi

            # --- 3. GEOMETRIC LOCAL CORRECTION (Throttled to every 2nd frame) ---
            if self.frame_count % 2 == 0 and self.odom is not None:
                all_local = self.wall_detector.extract_wall_points(self.fpv)
                
                if len(all_local) > 15:
                    pts = np.array(all_local, dtype=np.float32)
                    step_size = 3
                    dx = pts[step_size:, 0] - pts[:-step_size, 0]
                    dy = pts[step_size:, 1] - pts[:-step_size, 1]
                    
                    local_angles = np.arctan2(dy, dx)
                    hist, bin_edges = np.histogram(local_angles, bins=36, range=(-np.pi, np.pi))
                    dominant_bin = np.argmax(hist)
                    dominant_local_angle = (bin_edges[dominant_bin] + bin_edges[dominant_bin+1]) / 2.0
                    
                    angle_diffs = np.abs(np.arctan2(np.sin(local_angles - dominant_local_angle), 
                                                    np.cos(local_angles - dominant_local_angle)))
                    inlier_mask_segments = angle_diffs < np.deg2rad(15)
                    
                    if np.sum(inlier_mask_segments) > 5:
                        precise_local_angle = np.arctan2(
                            np.mean(np.sin(local_angles[inlier_mask_segments])), 
                            np.mean(np.cos(local_angles[inlier_mask_segments]))
                        )
                        
                        # Soft Heading correction based on Manhattan World
                        global_wall_angle = self.odom.theta + precise_local_angle
                        target_global_angle = np.round(global_wall_angle / (np.pi/2.0)) * (np.pi/2.0)
                        heading_error = target_global_angle - global_wall_angle
                        
                        self.odom.theta += heading_error * 0.15 
                        self.odom.theta = (self.odom.theta + np.pi) % (2 * np.pi) - np.pi
                        
                        # In AUTO mode, keep heading correction but DO NOT shift x/y.
                        # Those translation nudges can move the marker forward while
                        # the real robot is actually pinned on a wall.
                        if not self.is_autonomous:
                            inlier_mask_pts = np.append(inlier_mask_segments, [False]*step_size)
                            valid_local_pts = pts[inlier_mask_pts]
                            
                            if len(valid_local_pts) > 0:
                                global_X = self.odom.x + (valid_local_pts[:, 0] * np.cos(self.odom.theta) - valid_local_pts[:, 1] * np.sin(self.odom.theta))
                                global_Y = self.odom.y + (valid_local_pts[:, 0] * np.sin(self.odom.theta) + valid_local_pts[:, 1] * np.cos(self.odom.theta))
                                
                                mean_wall_x = np.mean(global_X)
                                mean_wall_y = np.mean(global_Y)
                                
                                norm_target = (target_global_angle + np.pi) % (2 * np.pi) - np.pi
                                is_horizontal = np.isclose(abs(norm_target), 0.0, atol=0.1) or np.isclose(abs(norm_target), np.pi, atol=0.1)
                                is_vertical = np.isclose(abs(norm_target), np.pi/2.0, atol=0.1)
                                translation_gain = 0.1 

                                if is_horizontal:
                                    if not self.historic_h_walls:
                                        self.historic_h_walls.append(mean_wall_y)
                                    else:
                                        closest_y = min(self.historic_h_walls, key=lambda y: abs(y - mean_wall_y))
                                        if abs(closest_y - mean_wall_y) < self.MATCH_THRESHOLD:
                                            self.odom.y += (closest_y - mean_wall_y) * translation_gain
                                        else:
                                            self.historic_h_walls.append(mean_wall_y)
                                            
                                elif is_vertical:
                                    if not self.historic_v_walls:
                                        self.historic_v_walls.append(mean_wall_x)
                                    else:
                                        closest_x = min(self.historic_v_walls, key=lambda x: abs(x - mean_wall_x))
                                        if abs(closest_x - mean_wall_x) < self.MATCH_THRESHOLD:
                                            self.odom.x += (closest_x - mean_wall_x) * translation_gain
                                        else:
                                            self.historic_v_walls.append(mean_wall_x)

            # --- 4. VLAD GLOBAL RELOCALIZATION & UI (Throttled to every 4th frame) ---
            if self.frame_count % 4 == 0 and self.odom is not None and self.slam_map is not None:
                feat = self.extractor.extract(self.fpv)
                sims = self.database @ feat
                best_idx = int(np.argmax(sims))
                
                # Complementary filter: Soft pull towards map pose if highly confident
                if sims[best_idx] > 0.85: 
                    best_file = self.file_list[best_idx]
                    if best_file in self.frame_poses:
                        mx, my, mtheta = self.frame_poses[best_file]
                        alpha = 0.15
                        
                        if not self.is_autonomous:
                            self.odom.x = (1 - alpha) * self.odom.x + alpha * mx
                            self.odom.y = (1 - alpha) * self.odom.y + alpha * my
                        
                        diff = (mtheta - self.odom.theta + np.pi) % (2 * np.pi) - np.pi
                        self.odom.theta += alpha * diff
                        self.odom.theta = (self.odom.theta + np.pi) % (2 * np.pi) - np.pi

                keys = pygame.key.get_pressed()
                if keys[pygame.K_q]:
                    self.display_next_best_view()
                self.display_global_map()

        # Update live camera stream every frame
        rgb = fpv[:, :, ::-1]
        surface = pygame.image.frombuffer(rgb.tobytes(), rgb.shape[1::-1], 'RGB')
        self.screen.blit(surface, (0, 0))
        pygame.display.update()

    def is_state_valid(self, x, y):
        if self.occupancy_map is None: 
            return True # Failsafe if map isn't loaded
            
        px = int(self.map_offset + (x * self.map_scale))
        py = int(self.map_offset - (y * self.map_scale))
        
        h, w = self.occupancy_map.shape
        if 0 <= px < w and 0 <= py < h:
            # Walls are usually near 0 (black).
            return self.occupancy_map[py, px] > 50
        return False # Out of bounds is invalid
    
    # -----------------------------------------------------------------------
    # Setup and Visualizer Methods
    # -----------------------------------------------------------------------
    def set_target_images(self, images):
        super().set_target_images(images)
        self.show_target_images()

    def pre_navigation(self):
        super().pre_navigation()
        self._build_database()
        self._build_graph()
        self._setup_goal()

    def _build_database(self):
        if self.database is not None:
            return
        self.extractor.load_sift_cache(self.file_list, self.subsample_rate)
        self.extractor.build_vocabulary(self.file_list)
        self.database = self.extractor.extract_batch(self.file_list)

    def _build_graph(self):
        if self.G is not None: return
        n = len(self.database)
        self.G = nx.DiGraph() 
        self.G.add_nodes_from(range(n))

        for i in range(n - 1):
            self.G.add_edge(i, i + 1, weight=TEMPORAL_WEIGHT, edge_type="temporal")

        sim = self.database @ self.database.T
        np.fill_diagonal(sim, -2)

        for i in range(n):
            lo = max(0, i - MIN_SHORTCUT_GAP)
            hi = min(n, i + MIN_SHORTCUT_GAP + 1)
            sim[i, lo:hi] = -2
        sim[~np.triu(np.ones((n, n), dtype=bool), k=1)] = -2

        flat = sim.ravel()
        top_k = self.top_k_shortcuts
        top_idx = np.argpartition(flat, -top_k)[-top_k:]
        top_idx = top_idx[np.argsort(-flat[top_idx])]

        for rank, fi in enumerate(top_idx):
            i, j = divmod(int(fi), n)
            s = float(flat[fi])
            d = float(np.sqrt(max(0, 2 - 2 * s)))
            self.G.add_edge(i, j, weight=VISUAL_WEIGHT_BASE + VISUAL_WEIGHT_SCALE * d, edge_type="visual")

    def _setup_goal(self):
        if self.goal_node is not None:
            return

        targets = self.get_target_images()
        if not targets:
            return

        n = len(self.database)
        agg = np.zeros(n, dtype=np.float32)

        valid_count = 0
        for t in targets:
            if t is None:
                continue
            feat = self.extractor.extract(t)
            if np.linalg.norm(feat) == 0:
                continue

            sims = self.database @ feat
            agg += sims
            valid_count += 1

        if valid_count == 0:
            return

        agg /= valid_count

        # small 1D smoothing so one isolated false match does not win
        smooth = agg.copy()
        for i in range(1, n - 1):
            smooth[i] = 0.25 * agg[i - 1] + 0.5 * agg[i] + 0.25 * agg[i + 1]

        self.goal_node = int(np.argmax(smooth))

        goal_file = self.file_list[self.goal_node]
        if goal_file in self.frame_poses:
            gx, gy, _ = self.frame_poses[goal_file]
            self.goal_world_coords = (gx, gy)

        # optional debug
        topk = np.argsort(-smooth)[:5]
        print("\nTop goal candidates:")
        for rank, idx in enumerate(topk, 1):
            print(rank, idx, self.file_list[idx], float(smooth[idx]))

    def _load_img(self, idx: int) -> np.ndarray | None:
        if 0 <= idx < len(self.file_list):
            return cv2.imread(os.path.join(IMAGE_DIR, self.file_list[idx]))
        return None

    def _get_current_node(self) -> int:
        feat = self.extractor.extract(self.fpv)
        return int(np.argmax(self.database @ feat))

    def _get_path(self, start: int) -> list[int]:
        try:
            return nx.shortest_path(self.G, start, self.goal_node, weight="weight")
        except nx.NetworkXNoPath:
            return [start]

    def _edge_action(self, a: int, b: int) -> str:
        REVERSE = {'FORWARD': 'BACKWARD', 'BACKWARD': 'FORWARD', 'LEFT': 'RIGHT', 'RIGHT': 'LEFT'}
        if b == a + 1 and a < len(self.motion_frames):
            return self.motion_frames[a]['action']
        elif b == a - 1 and b < len(self.motion_frames):
            return REVERSE.get(self.motion_frames[b]['action'], '?')
        return '?'

    def show_target_images(self):
        targets = self.get_target_images()
        if not targets: return
        top = cv2.hconcat(targets[:2])
        bot = cv2.hconcat(targets[2:])
        img = cv2.vconcat([top, bot])
        h, w = img.shape[:2]
        cv2.line(img, (w // 2, 0), (w // 2, h), (0, 0, 0), 2)
        cv2.line(img, (0, h // 2), (w, h // 2), (0, 0, 0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        for label, pos in [('Front', (10, 25)), ('Right', (w//2+10, 25)),
                           ('Back', (10, h//2+25)), ('Left', (w//2+10, h//2+25))]:
            cv2.putText(img, label, pos, font, 0.75, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow('Target Images', img)
        cv2.waitKey(1)

    def display_global_map(self):
        if self.slam_map is None or self.goal_node is None: return
        display_map = self.slam_map.copy()

        goal_file = self.file_list[self.goal_node]
        if goal_file in self.frame_poses:
            gx, gy, _ = self.frame_poses[goal_file]
            gpx, gpy = self.world_to_pixel(gx, gy)
            cv2.circle(display_map, (gpx, gpy), 10, (0, 0, 255), -1) 
            cv2.putText(display_map, "TARGET", (gpx + 15, gpy - 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        # Draw A* geometric shortest path (blue/cyan style)
        if self.global_path:
            for i in range(len(self.global_path) - 1):
                p1 = self.world_to_pixel(self.global_path[i][0], self.global_path[i][1])
                p2 = self.world_to_pixel(self.global_path[i + 1][0], self.global_path[i + 1][1])
                cv2.line(display_map, p1, p2, (255, 200, 0), 2)
                cv2.circle(display_map, p1, 2, (255, 200, 0), -1)

        # Draw current path start marker from odometry
        if self.odom is not None:
            rpx, rpy = self.world_to_pixel(self.odom.x, self.odom.y)
            cv2.circle(display_map, (rpx, rpy), 6, (0, 255, 0), -1)
            hx = int(rpx + 15 * np.cos(self.odom.theta))
            hy = int(rpy - 15 * np.sin(self.odom.theta))
            cv2.line(display_map, (rpx, rpy), (hx, hy), (0, 255, 0), 2)

        mode_txt = "AUTO" if self.is_autonomous else "MANUAL"
        mode_col = (0, 255, 0) if self.is_autonomous else (0, 0, 255)
        cv2.putText(display_map, "A* PATH", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 200, 0), 2, cv2.LINE_AA)
        cv2.putText(display_map, mode_txt, (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_col, 2, cv2.LINE_AA)

        cv2.imshow("Global Metric Map (Live Odometry + Constraints)", display_map)
        cv2.waitKey(1)

    def display_next_best_view(self):
        ACT = {'FORWARD': 'FWD', 'BACKWARD': 'BACK', 'LEFT': 'LEFT', 'RIGHT': 'RIGHT'}
        FONT = cv2.FONT_HERSHEY_SIMPLEX
        AA = cv2.LINE_AA
        TW, TH = 260, 195          
        PW, PH = TW * 3 // 5, TH * 3 // 5   
        N_PREVIEW = 5

        cur = self._get_current_node()
        cur_sim = float(self.database[cur] @ self.extractor.extract(self.fpv))
        cur_d = float(np.sqrt(max(0, 2 - 2 * cur_sim)))
        path = self._get_path(cur)
        hops = len(path) - 1

        edge_info = []
        for a, b in zip(path[:-1], path[1:]):
            et = self.G[a][b].get("edge_type", "temporal")
            if et == "temporal":
                act = ACT.get(self._edge_action(a, b), '?')
                edge_info.append(("seq", act, b == a + 1))
            else:
                edge_info.append(("vis", None, None))
        t_steps = sum(1 for e in edge_info if e[0] == "seq")
        v_jumps = len(edge_info) - t_steps

        if edge_info:
            etype, act, _ = edge_info[0]
            hint = act if etype == "seq" else "VISUAL JUMP"
        else:
            hint = "AT GOAL"
        near = hops <= 5

        panel_w = TW * 3
        bar = np.zeros((40, panel_w, 3), dtype=np.uint8)
        bar[:] = (0, 0, 160) if near else (50, 35, 15)
        txt = (f"Node {cur} (d={cur_d:.3f})"
               f"  |  Goal {self.goal_node}"
               f"  |  {hops} hops ({t_steps}s+{v_jumps}v)"
               f"  |  >> {hint}")
        cv2.putText(bar, txt, (8, 27), FONT, 0.48, (255, 255, 255), 1, AA)
        if near:
            cv2.putText(bar, "NEAR TARGET — SPACE", (panel_w - 220, 27), FONT, 0.48, (0, 255, 255), 1, AA)

        def thumb(img, label, color, extra=None):
            t = cv2.resize(img, (TW, TH))
            cv2.rectangle(t, (0, 0), (TW-1, TH-1), color, 2)
            cv2.putText(t, label, (6, 22), FONT, 0.55, color, 1, AA)
            if extra:
                cv2.putText(t, extra, (6, 44), FONT, 0.45, (200, 200, 200), 1, AA)
            return t

        fpv_t = thumb(self.fpv, "Live FPV", (255, 255, 255))
        match_img = self._load_img(cur)
        if match_img is None: match_img = np.zeros((TH, TW, 3), dtype=np.uint8)
        match_t = thumb(match_img, f"Match: node {cur}", (0, 255, 0), f"d={cur_d:.3f}")
        targets = self.get_target_images()
        tgt = targets[0] if targets else np.zeros((TH, TW, 3), dtype=np.uint8)
        tgt_t = thumb(tgt, "Target (front)", (0, 140, 255))
        row1 = cv2.hconcat([fpv_t, match_t, tgt_t])

        preview = path[1:1 + N_PREVIEW]
        cells = []
        for p in range(N_PREVIEW):
            if p < len(preview):
                img = self._load_img(preview[p])
                if img is None: img = np.zeros((PH, PW, 3), dtype=np.uint8)
                img = cv2.resize(img, (PW, PH))
                etype, act, is_fwd = edge_info[p]
                if etype == "seq":
                    lbl = f"{'>' if is_fwd else '<'} {act}"
                    clr = (200, 200, 0)
                else:
                    lbl = "~ VISUAL"
                    clr = (200, 100, 255)
                cv2.rectangle(img, (0, 0), (PW-1, PH-1), clr, 1)
                cv2.putText(img, f"+{p+1} node {preview[p]}", (4, 16), FONT, 0.38, (255, 255, 255), 1, AA)
                cv2.putText(img, lbl, (4, 34), FONT, 0.38, clr, 1, AA)
            else:
                img = np.zeros((PH, PW, 3), dtype=np.uint8)
            cells.append(img)
        row2 = cv2.hconcat(cells)

        if row2.shape[1] < panel_w:
            pad = np.zeros((PH, panel_w - row2.shape[1], 3), dtype=np.uint8)
            row2 = cv2.hconcat([row2, pad])

        panel = cv2.vconcat([bar, row1, row2])
        cv2.imshow("Navigation", panel)
        cv2.waitKey(1)

if __name__ == "__main__":
    import argparse
    import vis_nav_game

    parser = argparse.ArgumentParser()
    parser.add_argument("--subsample", type=int, default=5, help="Take every Nth motion frame (default: 5)")
    parser.add_argument("--n-clusters", type=int, default=128, help="VLAD codebook size (default: 128)")
    parser.add_argument("--top-k", type=int, default=30, help="Number of global visual shortcut edges (default: 30)")
    args = parser.parse_args()

    vis_nav_game.play(the_player=KeyboardPlayerPyGame(
        n_clusters=args.n_clusters,
        subsample_rate=args.subsample,
        top_k_shortcuts=args.top_k,
    ))