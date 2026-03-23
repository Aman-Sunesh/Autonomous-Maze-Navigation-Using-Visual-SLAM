import os
import json
import cv2
import numpy as np

# ---------------------------------------------------------------------------
# 1. Math Models
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

class MapBuilder:
    def __init__(self, map_size=800, scale=60.0):
        self.map_size = map_size
        self.scale = scale 
        self.offset_x = map_size // 2
        self.offset_y = map_size // 2
        self.grid = np.ones((self.map_size, self.map_size, 3), dtype=np.uint8) * 127
        self.trajectory = []
        self.frame_poses = {} # Stores pixel coordinates for each image

    def world_to_pixel(self, x, y):
        px = int(self.offset_x + (x * self.scale))
        py = int(self.offset_y - (y * self.scale)) 
        return px, py

    def update_grid(self, robot_x, robot_y, global_wall_points):
        rpx, rpy = self.world_to_pixel(robot_x, robot_y)
        for wx, wy in global_wall_points:
            wpx, wpy = self.world_to_pixel(wx, wy)
            if 0 <= wpx < self.map_size and 0 <= wpy < self.map_size and 0 <= rpx < self.map_size and 0 <= rpy < self.map_size:
                cv2.line(self.grid, (rpx, rpy), (wpx, wpy), (255, 255, 255), 2)
                cv2.circle(self.grid, (wpx, wpy), 3, (0, 0, 0), -1)

    def draw(self, robot_x, robot_y, theta, mode_str, img_filename=None):
        px, py = self.world_to_pixel(robot_x, robot_y)
        self.trajectory.append((px, py))
        
        if img_filename:
            self.frame_poses[img_filename] = (float(robot_x), float(robot_y), float(theta))

        display_img = self.grid.copy()

        if len(self.trajectory) > 1:
            for i in range(1, len(self.trajectory)):
                cv2.line(display_img, self.trajectory[i-1], self.trajectory[i], (255, 0, 0), 2)

        cv2.circle(display_img, (px, py), 5, (0, 0, 255), -1)
        hx = int(px + 15 * np.cos(theta))
        hy = int(py - 15 * np.sin(theta)) 
        cv2.line(display_img, (px, py), (hx, hy), (0, 200, 0), 2)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(display_img, f"Mode: {mode_str}", (10, 30), font, 0.75, (0, 200, 0), 2, cv2.LINE_AA)
        
        cv2.imshow("Offline SLAM Output", display_img)

# ---------------------------------------------------------------------------
# 2. Offline Replayer Logic
# ---------------------------------------------------------------------------
def parse_actions(actions_list):
    v, w = 0.0, 0.0
    base_v = 2.9462
    base_w = 4.27
    
    if 'FORWARD' in actions_list: v += base_v
    if 'BACKWARD' in actions_list: v -= base_v
    if 'LEFT' in actions_list: w += -base_w if 'BACKWARD' in actions_list else base_w
    if 'RIGHT' in actions_list: w += base_w if 'BACKWARD' in actions_list else -base_w
        
    return v, w

def run_offline_slam():
    CAMERA_W = 320
    CAMERA_H = 240
    CAMERA_F = np.round(CAMERA_W/2.0/np.tan(np.deg2rad(60)))
    K = np.array([[CAMERA_F, 0, CAMERA_W/2.0],
                  [0, CAMERA_F, CAMERA_H/2.0],
                  [0, 0, 1]])

    odom = PureOdometry(initial_x=0.0, initial_y=0.0, initial_theta=np.pi/2)
    map_builder = MapBuilder(map_size=800, scale=60.0)
    wall_detector = WallDetector(K)

    DATA_INFO_PATH = "data/data_info.json"
    IMAGE_DIR = "data/images/"

    if not os.path.exists(DATA_INFO_PATH):
        print(f"Cannot find {DATA_INFO_PATH}!")
        return

    with open(DATA_INFO_PATH) as f:
        raw_data = json.load(f)

    print(f"Loaded {len(raw_data)} frames. Starting SLAM with Loop Closure...")

    prev_step = None
    
    # --- GLOBAL WALL MEMORY ---
    historic_v_walls = [] 
    historic_h_walls = [] 
    MATCH_THRESHOLD = 0.15
    
    for frame in raw_data:
        curr_step = frame['step']
        actions = frame['action']
        img_filename = frame.get('image', None)

        dt = 0.01 if prev_step is None else (curr_step - prev_step) * 0.01
        v, w = parse_actions(actions)
        
        odom.update(v, w, dt)
        
        if img_filename:
            img_path = os.path.join(IMAGE_DIR, img_filename)
            if os.path.exists(img_path):
                fpv = cv2.imread(img_path)
                
                all_local = wall_detector.extract_wall_points(fpv)
                
                # ---------------------------------------------------------
                # 3. ROBUST MANHATTAN CONSTRAINT & LOOP CLOSURE
                # ---------------------------------------------------------
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
                        precise_local_angle = np.arctan2(np.mean(np.sin(local_angles[inlier_mask_segments])), 
                                                         np.mean(np.cos(local_angles[inlier_mask_segments])))
                        
                        global_wall_angle = odom.theta + precise_local_angle
                        target_global_angle = np.round(global_wall_angle / (np.pi/2.0)) * (np.pi/2.0)
                        
                        heading_error = target_global_angle - global_wall_angle
                        odom.theta += heading_error * 0.15 
                        odom.theta = (odom.theta + np.pi) % (2 * np.pi) - np.pi
                        
                        inlier_mask_pts = np.append(inlier_mask_segments, [False]*step_size)
                        valid_local_pts = pts[inlier_mask_pts]
                        
                        if len(valid_local_pts) > 0:
                            global_X = odom.x + (valid_local_pts[:, 0] * np.cos(odom.theta) - valid_local_pts[:, 1] * np.sin(odom.theta))
                            global_Y = odom.y + (valid_local_pts[:, 0] * np.sin(odom.theta) + valid_local_pts[:, 1] * np.cos(odom.theta))
                            
                            mean_wall_x = np.mean(global_X)
                            mean_wall_y = np.mean(global_Y)
                            
                            norm_target = (target_global_angle + np.pi) % (2 * np.pi) - np.pi
                            is_horizontal = np.isclose(abs(norm_target), 0.0, atol=0.1) or np.isclose(abs(norm_target), np.pi, atol=0.1)
                            is_vertical = np.isclose(abs(norm_target), np.pi/2.0, atol=0.1)
                            
                            translation_gain = 0.1 

                            if is_horizontal:
                                if not historic_h_walls:
                                    historic_h_walls.append(mean_wall_y)
                                else:
                                    closest_y = min(historic_h_walls, key=lambda y: abs(y - mean_wall_y))
                                    if abs(closest_y - mean_wall_y) < MATCH_THRESHOLD:
                                        error_y = closest_y - mean_wall_y
                                        odom.y += error_y * translation_gain
                                    else:
                                        historic_h_walls.append(mean_wall_y)
                                        
                            elif is_vertical:
                                if not historic_v_walls:
                                    historic_v_walls.append(mean_wall_x)
                                else:
                                    closest_x = min(historic_v_walls, key=lambda x: abs(x - mean_wall_x))
                                    if abs(closest_x - mean_wall_x) < MATCH_THRESHOLD:
                                        error_x = closest_x - mean_wall_x
                                        odom.x += error_x * translation_gain
                                    else:
                                        historic_v_walls.append(mean_wall_x)
                # ---------------------------------------------------------

                est_x, est_y, est_theta = odom.x, odom.y, odom.theta
                
                global_wall_points = []
                for X_robot, Y_robot in all_local:
                    X_world = est_x + (X_robot * np.cos(est_theta) - Y_robot * np.sin(est_theta))
                    Y_world = est_y + (X_robot * np.sin(est_theta) + Y_robot * np.cos(est_theta))
                    global_wall_points.append((X_world, Y_world))

                map_builder.update_grid(est_x, est_y, global_wall_points)
                map_builder.draw(est_x, est_y, est_theta, "Loop Closure SLAM", img_filename)
                
                cv2.imshow("Processing FPV", fpv)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        prev_step = curr_step

    print("Offline SLAM complete! Exporting clean map and poses...")
    os.makedirs("cache", exist_ok=True)
    cv2.imwrite("cache/slam_map_walls.png", map_builder.grid) 
    with open("cache/frame_poses.json", "w") as f:
        json.dump(map_builder.frame_poses, f)
        
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_offline_slam()