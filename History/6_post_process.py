import os
import json
import cv2
import numpy as np

# ---------------------------------------------------------------------------
# 1. Math Models
# ---------------------------------------------------------------------------
class VisualStuckDetector:
    def __init__(self, fx, diff_thresh=3.0, slide_thresh=1.5):
        self.fx = fx
        self.diff_thresh = diff_thresh
        self.slide_thresh = slide_thresh
        self.prev_gray = None

    def filter_actions(self, curr_img, commanded_v, commanded_w, dt, is_near_wall):
        curr_gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
        curr_float = np.float32(curr_gray) 

        if self.prev_gray is None:
            self.prev_gray = curr_float
            return commanded_v, commanded_w, "MOVING", 0.0, 0.0

        if commanded_v == 0.0 and commanded_w == 0.0:
            self.prev_gray = curr_float
            return 0.0, 0.0, "IDLE", 0.0, 0.0

        shift, _ = cv2.phaseCorrelate(self.prev_gray, curr_float)
        dx, dy = shift 

        diff = cv2.absdiff(self.prev_gray, curr_float)
        mean_diff = float(np.mean(diff))

        self.prev_gray = curr_float

        # If no wall is close in our map memory, bypass collision logic
        if not is_near_wall:
            return commanded_v, commanded_w, "MOVING", mean_diff, dx

        actual_delta_theta = np.arctan(dx / self.fx)
        visual_w = actual_delta_theta / dt

        if commanded_w == 0.0:
            if abs(dx) > self.slide_thresh:
                print("Sliding: ", dx)
                return 0.0, visual_w, "SLIDING", mean_diff, dx
            if mean_diff < self.diff_thresh:
                print("Stuck: ", mean_diff)
                return 0.0, 0.0, "STUCK", mean_diff, dx
        else:
            if abs(visual_w) < (abs(commanded_w) * 0.60):
                print("Block: ", visual_w)
                return 0.0, visual_w, "BLOCKED ROT", mean_diff, dx
            if mean_diff < self.diff_thresh:
                print("Stuck: ", mean_diff)
                return 0.0, 0.0, "STUCK", mean_diff, dx

        return commanded_v, commanded_w, "MOVING", mean_diff, dx
    
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
        
        # --- NEW: REAL-TIME WALL DATA DICTIONARY ---
        # Maps visual pixel coordinates (px, py) to world coordinates (wx, wy)
        self.live_2d_walls = {} 
        
        grid_m = 0.4
        offset_m = 0.2
        grid_px = int(grid_m * self.scale)
        offset_px = int(offset_m * self.scale)
        grid_color = (100, 100, 100) 
        
        for x in range(self.offset_x + offset_px, self.map_size, grid_px):
            cv2.line(self.grid, (x, 0), (x, self.map_size), grid_color, 1)
        for x in range(self.offset_x - offset_px, -1, -grid_px):
            cv2.line(self.grid, (x, 0), (x, self.map_size), grid_color, 1)
        for y in range(self.offset_y + offset_px, self.map_size, grid_px):
            cv2.line(self.grid, (0, y), (self.map_size, y), grid_color, 1)
        for y in range(self.offset_y - offset_px, -1, -grid_px):
            cv2.line(self.grid, (0, y), (self.map_size, y), grid_color, 1)
        
        self.trajectory = []
        self.frame_poses = {} 

    def world_to_pixel(self, x, y):
        px = int(self.offset_x + (x * self.scale))
        py = int(self.offset_y - (y * self.scale)) 
        return px, py

    def update_grid(self, robot_x, robot_y, global_wall_points, global_open_points):
        rpx, rpy = self.world_to_pixel(robot_x, robot_y)
        
        # 1. Create a hidden background mask to track exactly where the clearing rays go
        ray_mask = np.zeros((self.map_size, self.map_size), dtype=np.uint8)

        # Draw rays into open space
        for ox, oy in global_open_points:
            opx, opy = self.world_to_pixel(ox, oy)
            if 0 <= opx < self.map_size and 0 <= opy < self.map_size and 0 <= rpx < self.map_size and 0 <= rpy < self.map_size:
                cv2.line(ray_mask, (rpx, rpy), (opx, opy), 255, 2)
                cv2.line(self.grid, (rpx, rpy), (opx, opy), (255, 255, 255), 2)

        # Draw rays up to the physical walls
        snapped_walls = []
        for wx, wy in global_wall_points:
            snapped_x = round((wx - 0.2) / 0.4) * 0.4 + 0.2
            snapped_y = round((wy - 0.2) / 0.4) * 0.4 + 0.2
            
            if abs(wx - snapped_x) < abs(wy - snapped_y):
                wx = snapped_x
            else:
                wy = snapped_y

            wpx, wpy = self.world_to_pixel(wx, wy)
            
            if 0 <= wpx < self.map_size and 0 <= wpy < self.map_size and 0 <= rpx < self.map_size and 0 <= rpy < self.map_size:
                snapped_walls.append((wx, wy, wpx, wpy))
                cv2.line(ray_mask, (rpx, rpy), (wpx, wpy), 255, 2)
                cv2.line(self.grid, (rpx, rpy), (wpx, wpy), (255, 255, 255), 2)

        # --- 2. THE REAL-TIME DATA REMOVAL ---
        # Find every single pixel the white rays touched on the hidden mask
        erased_y, erased_x = np.where(ray_mask == 255)
        
        # Instantly delete those coordinates from our active memory
        for px, py in zip(erased_x, erased_y):
            self.live_2d_walls.pop((px, py), None)

        # # --- 3. ADD NEW WALLS --- MAYBE FASTER
        # for wx, wy, wpx, wpy in snapped_walls:
        #     # Draw the fat blob for the human eyes
        #     cv2.circle(self.grid, (wpx, wpy), 3, (0, 0, 0), -1)
            
        #     # Save ONLY the exact center coordinate to the data dictionary.
        #     # If a clearing ray hits this single anchor pixel later, the whole wall is deleted.
        #     self.live_2d_walls[(wpx, wpy)] = (wx, wy)
            
        # --- 3. ADD NEW WALLS ---
        for wx, wy, wpx, wpy in snapped_walls:
            cv2.circle(self.grid, (wpx, wpy), 3, (0, 0, 0), -1)
            
            # Map every pixel of the drawn circle to this coordinate
            # If a ray grazes the edge of the visual circle later, it deletes the data instantly.
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx*dx + dy*dy <= 9:
                        self.live_2d_walls[(wpx+dx, wpy+dy)] = (wx, wy)

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
    stuck_detector = VisualStuckDetector(fx=CAMERA_F, diff_thresh=5.0, slide_thresh=6.0) 

    DATA_INFO_PATH = "data_bad/data_info.json"
    IMAGE_DIR = "data_bad/images/"

    if not os.path.exists(DATA_INFO_PATH):
        print(f"Cannot find {DATA_INFO_PATH}!")
        return

    with open(DATA_INFO_PATH) as f:
        raw_data = json.load(f)

    print(f"Loaded {len(raw_data)} frames. Starting SLAM with Loop Closure...")

    prev_step = None
    historic_wall_centers = [] 
    MATCH_THRESHOLD = 0.2 
    frame_walls = {} 
    
    for frame in raw_data:
        curr_step = frame['step']
        actions = frame['action']
        img_filename = frame.get('image', None)
        
        if (curr_step < 00):
            continue

        dt = 0.01 if prev_step is None else (curr_step - prev_step) * 0.01
        cmd_v, cmd_w = parse_actions(actions)
        
        fpv = None
        status = "MOVING"
        visual_delta = 0.0
        shift_dx = 0.0

        if img_filename:
            img_path = os.path.join(IMAGE_DIR, img_filename)
            if os.path.exists(img_path):
                fpv = cv2.imread(img_path)
                
                is_near_wall = False
                rpx, rpy = map_builder.world_to_pixel(odom.x, odom.y)
                px_radius = int(0.05 * map_builder.scale)
                
                y1 = max(0, rpy - px_radius)
                y2 = min(map_builder.grid.shape[0], rpy + px_radius)
                x1 = max(0, rpx - px_radius)
                x2 = min(map_builder.grid.shape[1], rpx + px_radius)
                
                local_map = map_builder.grid[y1:y2, x1:x2]
                
                if np.any(np.all(local_map == [0, 0, 0], axis=-1)):
                    is_near_wall = True
                
                v, w, status, visual_delta, shift_dx = stuck_detector.filter_actions(fpv, cmd_v, cmd_w, dt, is_near_wall)
            else:
                v, w = cmd_v, cmd_w
        else:
            v, w = cmd_v, cmd_w

        odom.update(v, w, dt)
        
        if fpv is not None:
            all_local = wall_detector.extract_wall_points(fpv)
            
            # ---------------------------------------------------------
            # 3. ROBUST MANHATTAN CONSTRAINT & ABSOLUTE GRID SNAP
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
                    odom.theta += heading_error * 0.4
                    odom.theta = (odom.theta + np.pi) % (2 * np.pi) - np.pi
                    
                    inlier_mask_pts = np.append(inlier_mask_segments, [False]*step_size)
                    valid_local_pts = pts[inlier_mask_pts]
                    
                    if len(valid_local_pts) > 0:
                        global_X = odom.x + (valid_local_pts[:, 0] * np.cos(odom.theta) - valid_local_pts[:, 1] * np.sin(odom.theta))
                        global_Y = odom.y + (valid_local_pts[:, 0] * np.sin(odom.theta) + valid_local_pts[:, 1] * np.cos(odom.theta))
                        
                        mean_wall_x = np.mean(global_X)
                        mean_wall_y = np.mean(global_Y)
                        
                        historic_wall_centers.append((mean_wall_x, mean_wall_y))
                        
                        norm_target = (target_global_angle + np.pi) % (2 * np.pi) - np.pi
                        is_horizontal = np.isclose(abs(norm_target), 0.0, atol=0.1) or np.isclose(abs(norm_target), np.pi, atol=0.1)
                        is_vertical = np.isclose(abs(norm_target), np.pi/2.0, atol=0.1)
                        
                        translation_gain = 0.05

                        if is_horizontal:
                            # Snap directly to the nearest grid wall (k * 0.4 + 0.2)
                            target_y = np.round((mean_wall_y - 0.2) / 0.4) * 0.4 + 0.2
                            if abs(target_y - mean_wall_y) < MATCH_THRESHOLD:
                                error_y = target_y - mean_wall_y
                                odom.y += error_y * translation_gain
                                    
                        elif is_vertical:
                            # Snap directly to the nearest grid wall (k * 0.4 + 0.2)
                            target_x = np.round((mean_wall_x - 0.2) / 0.4) * 0.4 + 0.2
                            if abs(target_x - mean_wall_x) < MATCH_THRESHOLD:
                                error_x = target_x - mean_wall_x
                                odom.x += error_x * translation_gain
            # ---------------------------------------------------------

            est_x, est_y, est_theta = odom.x, odom.y, odom.theta
            
            est_x, est_y, est_theta = odom.x, odom.y, odom.theta
            
            global_wall_points = []
            global_open_points = []
            
            for pt in all_local:
                # Handle both the old (2-item) and new (3-item) extract_wall_points output
                if len(pt) == 3:
                    X_robot, Y_robot, is_wall = pt
                else:
                    X_robot, Y_robot = pt
                    is_wall = True 

                X_world = est_x + (X_robot * np.cos(est_theta) - Y_robot * np.sin(est_theta))
                Y_world = est_y + (X_robot * np.sin(est_theta) + Y_robot * np.cos(est_theta))
                
                if is_wall:
                    # Snap the real walls
                    snapped_x = round((X_world - 0.2) / 0.4) * 0.4 + 0.2
                    snapped_y = round((Y_world - 0.2) / 0.4) * 0.4 + 0.2
                    
                    if abs(X_world - snapped_x) < abs(Y_world - snapped_y):
                        X_world = snapped_x
                    else:
                        Y_world = snapped_y
                        
                    global_wall_points.append((X_world, Y_world))
                else:
                    # Save the open space rays (Un-snapped)
                    global_open_points.append((X_world, Y_world))

            if img_filename:
                frame_walls[img_filename] = global_wall_points

            # Fire both lists into the MapBuilder so it draws walls AND erases ghost pixels
            map_builder.update_grid(est_x, est_y, global_wall_points, global_open_points)
            
            map_mode_text = status if status != "MOVING" else "Loop Closure SLAM"
            map_builder.draw(est_x, est_y, est_theta, map_mode_text, img_filename)
            
            cv2.putText(fpv, f"Img: {img_filename}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(fpv, f"Cmd: {actions}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(fpv, f"X-Shift: {shift_dx:.2f} | Inj w: {w:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            if status != "MOVING":
                if status == "BLOCKED ROT":
                    color = (255, 0, 255) 
                elif status == "SLIDING":
                    color = (0, 165, 255) 
                else:
                    color = (0, 0, 255) 
                    
                cv2.putText(fpv, f"{status} (w: {w:.2f})", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.imshow("Processing FPV", fpv)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        prev_step = curr_step

    print("Offline SLAM complete! Exporting clean map and poses...")
    os.makedirs("cache", exist_ok=True)
    cv2.imwrite("cache/slam_map_walls.png", map_builder.grid) 
    
    with open("cache/frame_poses.json", "w") as f:
        json.dump(map_builder.frame_poses, f)
        
    # Extract the unique (x, y) coordinates of the surviving walls
    surviving_walls = list(set(map_builder.live_2d_walls.values()))
    with open("cache/slam_walls_realtime_cleaned.json", "w") as f:
        json.dump(surviving_walls, f)
        
    cv2.destroyAllWindows()
                 
if __name__ == "__main__":
    run_offline_slam()
