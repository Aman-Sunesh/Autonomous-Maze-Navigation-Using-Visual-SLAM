"""
offline_slam.py
-------------------------
This file acts as the main pre-navigation mapping engine. It taks the raw data
of robot control commands and camera images, processes them frame-by-frame, 
and generates a rough 2D map of the environment.

What this file does:
1. Translates recorded velocity commands into Odometry to estimate where the 
   robot should be at any given time.
2. Uses a pinhole camera model (Wall Detector) to find the boundary between 
   the sky and the walls, converting 2D pixels into 3D camera coordinates,
   then into 2D map coordinates.
3. Applies a Manhattan Constraint to analyze the geometry of the walls it sees, 
   gently snapping the robot's heading and position back onto a perfect 90-degree grid.
4. Analyzes consecutive camera frames (Visual Stuck Detector) to catch when the 
   robot is sliding, blocked, or stuck against a wall, preventing odometry drift.
5. Exports the raw wall coordinates and corrected robot poses into the cache directory.

In short, this file generates the map using the exploration data.
"""
import os
import json
import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Constants and Parameters
# * indicates important paramameter for tunning
# ---------------------------------------------------------------------------
# Data paths
DATA_INFO_PATH = "data/data_info.json"
IMAGE_DIR = "data/exploration_data/traj_0"

# Odometry parameters
BASE_V = 2.9462             # linear velocity
BASE_W = 4.27               # angular velocity

# Camera infos
CAMERA_W = 320
CAMERA_H = 240
CAMERA_F = 92
K = np.array([[CAMERA_F, 0, CAMERA_W/2.0],
              [0, CAMERA_F, CAMERA_H/2.0],
              [0, 0, 1]])

# Wall Detector parameters
CAMERA_HEIGHT = 0.21
WALL_HEIGHT = 0.30
SCAN_BINS = 60              # increase: detect more walls
MAX_DEPTH = 0.3             # increase: see more depth

# Map Builder parameters
GRID_M = 0.4                # set grid size as 0.4m x 0.4m
NEAR_WALL_RADIUS = 0.06     # increase*: larger radius to consider robot near a wall

# Manhattan Constraint parameters
STEP_SIZE = 3               # increase: use more points to smooth out candidate angles
NUM_BINS = 36               # increase: use more bins to find the dominant wall angle
ANGLE_THRESH = 15           # increase: detect more candidate angles for manhattan fix (less rigorous)
HEADING_GAIN = 0.4          # increase*: stronger direction fix
TRANSLATION_GAIN = 0.1      # increase*: stronger position fix
MATCH_THRESH = 0.18         # increase*: larger area to snap position fix

# Visual Stuck Detector parameters
CONF_THRESH = 0.05          # increase: ignore more uncertain shifts
DIFF_THRESH = 3.0           # increase: easier to find "STUCK"
SLIDE_THRESH = 5.0          # increase: harder to find "SLIDING"
SLOW_ANGLE_THRESH = 0.6     # increase: easier to find "BLOCKED ROT"


# ---------------------------------------------------------------------------
# Mapping Models
# ---------------------------------------------------------------------------
class Odometry:
    """
    Tracks and updates the robot's estimated physical state (x, y, heading) 
    in a global 2D coordinate frame based on velocity commands.
    """
    def __init__(self, initial_x=0.0, initial_y=0.0, initial_theta=0.0):
        """
        Initializes the robot's starting state in the world frame.
        x, y are in meters. theta is the heading in radians.
        """
        self.x = initial_x
        self.y = initial_y
        self.theta = initial_theta

    def update(self, v, w, dt):
        """
        Updates the robot's pose (x, y, theta) using basic kinematics 
        based on linear velocity (v), angular velocity (w), and the time step (dt).
        """
        # Error handler: negative time step
        if dt <= 0: 
            return
        
        self.x += v * dt * np.cos(self.theta)
        self.y += v * dt * np.sin(self.theta)
        self.theta += w * dt
        self.theta = (self.theta + np.pi) % (2 * np.pi) - np.pi # Normalize theta to stay within -pi to pi

class WallDetector:
    """
    Processes the camera images to visually detect walls 
    and calculate their 2D coordinates relative to the robot.
    """
    def __init__(self, K, camera_height=CAMERA_HEIGHT, wall_height=WALL_HEIGHT, max_depth=MAX_DEPTH, viz=False): 
        """
        Initializes the camera intrinsics and simulation environment parameters
        """
        self.K = K                      # camera intrinsics
        self.cam_h = camera_height      # height of the camera lens from the floor in meters
        self.wall_h = wall_height       # height of the walls in meters
        self.max_depth = max_depth      # Maximum distance to calculate the depth in meters
        self.viz = viz                  # visualize blurred frame

        # Extract lens properties from the K matrix
        self.fx = K[0, 0]
        self.fy = K[1, 1]
        self.cx = K[0, 2]
        self.cy = K[1, 2]

    def extract_wall_points(self, fpv):
        """
        Takes an image (fpv), finds the top edges of walls by isolating the sky background, 
        and calculates exactly how far away those walls are in camera frame using the pinhole camera model.
        Converts from the camera coordinate frame to the current robot's local coordinate frame
        Returns a list of wall coordinates and a list of open space rays.
        """
        h, w = fpv.shape[:2]
        
        # DEBUG: visually check the wall detection
        if self.viz:
            debug_img = fpv.copy()

        gray = cv2.cvtColor(fpv, cv2.COLOR_BGR2GRAY)
        # Find the sky mask by keeping only white pixel regions
        _, sky_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
                
        # Store the calculated (x, y) wall/open space points relative to the robot
        local_walls = []
        local_open_space = []

        # Determine how many vertical columns to scan for speed
        step = max(1, w // SCAN_BINS)
        
        # Top-down raycasting to extract the top edge of a wall
        for u in range(step // 2, w, step):         # u = current pixel column: scan the center of the bin
            found_wall = False
            # If the very top pixel is already a wall, the top is off-screen.
            if sky_mask[0, u] == 0:
                continue    # skip

            for v in range(0, int(self.cy)):        # v = current pixel row: scan from top to bottom
                
                # If the pixel is not part of the sky mask, it is the top edge of a wall
                if sky_mask[v, u] == 0:
                    found_wall = True
                    if self.viz:
                        cv2.circle(debug_img, (u, v), 2, (0, 0, 255), -1)
                    
                    # Calculate depth Z from camera model:
                    # y = (fy * Y) / Z where y = cy - v
                    # Z = (fy * Y) / y
                    Y = self.wall_h - self.cam_h 
                    denominator = max((self.cy - v), 1) 
                    Z_cam = (self.fy * Y) / denominator
                    
                    # Fallback: ignore if the wall is too far or too close.
                    if Z_cam > self.max_depth or Z_cam <= 0.1:
                        break
                    
                    # Calculate X from camera model:
                    # x = (fx * X) / Z where x = u - cx
                    # X = (x * Z) / fx
                    X_cam = ((u - self.cx) * Z_cam) / self.fx
                    
                    # Convert from the pinhole camera coordinate frame to the current robot's local coordinate frame
                    X_local, Y_local = Z_cam, -X_cam
                    
                    local_walls.append((X_local, Y_local))
                    break
                
            # The loop finished without finding a wall
            if not found_wall:
                Z_cam = self.max_depth
                X_cam = ((u - self.cx) * Z_cam) / self.fx
                X_local, Y_local = Z_cam, -X_cam
                
                # Store in the dedicated open space list
                local_open_space.append((X_local, Y_local))
        
        if self.viz:
            # Show the exact wall points
            cv2.imshow("Extracted Wall Edges", debug_img)

        return local_walls, local_open_space

class MapBuilder:
    """
    Manages the 2D grid map, drawing the robot's trajectory and the detected walls, 
    and clearing out false-positive walls using open space rays.
    """
    def __init__(self, map_size=800, scale=60.0):
        """
        Initializes the visual map canvas and draws the background grid.
        """
        self.map_size = map_size
        self.scale = scale 

        # Place the robot at the center of the map
        self.offset_x = map_size // 2
        self.offset_y = map_size // 2
        
        # Create a blank gray canvas
        self.grid = np.ones((self.map_size, self.map_size, 3), dtype=np.uint8) * 127
        
        # Convert grid in meters to pixels
        grid_px = int(GRID_M * self.scale)

        # Draw the background grid starting exactly at the absolute origin
        grid_color = (100, 100, 100)
        for x in range(self.offset_x, self.map_size, grid_px):  # center to right edges
            cv2.line(self.grid, (x, 0), (x, self.map_size), grid_color, 1)
        for x in range(self.offset_x, -1, -grid_px):            # center to left edges
            cv2.line(self.grid, (x, 0), (x, self.map_size), grid_color, 1)
        for y in range(self.offset_y, self.map_size, grid_px):  # center to bottom edges
            cv2.line(self.grid, (0, y), (self.map_size, y), grid_color, 1)
        for y in range(self.offset_y, -1, -grid_px):            # center to top edges
            cv2.line(self.grid, (0, y), (self.map_size, y), grid_color, 1)

        # Keep track of map states
        self.trajectory = []
        self.frame_poses = {} 
        self.frame_walls = {}
        self.live_walls = {} 

    def world_to_pixel(self, x, y):
        """
        Converts simulation coordinates (meters) to map coordinates (pixels).
        """
        px = int(self.offset_x + (x * self.scale))
        py = int(self.offset_y - (y * self.scale)) 
        return px, py
    
    def transform_local_to_global(self, local_walls, local_open_space, robot_x, robot_y, robot_theta):
        """
        Translates lists of local 2D coordinates relative to the robot.
        into absolute global map coordinates using the robot's current pose.
        """
        global_wall_points = []
        global_open_points = []
        
        # Convert from local wall points into global wall points
        for X_local, Y_local in local_walls:
            X_world = robot_x + (X_local * np.cos(robot_theta) - Y_local * np.sin(robot_theta))
            Y_world = robot_y + (X_local * np.sin(robot_theta) + Y_local * np.cos(robot_theta))
            global_wall_points.append((X_world, Y_world))

        # Converts from local open space points into global open space points
        for X_local, Y_local in local_open_space:
            X_world = robot_x + (X_local * np.cos(robot_theta) - Y_local * np.sin(robot_theta))
            Y_world = robot_y + (X_local * np.sin(robot_theta) + Y_local * np.cos(robot_theta))
            global_open_points.append((X_world, Y_world))
                
        return global_wall_points, global_open_points

    def update_walls(self, robot_x, robot_y, global_wall_points, global_open_points, img_filename=None):
        """
        Plots wall points onto the map grid.
        Utilizes raycasting against open space points to erase false-positive walls.
        """
        # Log the global wall points for this specific image frame
        if img_filename:
            self.frame_walls[img_filename] = global_wall_points

        # Get current robot position in pixel coordinates
        rpx, rpy = self.world_to_pixel(robot_x, robot_y)
        
        # Create a blank mask to track the clearing rays
        ray_mask = np.zeros((self.map_size, self.map_size), dtype=np.uint8)

        # Draw rays into open space to identify empty areas
        for ox, oy in global_open_points:
            opx, opy = self.world_to_pixel(ox, oy)
            if 0 <= opx < self.map_size and 0 <= opy < self.map_size and 0 <= rpx < self.map_size and 0 <= rpy < self.map_size:
                cv2.line(ray_mask, (rpx, rpy), (opx, opy), 255, 2)
                cv2.line(self.grid, (rpx, rpy), (opx, opy), (255, 255, 255), 2)

        # Process the wall points
        snapped_walls = []
        for wx, wy in global_wall_points:
            # Convert wall coordinates in meters to map coordinates in pixels
            wpx, wpy = self.world_to_pixel(wx, wy)
            
            # Ensure the calculated pixel coordinates fall within the map boundaries
            if 0 <= wpx < self.map_size and 0 <= wpy < self.map_size and 0 <= rpx < self.map_size and 0 <= rpy < self.map_size:
                snapped_walls.append((wx, wy, wpx, wpy))
                
                # Draw raycasts to the physical walls
                cv2.line(ray_mask, (rpx, rpy), (wpx, wpy), 255, 1)
                cv2.line(self.grid, (rpx, rpy), (wpx, wpy), (255, 255, 255), 1)

        # Erase old walls that the rays passed through
        erased_y, erased_x = np.where(ray_mask == 255)
        for px, py in zip(erased_x, erased_y):
            self.live_walls.pop((px, py), None)

        # Draw the wall points onto the map
        for wx, wy, wpx, wpy in snapped_walls:
            # Draw a black circle at each recorded wall location
            cv2.circle(self.grid, (wpx, wpy), 1, (0, 0, 0), -1)
            
            # Register the wall coordinates in the live map dictionary 
            self.live_walls[(wpx, wpy)] = (wx, wy)

    def is_near_wall(self, robot_x, robot_y, radius=NEAR_WALL_RADIUS):
        """
        Creates a bounding box around the robot's current position to check if 
        any previously drawn wall pixels exist within a specific radius.
        """
        rpx, rpy = self.world_to_pixel(robot_x, robot_y)
        px_radius = int(radius * self.scale)
        
        # Calculate bounding box coordinates and clampe to grid edges to handle index errors
        y1 = max(0, rpy - px_radius)
        y2 = min(self.grid.shape[0], rpy + px_radius)
        x1 = max(0, rpx - px_radius)
        x2 = min(self.grid.shape[1], rpx + px_radius)
        
        # Extract the small local patch around the robot
        local_map = self.grid[y1:y2, x1:x2]

        # Check if any pixel in this local patch is pure black [0, 0, 0] (a drawn wall)
        return np.any(np.all(local_map == [0, 0, 0], axis=-1))

    def draw(self, robot_x, robot_y, theta, img_filename=None, current_walls=None):
        """
        Draws the robot's current position in red dot, heading in green line, 
        and historical trajectory in blue line onto the map display.
        """
        rpx, rpy = self.world_to_pixel(robot_x, robot_y)  # Get current robot position in pixel coordinates
        self.trajectory.append((rpx, rpy))                # append to the trajectory history              
        
        # Log the robot's pose for each image frame
        if img_filename:
            self.frame_poses[img_filename] = (float(robot_x), float(robot_y), float(theta))

        display_img = self.grid.copy()

        # DEBUG: show current walls in red dots
        if current_walls:
            for wx, wy in current_walls:
                wpx, wpy = self.world_to_pixel(wx, wy)
                if 0 <= wpx < self.map_size and 0 <= wpy < self.map_size:
                    cv2.circle(display_img, (wpx, wpy), 2, (0, 0, 255), -1)

        # Draw the trajectory history line in blue
        if len(self.trajectory) > 1:
            for i in range(1, len(self.trajectory)):
                cv2.line(display_img, self.trajectory[i-1], self.trajectory[i], (255, 0, 0), 2)

        # Draw the current robot position in red dot
        cv2.circle(display_img, (rpx, rpy), 5, (0, 0, 255), -1)

        # Calculate the endpoint (hx, hy) for line to show the robot's heading.
        heading_size = 15
        hx = int(rpx + heading_size * np.cos(theta))
        hy = int(rpy - heading_size * np.sin(theta)) 
        # Draw the robot's heading in green line
        cv2.line(display_img, (rpx, rpy), (hx, hy), (0, 200, 0), 2)
               
        cv2.imshow("SLAM Output", display_img)

class ManhattanConstraint:
    """
    Corrects odometry drift by analyzing detected walls and forcing linear segments 
    to snap to perfect 90-degree grids using a Manhattan world assumption.
    """
    def __init__(self, step_size=STEP_SIZE, num_bins=NUM_BINS, angle_thresh=ANGLE_THRESH,
                 heading_gain=HEADING_GAIN, translation_gain=TRANSLATION_GAIN, 
                 grid_m=GRID_M, match_thresh=MATCH_THRESH, viz=False):
        """
        Initializes parameters for Manhattan world constraints.
        """
        self.step_size = step_size
        self.num_bins = num_bins
        self.angle_thresh = angle_thresh
        self.heading_gain = heading_gain
        self.translation_gain = translation_gain
        self.grid_m = grid_m
        self.match_thresh = match_thresh
        self.viz = viz

    def correct_odometry(self, local_walls, odom, map_builder, historic_wall_centers):
        """
        Analyzes the local wall points to vote on the dominant wall direction.
        If a strong linear wall is found, gently adjusts the robot's heading and position toward the ideal grid.
        """
        if len(local_walls) > 15:   # set threshold of 15 for robustness
            pts = np.array(local_walls, dtype=np.float32)   # Convert the list of (X, Y) tuples into a 2D NumPy array for fast math
            
            # Determine how many points to calcuate the angles for smoothness
            # Calculate rise and run between point N and point N+step_szie
            dx = pts[self.step_size:, 0] - pts[:-self.step_size, 0]
            dy = pts[self.step_size:, 1] - pts[:-self.step_size, 1]
            # Calculate the directional angle of every segment
            local_angles = np.arctan2(dy, dx)
            
            # Voting system to figure out the true direction of the wall
            # Build histrogram to store the vote count for each bins
            dominant_bin = np.argmax(np.histogram(local_angles, bins=self.num_bins, range=(-np.pi, np.pi))[0])
            hist, bin_edges = np.histogram(local_angles, bins=self.num_bins, range=(-np.pi, np.pi))
            dominant_local_angle = (bin_edges[dominant_bin] + bin_edges[dominant_bin+1]) / 2.0
            
            # Calculate the angular distance between each segment and the dominant direction
            angle_diffs = np.abs(np.arctan2(np.sin(local_angles - dominant_local_angle), 
                                            np.cos(local_angles - dominant_local_angle)))
            # Create an inlier mask: inlier means the segment is a part of the main wall.
            inlier_mask_segments = angle_diffs < np.deg2rad(self.angle_thresh)
            

            if np.sum(inlier_mask_segments) > 5:    # set threshold of 15 for robustness
                # 1. Fix rotation based on the Manhattan constraints
                # Calculate the average angle of the wall
                average_local_angle = np.arctan2(np.mean(np.sin(local_angles[inlier_mask_segments])), 
                                                 np.mean(np.cos(local_angles[inlier_mask_segments])))
                global_wall_angle = odom.theta + average_local_angle    # Convert the wall's local angle into a global map angle
                # Snap the calculated global angle to the nearest perfect 90-degree (pi/2) increment
                target_global_angle = np.round(global_wall_angle / (np.pi/2.0)) * (np.pi/2.0)
                
                # Calculate the error between the ideal robot's heading and the current robot's heading
                heading_error = target_global_angle - global_wall_angle
                # Gently adjust the current heading toward the ideal heading
                odom.theta += heading_error * self.heading_gain
                odom.theta = (odom.theta + np.pi) % (2 * np.pi) - np.pi # Normalize theta to stay within -pi to pi
                
                # 2. Fix position based on the Manhattan constraints
                # Match the length of the inlier mask to the length of the wall points
                inlier_mask_pts = np.append(inlier_mask_segments, [False]*self.step_size)
                valid_local_pts = pts[inlier_mask_pts]
                
                if len(valid_local_pts) > 0:
                    # Convert the wall points' local position into a global map position
                    global_X = odom.x + (valid_local_pts[:, 0] * np.cos(odom.theta) - valid_local_pts[:, 1] * np.sin(odom.theta))
                    global_Y = odom.y + (valid_local_pts[:, 0] * np.sin(odom.theta) + valid_local_pts[:, 1] * np.cos(odom.theta))
                    
                    # Find the center of the wall
                    mean_wall_x = np.mean(global_X)
                    mean_wall_y = np.mean(global_Y)
                    
                    # DEBUG: visually check the detected wall position
                    if self.viz:
                        historic_wall_centers.append((mean_wall_x, mean_wall_y))
                        cx, cy = map_builder.world_to_pixel(mean_wall_x, mean_wall_y)
                        cv2.circle(map_builder.grid, (cx, cy), 3, (255, 0, 255), -1)
                    
                    # Normalize target angle to stay within -pi to pi
                    norm_target = (target_global_angle + np.pi) % (2 * np.pi) - np.pi
                    # Classify the wall as horizontal or vertical
                    is_horizontal = np.isclose(abs(norm_target), 0.0, atol=0.1) or np.isclose(abs(norm_target), np.pi, atol=0.1)
                    is_vertical = np.isclose(abs(norm_target), np.pi/2.0, atol=0.1)
                    
                    if is_horizontal:
                        # Snap the wall to the nearest absolute 0.4m grid line
                        target_y = np.round(mean_wall_y / self.grid_m) * self.grid_m
                        if abs(target_y - mean_wall_y) < self.match_thresh:
                            # Gently adjust the robot's y position toward the idead y position
                            odom.y += (target_y - mean_wall_y) * self.translation_gain
                                
                    elif is_vertical:
                        # Snap the wall to the nearest absolute 0.4m grid line
                        target_x = np.round(mean_wall_x / self.grid_m) * self.grid_m
                        if abs(target_x - mean_wall_x) < self.match_thresh:
                            # Gently adjust the robot's x position toward the idead x position
                            odom.x += (target_x - mean_wall_x) * self.translation_gain

    def snap_wall_points(self, global_wall_points):
        """
        Forces all wall points to snap perfectly onto the grid axes to clean up drawing.
        """
        snapped_points = []
        for wx, wy in global_wall_points:           
            # Find the absolute nearest perfect grid lines for both X and Y direction.
            snapped_x = round(wx / self.grid_m) * self.grid_m
            snapped_y = round(wy / self.grid_m) * self.grid_m
            
            # If the nearest vertical grid is closer than the nearest horizontal grid,
            if abs(wx - snapped_x) < abs(wy - snapped_y):
                # Force X to the perfect grid line, leave Y alone.
                wx = snapped_x
            # Otherwise,
            else:
                # Force Y to the perfect grid line, leave X alone.
                wy = snapped_y

            snapped_points.append((wx, wy))
            
        return snapped_points

class VisualStuckDetector:
    """
    Determine if the robot is successfully moving, sliding sideways, rotating with blocking, or physically stuck against a wall.
    """
    def __init__(self, fx, diff_thresh=DIFF_THRESH, slide_thresh=SLIDE_THRESH, conf_thresh=CONF_THRESH, slow_angle_thresh=SLOW_ANGLE_THRESH):
        """
        Initializes threshold parameters for pixel differences, sliding detection, 
        and phase correlation confidence to tune the detector's sensitivity.
        """
        self.fx = fx
        self.diff_thresh = diff_thresh      
        self.slide_thresh = slide_thresh    
        self.conf_thresh = conf_thresh      
        self.slow_angle_thresh = slow_angle_thresh    
        self.prev_gray = None

    def filter_actions(self, curr_img, commanded_v, commanded_w, dt, is_near_wall):
        """
        Compares the robot's commanded velocities against its visually observed velocities. 
        If an obstruction is detected, overrides the commands with zero or calculated velocities.
        """
        # Convert to grayscale and to 32-bit floats for cv2.phaseCorrelate
        curr_gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
        curr_float = np.float32(curr_gray)

        # Initialization check
        if self.prev_gray is None:
            self.prev_gray = curr_float
            return commanded_v, commanded_w, "MOVING", 0.0, 0.0

        # Idle check
        if commanded_v == 0.0 and commanded_w == 0.0:
            self.prev_gray = curr_float
            return 0.0, 0.0, "IDLE", 0.0, 0.0
        
        # Check if near wall
        if not is_near_wall:
            return commanded_v, commanded_w, "MOVING", 0.0, 0.0

        # Calculate the pixel shift between the previous and current frame
        shift, confidence = cv2.phaseCorrelate(self.prev_gray, curr_float)
        dx, dy = shift      # unpack the shift into horizontal and vertical

        # Ignore the calculated shift if low confidence
        if confidence < self.conf_thresh:
            dx = 0.0

        # Calculate the pixel difference between the two frames
        diff = cv2.absdiff(self.prev_gray, curr_float)
        mean_diff = float(np.mean(diff))    # average all those pixel differences

        # Save the current gray image to calculate for the next pixel shift
        self.prev_gray = curr_float

        # Calculate the angle from camera model:
        # yaw = = arctan(dx / f)
        delta_yaw = np.arctan(dx / self.fx)
        visual_w = delta_yaw / dt

        # 1. Command was 'FORWARD' or 'BACKWARD'
        if commanded_w == 0.0:
            # If the horizontal shift above the slide threshold, the robot is slipping sideways
            if abs(dx) > self.slide_thresh and abs(dx) < 70:    # upper bound to remove false detection
                return 0.0, visual_w, "SLIDING", mean_diff, dx
        # 2. Command was 'LEFT' or 'RIGHT'
        else:
            # Considering the noise, check if the visually measured angular velocity is less than thereshold
            if abs(visual_w) < (abs(commanded_w) * self.slow_angle_thresh):
                return 0.0, visual_w, "BLOCKED ROT", mean_diff, dx
            
        # Check if not sliding or blocked but the scene is static
        if mean_diff < self.diff_thresh:
            return 0.0, 0.0, "STUCK", mean_diff, dx

        # Otherwise, execute normally
        return commanded_v, commanded_w, "MOVING", mean_diff, dx


# ---------------------------------------------------------------------------
# Prenavigation Logic
# ---------------------------------------------------------------------------
def parse_actions(actions_list):
    """
    Parses textual control inputs such as 'FORWARD' and 'LEFT' from the dataset 
    into numerical linear velocity (v) and angular velocity (w).
    """
    v, w = 0.0, 0.0
    
    # Set linear velocity (v)
    if 'FORWARD' in actions_list: v += BASE_V
    if 'BACKWARD' in actions_list: v -= BASE_V

    # Set angular velocity (w)
    if 'LEFT' in actions_list: w += BASE_W
    if 'RIGHT' in actions_list:  w -= BASE_W
        
    return v, w

def run_slam():
    """
    Loads a dataset, calculates time steps, parses velocity commands, extracts visual elements (Wall Detector),
    applies optimization (Manhattan Constraint and Visual Stuck Detector), and updates the robot's map and pose.
    """

    # Initialize Odometry, Map Builder, Wall Detector, Manhattan Constraint, Visual Stuck Detector
    odom = Odometry(initial_x=0.2, initial_y=0.2, initial_theta=np.pi/2)    # spawn robot in the middle of the hallway
    map_builder = MapBuilder(map_size=800, scale=60.0)
    wall_detector = WallDetector(K, viz=False)
    manhattan_constraint = ManhattanConstraint()
    stuck_detector = VisualStuckDetector(fx=K[0, 0])
    debug_mode = False                                                      # debug mode: process frame each time a key is pressed

    # Error handler while loading
    if not os.path.exists(DATA_INFO_PATH):
        print(f"Cannot find {DATA_INFO_PATH}!")
        return

    with open(DATA_INFO_PATH) as f:
        raw_data = json.load(f)

    # DEBUG: check if loaded the whole dataset
    print(f"Loaded {len(raw_data)} frames. Starting SLAM.")

    # Keep track of the previous time step to calculate how much time has passed
    prev_step = None
    historic_wall_centers = [] 
    
    for frame in raw_data:
        curr_step = frame['step']
        actions = frame['action']
        img_filename = frame.get('image', None)

        # Calculate the time step (dt): set each step is 0.01 second.
        dt = 0.01 if prev_step is None else (curr_step - prev_step) * 0.01
        
        # Parse the text command into linear and angular velocity.
        cmd_v, cmd_w = parse_actions(actions)
        
        fpv = None
        v, w = cmd_v, cmd_w
        status = "N/A"

        # If a camera image exists for this frame, load it
        if img_filename:
            img_path = os.path.join(IMAGE_DIR, img_filename)
            if os.path.exists(img_path):
                fpv = cv2.imread(img_path)

        local_walls, local_open_space = [], []

        # Detect walls from the Camera 
        if fpv is not None:
            # Extract walls and open space from the current camera view early to aid stuck detector
            local_walls, local_open_space = wall_detector.extract_wall_points(fpv)
            
            # Ask the map if we are near a wall
            is_near_wall = map_builder.is_near_wall(odom.x, odom.y)

            # Filter incorrect commanded velocities 
            v, w, status, mean_diff, dx = stuck_detector.filter_actions(fpv, cmd_v, cmd_w, dt, is_near_wall)

        # Localize the robot using the velocities and time step
        odom.update(v, w, dt)
        
        if fpv is not None:
            # Correct the odometry drift using the new ManhattanConstraint class
            manhattan_constraint.correct_odometry(local_walls, odom, map_builder, historic_wall_centers)

            est_x, est_y, est_theta = odom.x, odom.y, odom.theta
            
            # Translate local scans to global map coordinates
            global_wall_points, global_open_points = map_builder.transform_local_to_global(local_walls, local_open_space, est_x, est_y, est_theta)

            # Snap walls to the Manhattan grid to clean up the drawing
            snapped_wall_points = manhattan_constraint.snap_wall_points(global_wall_points)

            # Update map with the detected walls
            map_builder.update_walls(est_x, est_y, snapped_wall_points, global_open_points, img_filename)
            
            # DEBUY: overlay info onto the image
            cv2.putText(fpv, f"Img: {img_filename}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(fpv, f"Cmd: {actions} | Status: {status}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.imshow("Processing FPV", fpv)
        
        # Update the map builder
        map_builder.draw(odom.x, odom.y, odom.theta, img_filename)
        
        # Wait for the key press
        delay = 0 if debug_mode else 1
        key = cv2.waitKey(delay) & 0xFF
        
        if key == ord('q'):     # quit
            break
        elif key == ord('a'):   # toggle debug mode
            debug_mode = not debug_mode

        # Save the current time step to calculate for the next time step
        prev_step = curr_step

    # Export the SLAM output
    print("SLAM complete. Exporting map and poses.")
    os.makedirs("cache", exist_ok=True)

    # Save a picture of the final map
    cv2.imwrite("cache/slam_map.png", map_builder.grid)  

    # Save the dictionary that links image frames to their exact 2D coordinates (x, y, theta)
    with open("cache/frame_poses.json", "w") as f:
        json.dump(map_builder.frame_poses, f)
    
    # Save the coordinates of all final walls that survived the raycasting cleanup
    surviving_walls = list(set(map_builder.live_walls.values()))
    with open("cache/slam_walls.json", "w") as f:
        json.dump(surviving_walls, f)
        
    cv2.destroyAllWindows()
                 
if __name__ == "__main__":
    run_slam()
