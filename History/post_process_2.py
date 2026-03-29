import json
import cv2
import numpy as np
import os
import math

# --- HELPER FUNCTIONS FOR LINE INTERSECTION ---
def ccw(A, B, C):
    """Check if three points are listed in a counter-clockwise order."""
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def segments_intersect(A, B, C, D):
    """Return True if line segment AB intersects line segment CD."""
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)
# ----------------------------------------------

def visualize_map_with_trajectory(walls_path="cache/slam_walls_realtime_cleaned.json", 
                                  poses_path="cache/frame_poses.json", 
                                  map_size=800, scale=60.0, threshold=0.6):
                                  
    if not os.path.exists(walls_path) or not os.path.exists(poses_path):
        print(f"Error: Missing JSON files in cache/. Run the SLAM script first.")
        return

    with open(walls_path, "r") as f:
        surviving_walls = json.load(f)
        
    with open(poses_path, "r") as f:
        poses_data = json.load(f)

    # --- PARAMETERS ---
    grid_size = 0.4      
    offset = 0.2         
    num_bins = 10 
    bin_size = grid_size / num_bins 

    v_lines = {} 
    h_lines = {} 

    # 1. Group the surviving points by their exact 0.4m grid segments
    for wx, wy in surviving_walls:
        snapped_x = round((wx - offset) / grid_size) * grid_size + offset
        snapped_y = round((wy - offset) / grid_size) * grid_size + offset

        if abs(wx - snapped_x) < abs(wy - snapped_y):
            y_start = math.floor((wy - offset) / grid_size) * grid_size + offset
            key = (round(snapped_x, 3), round(y_start, 3))
            if key not in v_lines: v_lines[key] = []
            v_lines[key].append(wy)
        else:
            x_start = math.floor((wx - offset) / grid_size) * grid_size + offset
            key = (round(snapped_y, 3), round(x_start, 3))
            if key not in h_lines: h_lines[key] = []
            h_lines[key].append(wx)

    valid_segments = []

    # 2. Evaluate Vertical Segments
    for (x_line, y_start), y_vals in v_lines.items():
        hit_bins = set()
        for y in y_vals:
            bin_idx = int((y - y_start) / bin_size)
            if 0 <= bin_idx < num_bins: hit_bins.add(bin_idx)
        if (len(hit_bins) / float(num_bins)) >= threshold:
            valid_segments.append(((x_line, y_start), (x_line, y_start + grid_size)))

    # 3. Evaluate Horizontal Segments
    for (y_line, x_start), x_vals in h_lines.items():
        hit_bins = set()
        for x in x_vals:
            bin_idx = int((x - x_start) / bin_size)
            if 0 <= bin_idx < num_bins: hit_bins.add(bin_idx)
        if (len(hit_bins) / float(num_bins)) >= threshold:
            valid_segments.append(((x_start, y_line), (x_start + grid_size, y_line)))

    # --- NEW LOGIC: Remove walls intersected by the trajectory ---
    
    # First, extract the world coordinates of the trajectory
    world_trajectory = []
    for img_name, pose in poses_data.items():
        rx, ry, theta = pose
        world_trajectory.append((rx, ry))

    # Next, filter out walls that the trajectory passes through
    filtered_segments = []
    for wall in valid_segments:
        wall_p1, wall_p2 = wall
        intersected = False
        
        # Check against every segment of the robot's path
        for i in range(1, len(world_trajectory)):
            traj_p1 = world_trajectory[i-1]
            traj_p2 = world_trajectory[i]
            
            if segments_intersect(traj_p1, traj_p2, wall_p1, wall_p2):
                intersected = True
                break # Move on to the next wall as soon as one collision is found
                
        if not intersected:
            filtered_segments.append(wall)

    valid_segments = filtered_segments
    # -------------------------------------------------------------

    # 4. Render Canvas & Faint Grid
    offset_x = map_size // 2
    offset_y = map_size // 2
    clean_map = np.ones((map_size, map_size, 3), dtype=np.uint8) * 255 

    grid_px = int(grid_size * scale)
    offset_px = int(offset * scale)
    grid_color = (235, 235, 235) 
    for x in range(offset_x + offset_px, map_size, grid_px): cv2.line(clean_map, (x, 0), (x, map_size), grid_color, 1)
    for x in range(offset_x - offset_px, -1, -grid_px): cv2.line(clean_map, (x, 0), (x, map_size), grid_color, 1)
    for y in range(offset_y + offset_px, map_size, grid_px): cv2.line(clean_map, (0, y), (map_size, y), grid_color, 1)
    for y in range(offset_y - offset_px, -1, -grid_px): cv2.line(clean_map, (0, y), (map_size, y), grid_color, 1)

    # 5. Draw the validated (and now filtered) walls
    for (x1, y1), (x2, y2) in valid_segments:
        px1 = int(offset_x + (x1 * scale))
        py1 = int(offset_y - (y1 * scale))
        px2 = int(offset_x + (x2 * scale))
        py2 = int(offset_y - (y2 * scale))
        cv2.line(clean_map, (px1, py1), (px2, py2), (0, 0, 0), 3)

    # 6. DRAW THE ROBOT TRAJECTORY
    trajectory_pts = []
    # poses_data is a dictionary where values are [x, y, theta]
    for img_name, pose in poses_data.items():
        rx, ry, theta = pose
        px = int(offset_x + (rx * scale))
        py = int(offset_y - (ry * scale))
        trajectory_pts.append((px, py))

    # if len(trajectory_pts) > 1:
        # Draw the blue path
        # for i in range(1, len(trajectory_pts)):
            # cv2.line(clean_map, trajectory_pts[i-1], trajectory_pts[i], (255, 0, 0), 2)
        
        # Draw a Green dot for Start, Red dot for End
        # cv2.circle(clean_map, trajectory_pts[0], 5, (0, 200, 0), -1)
        # cv2.circle(clean_map, trajectory_pts[-1], 5, (0, 0, 255), -1)

    # Save and Show
    cv2.imwrite("cache/final_map_with_trajectory_remove.png", clean_map)
    print(f"Success! Map drawn with {len(valid_segments)} walls and trajectory showing {len(trajectory_pts)} poses.")
    cv2.imshow("Final Map with Trajectory", clean_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    visualize_map_with_trajectory(threshold=0.4)