"""
cleanup.py
-------------------------
This file takes the raw wall coordinates and frame poses produced by the offline
SLAM / mapping stage and turns them into a clean, human-readable maze map that is 
easier to navigate.

What this file does:
1. The raw SLAM output contains many noisy wall points.
2. Those points need to be grouped into proper wall segments on the maze grid.
3. Any wall that the robot trajectory clearly passes through is likely a false wall
   or mapping artifact, so we remove it here.
4. The final cleaned map is saved as an image and used later by the navigation
   system for visualization and path planning.

In simple terms, this file is the "cleanup" step between noisy mapping output and
usable final map output.
"""

import json
import cv2
import numpy as np
import os
import math

# ---------------------------------------------------------------------------
# Geometry Helpers
# ---------------------------------------------------------------------------
def ccw(A, B, C):
    """
    Return True if the three 2D points A, B, C are arranged in counter-clockwise order.

    Args:
        A, B, C: 2D points as (x, y) tuples.

    Returns:
        bool: True if the orientation is counter-clockwise, otherwise False.
    """
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def segments_intersect(A, B, C, D):
    """
    Return True if line segment AB intersects line segment CD.

    Args:
        A, B: Endpoints of the first segment.
        C, D: Endpoints of the second segment.

    Returns:
        bool: True if the two segments intersect, otherwise False.
    """
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


# ---------------------------------------------------------------------------
# Main Cleanup / Visualization Routine
# ---------------------------------------------------------------------------
def visualize_map_with_trajectory(walls_path="cache/slam_walls_realtime_cleaned.json", 
                                  poses_path="cache/frame_poses.json", 
                                  map_size=800, scale=60.0, threshold=0.6):
    """
    Build a final cleaned map image from raw SLAM wall points and frame poses.

    High-level idea:
    1. Load surviving wall points from the SLAM cleanup stage.
    2. Group those points into candidate wall segments on the known maze grid.
    3. Keep only segments that have enough support from the raw points.
    4. Remove any segment that the robot path clearly passes through.
    5. Render the final clean map and save it for later use.

    Args:
        walls_path (str): Path to JSON file containing the cleaned surviving wall
            coordinates from the SLAM stage.
        poses_path (str): Path to JSON file containing image/frame poses
            (x, y, theta) across the exploration trajectory.
        map_size (int): Output map image size in pixels.
        scale (float): Conversion factor from world meters to image pixels.
        threshold (float): Minimum fraction of occupied mini-bins needed to keep a
            wall segment as valid.

    Returns:
        None. The function saves the final map image to cache and displays it.
    """


    # We cannot build the cleaned map unless both the wall data and pose data exist.
    # If either file is missing, the SLAM stage probably has not been run yet.
    if not os.path.exists(walls_path) or not os.path.exists(poses_path):
        print(f"Error: Missing JSON files in cache/. Run the SLAM script first.")
        return

    # Load the surviving wall coordinates collected from the earlier SLAM pass.
    with open(walls_path, "r") as f:
        surviving_walls = json.load(f)

    # Load the robot poses for each frame. We use these to reconstruct the
    # trajectory and remove impossible walls later.
    with open(poses_path, "r") as f:
        poses_data = json.load(f)

    # -----------------------------------------------------------------------
    # Grid / binning parameters
    # -----------------------------------------------------------------------
    # The maze is assumed to live on a 0.4m grid with walls centered at offsets
    # of +0.2m. We use these constants to snap noisy wall points back onto the
    # intended maze structure.
    grid_size = 0.4      
    offset = 0.2  

    # Each candidate 0.4m wall segment is subdivided into many tiny bins.
    # A wall is kept only if enough of those bins received evidence from the raw
    # wall points. This avoids keeping segments created by only a few noisy points.       
    num_bins = 10 
    bin_size = grid_size / num_bins 

    # Dictionaries used to group noisy wall points into vertical and horizontal
    # candidate wall segments.
    v_lines = {} 
    h_lines = {} 


    # -----------------------------------------------------------------------
    # Group noisy points into grid-aligned wall candidates
    # -----------------------------------------------------------------------
    # For each wall point, we determine whether it is closer to a vertical or a
    # horizontal maze wall, then assign it to the corresponding 0.4m segment.
    for wx, wy in surviving_walls:
        # Snap the point to the nearest grid-aligned vertical / horizontal wall line.
        snapped_x = round((wx - offset) / grid_size) * grid_size + offset
        snapped_y = round((wy - offset) / grid_size) * grid_size + offset

        # If the point is closer to a vertical wall, group it into a vertical segment.
        if abs(wx - snapped_x) < abs(wy - snapped_y):
            y_start = math.floor((wy - offset) / grid_size) * grid_size + offset
            key = (round(snapped_x, 3), round(y_start, 3))
            if key not in v_lines: v_lines[key] = []
            v_lines[key].append(wy)

        # Otherwise, group it into a horizontal segment.
        else:
            x_start = math.floor((wx - offset) / grid_size) * grid_size + offset
            key = (round(snapped_y, 3), round(x_start, 3))
            if key not in h_lines: h_lines[key] = []
            h_lines[key].append(wx)

    valid_segments = []

    # -----------------------------------------------------------------------
    # Keep only well-supported vertical segments
    # -----------------------------------------------------------------------
    # Each candidate segment is divided into mini-bins. If enough mini-bins contain
    # observed wall points, the segment is treated as a real wall.
    for (x_line, y_start), y_vals in v_lines.items():
        hit_bins = set()
        for y in y_vals:
            bin_idx = int((y - y_start) / bin_size)
            if 0 <= bin_idx < num_bins: hit_bins.add(bin_idx)

        # Only accept the segment if the observed support is strong enough.
        if (len(hit_bins) / float(num_bins)) >= threshold:
            valid_segments.append(((x_line, y_start), (x_line, y_start + grid_size)))


    # -----------------------------------------------------------------------
    # Keep only well-supported horizontal segments
    # -----------------------------------------------------------------------
    for (y_line, x_start), x_vals in h_lines.items():
        hit_bins = set()
        for x in x_vals:
            bin_idx = int((x - x_start) / bin_size)
            if 0 <= bin_idx < num_bins: hit_bins.add(bin_idx)
        if (len(hit_bins) / float(num_bins)) >= threshold:
            valid_segments.append(((x_start, y_line), (x_start + grid_size, y_line)))


    # -----------------------------------------------------------------------
    # 5. Remove walls that the robot trajectory passes through
    # -----------------------------------------------------------------------
    # If the robot path physically crosses a wall segment, that wall cannot be real.
    # It is most likely a false wall produced by noisy perception, snapping, or
    # repeated observations from different viewpoints.
        
    # Extract the robot trajectory in world coordinates.
    world_trajectory = []
    for img_name, pose in poses_data.items():
        rx, ry, theta = pose
        world_trajectory.append((rx, ry))

    # Next, filter out walls that the trajectory passes through
    filtered_segments = []
    for wall in valid_segments:
        wall_p1, wall_p2 = wall
        intersected = False
        
        # Check the wall against every small piece of the robot trajectory.
        for i in range(1, len(world_trajectory)):
            traj_p1 = world_trajectory[i-1]
            traj_p2 = world_trajectory[i]
            
            # If the path segment intersects the wall segment, this wall is invalid.
            if segments_intersect(traj_p1, traj_p2, wall_p1, wall_p2):
                intersected = True
                break # Move on to the next wall as soon as one collision is found
                
        # Keep only walls that were never crossed by the robot path.
        if not intersected:
            filtered_segments.append(wall)

    valid_segments = filtered_segments

    # -----------------------------------------------------------------------
    # Create final visualization canvas with faint grid
    # -----------------------------------------------------------------------
    # The final map is white with a light gray background grid so the maze structure
    # is easy to read. This is only for readability; the black walls are the main
    # output used later.
    offset_x = map_size // 2
    offset_y = map_size // 2
    clean_map = np.ones((map_size, map_size, 3), dtype=np.uint8) * 255 

    grid_px = int(grid_size * scale)
    offset_px = int(offset * scale)
    grid_color = (235, 235, 235) 

    # Draw vertical faint grid lines to show the maze cell structure.
    for x in range(offset_x + offset_px, map_size, grid_px): cv2.line(clean_map, (x, 0), (x, map_size), grid_color, 1)
    for x in range(offset_x - offset_px, -1, -grid_px): cv2.line(clean_map, (x, 0), (x, map_size), grid_color, 1)

    # Draw horizontal faint grid lines.
    for y in range(offset_y + offset_px, map_size, grid_px): cv2.line(clean_map, (0, y), (map_size, y), grid_color, 1)
    for y in range(offset_y - offset_px, -1, -grid_px): cv2.line(clean_map, (0, y), (map_size, y), grid_color, 1)

    # -----------------------------------------------------------------------
    # Draw the validated wall segments
    # -----------------------------------------------------------------------
    # At this stage, valid_segments contains only strong, trajectory-consistent wall
    # segments. These are drawn as bold black lines on the final clean map.
    for (x1, y1), (x2, y2) in valid_segments:
        px1 = int(offset_x + (x1 * scale))
        py1 = int(offset_y - (y1 * scale))
        px2 = int(offset_x + (x2 * scale))
        py2 = int(offset_y - (y2 * scale))
        cv2.line(clean_map, (px1, py1), (px2, py2), (0, 0, 0), 3)

    # -----------------------------------------------------------------------
    # Reconstruct the trajectory in pixel coordinates
    # -----------------------------------------------------------------------
    # We still compute the trajectory points here because they are useful for logging,
    # debugging, and optional visualization, even though we do not currently draw the
    # path on the final map.
    trajectory_pts = []
    # poses_data is a dictionary where values are [x, y, theta]
    for img_name, pose in poses_data.items():
        rx, ry, theta = pose
        px = int(offset_x + (rx * scale))
        py = int(offset_y - (ry * scale))
        trajectory_pts.append((px, py))

    # -----------------------------------------------------------------------
    # Save and show final result
    # -----------------------------------------------------------------------
    # The final cleaned image is saved into cache because the navigation code loads
    # it later as its visual / planning map.
    cv2.imwrite("cache/slam_map_walls_cleaned.png", clean_map)
    print(f"Success! Map drawn with {len(valid_segments)} walls and trajectory showing {len(trajectory_pts)} poses.")
    cv2.imshow("Final Map with Trajectory", clean_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Run the cleanup stage using the default cached inputs.
if __name__ == "__main__":
    # 'threshold' controls how much wall evidence is required to keep a wall segment.
    # Higher threshold = stricter wall filtering, lower threshold = more walls kept.
    visualize_map_with_trajectory(threshold=0.6)