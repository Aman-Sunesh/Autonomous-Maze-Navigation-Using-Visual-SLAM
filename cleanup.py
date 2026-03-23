import os
import cv2
import numpy as np

INPUT_MAP = "cache/slam_map_walls.png"

OUTPUT_VISUAL = "cache/slam_map_walls_cleaned.png"
OUTPUT_OCCUPANCY = "cache/slam_map_occupancy.png"

BACKGROUND_GRAY = 127
WALL_BLACK = 0
FREE_WHITE = 255

MIN_WALL_AREA = 25
OPEN_KERNEL = 3
CLOSE_KERNEL = 3
INFLATION_RADIUS = 3

BOUNDARY_PAD = 2
BOUNDARY_THICKNESS = 4


def add_outer_boundary_from_bbox(wall_mask, pad=2, thickness=4):
    """
    Draw a solid rectangular boundary around the detected maze walls.
    """
    pts = cv2.findNonZero(wall_mask)
    if pts is None:
        return wall_mask

    x, y, w, h = cv2.boundingRect(pts)

    x0 = max(1, x - pad)
    y0 = max(1, y - pad)
    x1 = min(wall_mask.shape[1] - 2, x + w - 1 + pad)
    y1 = min(wall_mask.shape[0] - 2, y + h - 1 + pad)

    out = wall_mask.copy()
    cv2.rectangle(out, (x0, y0), (x1, y1), 255, thickness)
    return out


def keep_large_components(binary_mask, min_area):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    cleaned = np.zeros_like(binary_mask)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == label] = 255
    return cleaned


def flood_exterior(non_wall_mask):
    h, w = non_wall_mask.shape
    ff = non_wall_mask.copy()
    mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(ff, mask, (0, 0), 128)
    exterior = np.zeros_like(non_wall_mask)
    exterior[ff == 128] = 255
    return exterior


def skeletonize(binary):
    """
    Morphological skeletonization using only standard OpenCV.
    Input: binary mask with walls = 255
    Output: thin wall mask with walls = 255
    """
    skel = np.zeros(binary.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    img = binary.copy()

    while True:
        opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(img, opened)
        eroded = cv2.erode(img, element)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()

        if cv2.countNonZero(img) == 0:
            break

    return skel


def build_base_masks(gray_img):
    # Re-quantize to 3-color map
    quant = np.full_like(gray_img, BACKGROUND_GRAY)
    quant[gray_img < 80] = WALL_BLACK
    quant[gray_img > 180] = FREE_WHITE

    wall_mask = np.zeros_like(gray_img, dtype=np.uint8)
    wall_mask[quant == WALL_BLACK] = 255

    # First remove tiny thin noise barriers
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPEN_KERNEL, OPEN_KERNEL))
    wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_OPEN, open_kernel)

    # Then reconnect genuine wall gaps
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, close_kernel)

    # Remove leftover tiny wall fragments
    wall_mask = keep_large_components(wall_mask, MIN_WALL_AREA)

    # Force a solid outer boundary around the maze
    wall_mask = add_outer_boundary_from_bbox(
        wall_mask,
        pad=BOUNDARY_PAD,
        thickness=BOUNDARY_THICKNESS
    )

    non_wall = cv2.bitwise_not(wall_mask)
    exterior = flood_exterior(non_wall)
    interior = cv2.bitwise_and(non_wall, cv2.bitwise_not(exterior))

    return wall_mask, interior, exterior


def build_clean_visual_map(wall_mask, interior_mask):
    """
    Decent Looking Map:
    - thin black walls
    - white interior
    - gray exterior
    """
    thin_walls = skeletonize(wall_mask)

    # Optional: make skeleton slightly more visible but still thin
    thin_walls = cv2.dilate(
        thin_walls,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)),
        iterations=1
    )

    clean_visual = np.full(wall_mask.shape, BACKGROUND_GRAY, dtype=np.uint8)
    clean_visual[interior_mask > 0] = FREE_WHITE
    clean_visual[thin_walls > 0] = WALL_BLACK

    return clean_visual, thin_walls


def build_occupancy_map(wall_mask, interior_mask):
    """
    Safe map for navigation:
    - keep thick/inflated walls
    - traversable interior stays white
    """
    k = 2 * INFLATION_RADIUS + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    inflated_walls = cv2.dilate(wall_mask, kernel, iterations=1)

    occupancy = np.zeros_like(wall_mask)
    occupancy[(interior_mask > 0) & (inflated_walls == 0)] = 255
    return occupancy


def main():
    if not os.path.exists(INPUT_MAP):
        print(f"Could not find input map: {INPUT_MAP}")
        return

    gray = cv2.imread(INPUT_MAP, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        print(f"Failed to load: {INPUT_MAP}")
        return

    wall_mask, interior_mask, _ = build_base_masks(gray)
    clean_visual, thin_walls = build_clean_visual_map(wall_mask, interior_mask)
    occupancy = build_occupancy_map(wall_mask, interior_mask)

    os.makedirs("cache", exist_ok=True)
    cv2.imwrite(OUTPUT_VISUAL, clean_visual)
    cv2.imwrite(OUTPUT_OCCUPANCY, occupancy)

    print(f"Saved visual map:    {OUTPUT_VISUAL}")
    print(f"Saved occupancy map: {OUTPUT_OCCUPANCY}")

    cv2.imshow("Original", gray)
    cv2.imshow("Thin Visual Map", clean_visual)
    cv2.imshow("Occupancy Map", occupancy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()