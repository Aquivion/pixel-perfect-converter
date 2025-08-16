import sys
import numpy as np
import math
from pathlib import Path
from PIL import Image
from typing import List, Optional
import cv2


def scale_down_image_from_array(img_arr: np.ndarray, block_size: int, output_path: str):
    """
    Scale down the image (provided as a NumPy array) to (image_size / block_size)
    using the nearest neighbor method and save it to the specified output path.
    """
    # Convert the NumPy array to a PIL Image
    img = Image.fromarray(img_arr, mode="RGB")

    # Calculate the new dimensions
    height, width = img_arr.shape[:2]
    new_width = width // block_size
    new_height = height // block_size

    # Scale down the image using nearest neighbor method
    scaled_img = img.resize((new_width, new_height), Image.NEAREST)

    # Save the scaled image
    scaled_img.save(output_path)
    print(f"Scaled image saved to {output_path}")


def _cluster_palette(unique_cols: np.ndarray, thr: int) -> tuple[dict, np.ndarray]:
    """
    Greedy single-pass clustering: for every unique colour, either join an
    existing cluster (if any representative is within `thr`) or start a new one.

    Returns
    -------
    mapping : dict[(int,int,int) → int]    # maps a colour → cluster index
    reps    : np.ndarray[K,3]              # representative colour for each cluster
    """
    representatives: list[np.ndarray] = []
    clusters: list[list[np.ndarray]] = []
    mapping: dict[tuple[int, int, int], int] = {}

    for col in unique_cols:
        # Try to place `col` into the first cluster whose rep is close enough
        placed = False
        for idx, rep in enumerate(representatives):
            if np.linalg.norm(col - rep) <= thr:
                clusters[idx].append(col)
                mapping[tuple(col)] = idx
                placed = True
                break
        if not placed:
            representatives.append(col)
            clusters.append([col])
            mapping[tuple(col)] = len(representatives) - 1

    # Use the **median** of each cluster as final representative
    reps = np.array(
        [np.median(np.vstack(c), axis=0).astype(np.uint8) for c in clusters],
        dtype=np.uint8,
    )
    return mapping, reps


def merge_similar_colours(
    scaled_path,
    out_path,
    threshold: int = 20,
) -> None:
    img = Image.open(scaled_path).convert("RGB")
    arr = np.asarray(img)
    h, w, _ = arr.shape

    # 1. Cluster unique colours
    unique_cols = np.unique(arr.reshape(-1, 3), axis=0)
    mapping, reps = _cluster_palette(unique_cols, threshold)

    # 2. Re-map every pixel through the palette mapping
    flat = arr.reshape(-1, 3)
    out_flat = np.empty_like(flat)
    for i, col in enumerate(flat):
        rep_idx = mapping[tuple(col)]
        out_flat[i] = reps[rep_idx]

    out_arr = out_flat.reshape(h, w, 3)

    # 3. Save
    Image.fromarray(out_arr, mode="RGB").save(out_path)
    print(
        f"Merged palette → {len(reps)} colours (from {len(unique_cols)}) "
        f"and wrote {out_path}"
    )


def quantize_by_median(
    image_path: str, threshold: int = 50, output_path: Optional[str] = None
) -> Image.Image:
    """
    Reduces the number of colors in an image by grouping similar colors
    and replacing them with the median color of the group.

    The function works in several steps:
    1.  It opens the image and gets a list of all unique colors.
    2.  It iterates through the unique colors, grouping them based on a similarity
        `threshold`. Similarity is measured as the Euclidean distance in the
        RGB color space.
    3.  For each group of similar colors, it calculates the median color.
    4.  It creates a new image by swapping each original color with its group's
        calculated median color.

    Args:
        image_path (str): The path to the input PNG image.
        threshold (int): The color similarity threshold. Colors with a distance
                         less than this value will be grouped together. A higher
                         threshold leads to fewer final colors. Defaults to 50.
        output_path (Optional[str]): If provided, the resulting image will be
                                     saved to this path. Defaults to None.

    Returns:
        Image.Image: The processed PIL Image object. Returns None if the input
                     file cannot be opened.
    """
    # 1. Load the image and convert it to a NumPy array for processing
    try:
        # Ensure image is in RGB for consistent color channel handling
        img = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"❌ Error: The file at '{image_path}' was not found.")
        return None

    pixels = np.array(img, dtype=int)
    original_shape = pixels.shape

    # Flatten the image to a list of pixels (N, 3) for easier processing
    flat_pixels = pixels.reshape(-1, 3)

    # Get unique colors to significantly speed up the grouping process
    unique_colors = np.unique(flat_pixels, axis=0)

    # 2. Group similar colors using a greedy approach
    groups = []
    for color in unique_colors:
        found_group = False
        for group in groups:
            # Use the first color in the group as its representative
            representative_color = group[0]
            # Calculate Euclidean distance in the RGB space
            distance = np.linalg.norm(color - representative_color)

            if distance < threshold:
                group.append(color)
                found_group = True
                break

        if not found_group:
            # If no similar group is found, create a new one
            groups.append([color])

    # 3. Calculate the median for each group and create a color map
    color_map = {}
    print(f"🎨 Original unique colors: {len(unique_colors)}")
    print(f"🎨 Reduced to {len(groups)} color groups.")

    for group in groups:
        group_array = np.array(group)
        # Calculate the median color for the current group (per channel)
        median_color = np.median(group_array, axis=0).astype(int)

        # Map each original color in the group to the single median color
        for original_color in group:
            color_map[tuple(original_color)] = tuple(median_color)

    # 4. Apply the color map to the original flattened pixel array
    # A list comprehension is more efficient here than nested loops
    new_flat_pixels = np.array([color_map[tuple(p)] for p in flat_pixels])

    # 5. Reshape the array back to the original image dimensions
    new_pixels = new_flat_pixels.reshape(original_shape)

    # 6. Convert the NumPy array back to a PIL image
    new_img = Image.fromarray(new_pixels.astype("uint8"), "RGB")

    # 7. Optionally save the image to a file
    if output_path:
        try:
            new_img.save(output_path)
            print(f"✅ Processed image saved to '{output_path}'")
        except Exception as e:
            print(f"❌ Error saving image: {e}")

    return new_img


def rectify(
    img_arr: np.ndarray, image_height: int, image_width: int, cell_size: int
) -> np.ndarray:
    """
    Builds the corrected image in the same size of the original one:
    • Each logical pixel → exactly `cell_size`×`cell_size`
    • Colour of each block = median RGB of the original block
    """
    cells_x = math.ceil(image_width / cell_size)
    cells_y = math.ceil(image_height / cell_size)
    print("cells_x: ", cells_x)
    print("cells_y: ", cells_y)
    out = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    divider = 4

    for y in range(cells_y):
        y0, y1 = int(y * cell_size), int((y + 1) * cell_size)
        y0_inner, y1_inner = (
            int(y * cell_size + (cell_size / divider)),
            int((y + 1) * cell_size - (cell_size / divider)),
        )
        for x in range(cells_x):
            x0, x1 = int(x * cell_size), int((x + 1) * cell_size)
            x0_inner, x1_inner = (
                int(x * cell_size + (cell_size / divider)),
                int((x + 1) * cell_size - (cell_size / divider)),
            )
            block = img_arr[y0_inner:y1_inner, x0_inner:x1_inner]
            # block = img_arr[y0:y1, x0:x1]

            # print("block: ", block)

            # Use median to stay closer to the palette & reject noise
            med_color = np.median(block.reshape(-1, 3), axis=0)
            index_y = int((y + 1) * cell_size - 1)
            index_x = int((x + 1) * cell_size - 1)
            # print("index_y: ", index_y)
            # print("index_x: ", index_x)
            # med_color = img_arr[index_y][index_x]
            out[y0:y1, x0:x1] = med_color.astype(np.uint8)

    return out


def get_median_of_differences(data: List[int]) -> float:
    if len(data) < 2:
        return 0.0

    # Convert the list to a NumPy array for efficient vectorized operations
    arr = np.array(data)

    # Calculate the absolute difference between each adjacent element
    # This is done by subtracting the array from a shifted version of itself
    differences = np.abs(arr[1:] - arr[:-1])

    # Calculate and return the median of the resulting differences
    median_value = np.median(differences)

    return int(median_value)


def line_already_found(x, lines, offset) -> bool:
    for i in range(-offset, offset):
        if x + i in lines:
            # print("line: ", x + i, "already found")
            return True
    return False


def draw_edge_lines(
    arr: np.ndarray, v_lines, h_lines, image_height, image_width, output_edge_lines
) -> np.ndarray:
    line_color = (0, 0, 0)
    thickness = 1
    # Draw all vertical lines
    for x_coord in v_lines:
        start_point = (x_coord, 0)
        end_point = (x_coord, image_height)
        cv2.line(
            arr,
            start_point,
            end_point,
            line_color,
            thickness,
        )

    # Draw all horizontal lines
    for y_coord in h_lines:
        start_point = (0, y_coord)
        end_point = (image_width, y_coord)
        cv2.line(
            arr,
            start_point,
            end_point,
            line_color,
            thickness,
        )
    Image.fromarray(arr, mode="RGB").save(output_edge_lines)
    print(f"Saved rectified image → {output_edge_lines}")


def calculate_lines(
    img_arr: np.ndarray, image_height: int, image_width: int, cell_size: int
):
    block_size = 3
    threshold = 35
    jump = int(cell_size / 2)
    v_lines, h_lines = [], []
    y = 0
    shortener = 1

    while y < image_height / shortener:
        if y + block_size >= image_height:
            y += 1
            continue
        y0, y1 = y, y + block_size
        x = 0
        h_line_found = False
        while x < image_width / shortener:
            if line_already_found(x + block_size, v_lines, jump):
                x += jump
                continue
            if x + block_size >= image_width:
                x += 1
                continue

            x0, x1 = x, x + block_size
            block = img_arr[y0:y1, x0:x1]
            h_block = img_arr[y0:y1, x1 : x1 + block_size]
            v_block = img_arr[y1 : y1 + block_size, x0:x1]

            m = np.median(block.reshape(-1, 3), axis=0)
            h_m = np.median(h_block.reshape(-1, 3), axis=0)
            v_m = np.median(v_block.reshape(-1, 3), axis=0)

            h_diff = np.abs(h_m - m)
            v_diff = np.abs(v_m - m)

            h_diff_sum = h_diff.sum()
            v_diff_sum = v_diff.sum()

            if h_diff_sum > threshold:
                v_lines.append(x + block_size)
                x += jump
            else:
                x += 1

            if v_diff_sum > threshold and not h_line_found:
                h_lines.append(y + block_size)
                y += jump
                h_line_found = True
        y += 1

    # add one last vertical and horizontal line at the end and one line at the beginning
    v_lines.insert(0, 0)
    h_lines.insert(0, 0)
    v_lines.append(image_width - 1)
    h_lines.append(image_height - 1)

    v_lines.sort()
    h_lines.sort()

    return v_lines, h_lines


def rectify_v2(
    img_arr: np.ndarray,
    image_height: int,
    image_width: int,
    estimated_cell_size: int,
    output_edge_lines: str,
):

    v_lines, h_lines = calculate_lines(
        img_arr, image_height, image_width, estimated_cell_size
    )
    # print("v_lines len: ", len(v_lines))
    # print("v_lines len: ", v_lines)
    cell_size = get_median_of_differences(v_lines)
    print("Actual cellSize: ", cell_size)
    # print("h_lines len: ", len(h_lines))
    # print("h_lines: ", h_lines)
    output_image_with_lines = img_arr.copy()

    cells_x = len(v_lines) - 1
    cells_y = len(h_lines) - 1
    new_image_width = cells_x * cell_size
    new_image_height = cells_y * cell_size
    out = np.zeros((new_image_height, new_image_width, 3), dtype=np.uint8)

    for y in range(cells_y):
        y0, y1 = int(y * cell_size), int((y + 1) * cell_size)
        y_mid = (h_lines[y + 1] - h_lines[y]) / 2
        y0_inner, y1_inner = (
            int(h_lines[y] + y_mid - 1),
            int(h_lines[y] + y_mid + 2),
        )
        for x in range(cells_x):
            x0, x1 = int(x * cell_size), int((x + 1) * cell_size)
            x_mid = (v_lines[x + 1] - v_lines[x]) / 2
            x0_inner, x1_inner = (
                int(v_lines[x] + x_mid - 1),
                int(v_lines[x] + x_mid + 2),
            )
            block = img_arr[y0_inner:y1_inner, x0_inner:x1_inner]
            # block = img_arr[y0:y1, x0:x1]

            # print("block: ", block)

            # Use median to stay closer to the palette & reject noise
            med_color = np.median(block.reshape(-1, 3), axis=0)
            # print("index_y: ", index_y)
            # print("index_x: ", index_x)
            # med_color = img_arr[index_y][index_x]
            # print("med_color: ", med_color)

            out[y0:y1, x0:x1] = med_color.astype(np.uint8)

    draw_edge_lines(
        output_image_with_lines,
        v_lines,
        h_lines,
        image_height,
        image_width,
        output_edge_lines,
    )

    return out, cell_size


def main(
    in_path: str,
    out_path: str,
    out_scaled_path: str,
    out_unified_palette: str,
    size_path: str,
    output_edge_lines: str,
):
    im = Image.open(in_path).convert("RGB")
    arr = np.asarray(im)
    img_height, img_width = arr.shape[0], arr.shape[1]

    try:
        with open(size_path, "r") as f:
            estimated_cell_size = int(f.read().strip())
        print(f"Read cell_size from '{size_path}': {estimated_cell_size}")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error reading cell size from {size_path}: {e}")
        sys.exit(1)

    corrected_arr, cell_size = rectify_v2(
        arr, img_height, img_width, estimated_cell_size, output_edge_lines
    )

    Image.fromarray(corrected_arr, mode="RGB").save(out_path)
    print(f"Saved rectified image → {out_path}")

    scale_down_image_from_array(corrected_arr, cell_size, out_scaled_path)
    quantize_by_median(out_scaled_path, 8, out_unified_palette)
    # merge_similar_colours(out_scaled_path, out_unified_palette, 40)


if __name__ == "__main__":
    # --- Configuration ---
    # Set to True to process all subfolders in ../images/
    # Set to False to process only the SINGLE_IMAGE_NAME below
    BATCH_MODE = False
    SINGLE_IMAGE_NAME = "ice_tundra_v2"
    # ---------------------

    images_root = Path(__file__).parent.parent / "images"

    if BATCH_MODE:
        print("--- Running in BATCH MODE ---")
        processed_count = 0
        for image_folder in images_root.iterdir():
            if image_folder.is_dir():
                image_name = image_folder.name
                print(f"\nProcessing folder: '{image_name}'")

                input_path = image_folder / "input.png"
                size_file_path = image_folder / "size.txt"

                if not input_path.exists():
                    print(f"  - Skipping: 'input.png' not found.")
                    continue
                if not size_file_path.exists():
                    print(f"  - Skipping: 'size.txt' not found.")
                    continue

                # All required files are present, construct output paths
                output_path = image_folder / "output.png"
                output_scaled_path = image_folder / "output_scaled.png"
                output_unified_palette = image_folder / "output_palette_unified.png"
                output_edge_lines = image_folder / "output_edge_lines.png"

                print(f"  - Found input files. Processing...")
                main(
                    str(input_path),
                    str(output_path),
                    str(output_scaled_path),
                    str(output_unified_palette),
                    str(size_file_path),
                    str(output_edge_lines),
                )
                processed_count += 1
        print(
            f"\n--- Batch processing complete. Processed {processed_count} folders. ---"
        )
    else:  # Single image mode
        print(f"--- Running in SINGLE IMAGE MODE for '{SINGLE_IMAGE_NAME}' ---")
        image_name = SINGLE_IMAGE_NAME
        image_folder = images_root / image_name

        input_path = image_folder / "input.png"
        size_file_path = image_folder / "size.txt"
        output_path = image_folder / "output.png"
        output_scaled_path = image_folder / "output_scaled.png"
        output_unified_palette = image_folder / "output_palette_unified.png"
        output_edge_lines = image_folder / "output_edge_lines.png"

        main(
            str(input_path),
            str(output_path),
            str(output_scaled_path),
            str(output_unified_palette),
            str(size_file_path),
            str(output_edge_lines),
        )
