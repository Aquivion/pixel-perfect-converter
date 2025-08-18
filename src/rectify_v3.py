import sys
import numpy as np
import math
from pathlib import Path
from PIL import Image
from typing import List, Optional
import cv2


def scale_down_image_from_array(img_arr: np.ndarray, block_size: int):
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
    return scaled_img


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

    corrected_arr = rectify(arr, img_height, img_width, estimated_cell_size)

    Image.fromarray(corrected_arr, mode="RGB").save(out_path)
    print(f"Saved rectified image → {out_path}")

    scaled_img = scale_down_image_from_array(corrected_arr, estimated_cell_size)
    if out_scaled_path:
        try:
            scaled_img.save(out_scaled_path)
            print(f"✅ Scaled image saved to '{output_path}'")
        except Exception as e:
            print(f"❌ Error saving image: {e}")

    quantized_img = quantize_by_median(out_scaled_path, 8, out_unified_palette)
    if out_unified_palette:
        try:
            quantized_img.save(out_unified_palette)
            print(f"✅ Quantized image saved to '{output_path}'")
        except Exception as e:
            print(f"❌ Error saving image: {e}")


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
