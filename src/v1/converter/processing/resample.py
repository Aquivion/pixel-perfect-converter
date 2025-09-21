import numpy as np
import cv2


def create_pixel_art(
    bgr_image: np.ndarray,
    completed_x: list[float],
    completed_y: list[float],
    sample_size: int = 3,
) -> np.ndarray:
    """
    Creates a downscaled, pixel-perfect image from the original image and a completed grid.

    Iterates through each grid cell, finds its center in the original image,
    samples a small area to determine the cell's median color, and assigns that
    color to the corresponding pixel in the new output image.

    Args:
        bgr_image: The source BGR image (after any initial scaling).
        completed_x: A sorted list of the final X-coordinates of vertical grid lines.
        completed_y: A sorted list of the final Y-coordinates of horizontal grid lines.
        sample_size: The width/height of the square area to sample at the center of each cell.

    Returns:
        A new NumPy array representing the final, clean pixel art image.
    """
    if (
        not completed_x
        or not completed_y
        or len(completed_x) < 2
        or len(completed_y) < 2
    ):
        return np.zeros((1, 1, 3), dtype=np.uint8)

    num_cells_x = len(completed_x) - 1
    num_cells_y = len(completed_y) - 1

    # Create the new array for the final pixel art image
    pixel_art_array = np.zeros((num_cells_y, num_cells_x, 3), dtype=np.uint8)

    for y_idx in range(num_cells_y):
        y_start, y_end = completed_y[y_idx], completed_y[y_idx + 1]

        for x_idx in range(num_cells_x):
            x_start, x_end = completed_x[x_idx], completed_x[x_idx + 1]

            # Find the center of the cell in the original image
            x_center = (x_start + x_end) / 2
            y_center = (y_start + y_end) / 2

            # Define a small inner sampling box to avoid edge artifacts
            s_half = sample_size // 2
            sample_x0 = max(0, int(x_center - s_half))
            sample_x1 = min(bgr_image.shape[1], int(x_center + s_half) + 1)
            sample_y0 = max(0, int(y_center - s_half))
            sample_y1 = min(bgr_image.shape[0], int(y_center + s_half) + 1)

            sample_block = bgr_image[sample_y0:sample_y1, sample_x0:sample_x1]

            if sample_block.size == 0:
                # Fallback for zero-sized slices
                median_color = np.array([0, 0, 0])
            else:
                # Calculate the median color of the sample area
                median_color = np.median(sample_block.reshape(-1, 3), axis=0)

            # Assign the color to the pixel in the output array
            pixel_art_array[y_idx, x_idx] = median_color.astype(np.uint8)

    return pixel_art_array


def upscale_pixel_art(
    pixel_art_array: np.ndarray, target_width: int, target_height: int
) -> np.ndarray:
    """
    Upscales the final pixel art array to the target dimensions using nearest-neighbor interpolation.

    Args:
        pixel_art_array: The small, clean pixel art image.
        target_width: The width of the original input image.
        target_height: The height of the original input image.

    Returns:
        The upscaled pixel art image.
    """
    if pixel_art_array.size == 0:
        return np.zeros((target_height, target_width, 3), dtype=np.uint8)

    return cv2.resize(
        pixel_art_array,
        (target_width, target_height),
        interpolation=cv2.INTER_NEAREST,
    )
