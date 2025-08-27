import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import median_filter
from skimage import measure, morphology
from collections import Counter


def detect_grid_lines(
    image_path,
    edge_threshold1=50,
    edge_threshold2=150,
    hough_threshold=100,
    min_line_length=50,
    max_line_gap=10,
    angle_tolerance=2,
    noise_reduction=True,
):
    """
    Detect grid lines in a noisy pixel art image with uneven pixel sizes.

    Parameters:
    -----------
    image_path : str
        Path to the input image
    edge_threshold1, edge_threshold2 : int
        Canny edge detection thresholds
    hough_threshold : int
        Minimum votes for Hough line detection
    min_line_length : int
        Minimum length of detected lines
    max_line_gap : int
        Maximum gap between line segments to connect them
    angle_tolerance : float
        Tolerance in degrees for grouping parallel lines
    noise_reduction : bool
        Apply noise reduction preprocessing

    Returns:
    --------
    result_image : numpy array
        Original image with grid lines overlaid
    grid_info : dict
        Information about detected grid lines
    """

    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Store original dimensions
    height, width = gray.shape

    # Step 1: Noise reduction (optional but recommended for noisy images)
    if noise_reduction:
        # Apply bilateral filter to reduce noise while preserving edges
        gray = cv2.bilateralFilter(gray, 9, 75, 75)

        # Apply median filter to further reduce salt-and-pepper noise
        gray = median_filter(gray, size=3)

    # Step 2: Edge detection using multiple methods for robustness

    # Method A: Gradient-based edge detection
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Compute gradient magnitude
    gradient_mag = np.sqrt(sobelx**2 + sobely**2)
    gradient_mag = (gradient_mag / gradient_mag.max() * 255).astype(np.uint8)

    # Method B: Canny edge detection
    edges_canny = cv2.Canny(gray, edge_threshold1, edge_threshold2)

    # Combine edge detection methods
    edges_combined = cv2.bitwise_or(
        edges_canny, cv2.threshold(gradient_mag, 50, 255, cv2.THRESH_BINARY)[1]
    )

    # Step 3: Morphological operations to connect broken lines
    kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))

    # Process vertical and horizontal edges separately
    edges_vertical = cv2.morphologyEx(edges_combined, cv2.MORPH_CLOSE, kernel_vertical)
    edges_horizontal = cv2.morphologyEx(
        edges_combined, cv2.MORPH_CLOSE, kernel_horizontal
    )

    edges_processed = cv2.bitwise_or(edges_vertical, edges_horizontal)

    # Step 4: Detect lines using Probabilistic Hough Transform
    lines = cv2.HoughLinesP(
        edges_processed,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )

    if lines is None:
        print("No lines detected. Try adjusting parameters.")
        return img_rgb, {}

    # Step 5: Filter and group lines
    vertical_lines = []
    horizontal_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]

        # Calculate angle
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

        # Classify as vertical or horizontal based on angle
        if abs(angle) < angle_tolerance or abs(angle - 180) < angle_tolerance:
            horizontal_lines.append((x1, y1, x2, y2))
        elif abs(angle - 90) < angle_tolerance or abs(angle + 90) < angle_tolerance:
            vertical_lines.append((x1, y1, x2, y2))

    # Step 6: Merge nearby parallel lines
    def merge_parallel_lines(lines, axis="vertical"):
        if not lines:
            return []

        merged = []
        if axis == "vertical":
            # Sort by x-coordinate
            lines = sorted(lines, key=lambda l: (l[0] + l[2]) / 2)
            merge_threshold = 10  # pixels

            current_group = [lines[0]]
            for line in lines[1:]:
                avg_x_current = (line[0] + line[2]) / 2
                avg_x_group = sum((l[0] + l[2]) / 2 for l in current_group) / len(
                    current_group
                )

                if abs(avg_x_current - avg_x_group) < merge_threshold:
                    current_group.append(line)
                else:
                    # Merge current group into single line
                    if current_group:
                        avg_x = sum((l[0] + l[2]) / 2 for l in current_group) / len(
                            current_group
                        )
                        merged.append((int(avg_x), 0, int(avg_x), height))
                    current_group = [line]

            # Don't forget the last group
            if current_group:
                avg_x = sum((l[0] + l[2]) / 2 for l in current_group) / len(
                    current_group
                )
                merged.append((int(avg_x), 0, int(avg_x), height))

        else:  # horizontal
            # Sort by y-coordinate
            lines = sorted(lines, key=lambda l: (l[1] + l[3]) / 2)
            merge_threshold = 10  # pixels

            current_group = [lines[0]]
            for line in lines[1:]:
                avg_y_current = (line[1] + line[3]) / 2
                avg_y_group = sum((l[1] + l[3]) / 2 for l in current_group) / len(
                    current_group
                )

                if abs(avg_y_current - avg_y_group) < merge_threshold:
                    current_group.append(line)
                else:
                    # Merge current group into single line
                    if current_group:
                        avg_y = sum((l[1] + l[3]) / 2 for l in current_group) / len(
                            current_group
                        )
                        merged.append((0, int(avg_y), width, int(avg_y)))
                    current_group = [line]

            # Don't forget the last group
            if current_group:
                avg_y = sum((l[1] + l[3]) / 2 for l in current_group) / len(
                    current_group
                )
                merged.append((0, int(avg_y), width, int(avg_y)))

        return merged

    vertical_lines = merge_parallel_lines(vertical_lines, "vertical")
    horizontal_lines = merge_parallel_lines(horizontal_lines, "horizontal")

    # Step 7: Detect grid spacing (for uneven pixel sizes)
    def detect_spacing(lines, axis="vertical"):
        if len(lines) < 2:
            return []

        positions = []
        if axis == "vertical":
            positions = sorted([(l[0] + l[2]) / 2 for l in lines])
        else:
            positions = sorted([(l[1] + l[3]) / 2 for l in lines])

        spacings = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
        return spacings

    v_spacings = detect_spacing(vertical_lines, "vertical")
    h_spacings = detect_spacing(horizontal_lines, "horizontal")

    # Step 8: Draw the detected grid lines on the original image
    result_image = img_rgb.copy()

    # Draw vertical lines in red
    for line in vertical_lines:
        x1, y1, x2, y2 = line
        cv2.line(result_image, (x1, y1), (x2, y2), (255, 0, 0), 1)

    # Draw horizontal lines in blue
    for line in horizontal_lines:
        x1, y1, x2, y2 = line
        cv2.line(result_image, (x1, y1), (x2, y2), (0, 0, 255), 1)

    # Calculate grid statistics
    grid_info = {
        "num_vertical_lines": len(vertical_lines),
        "num_horizontal_lines": len(horizontal_lines),
        "vertical_spacings": v_spacings,
        "horizontal_spacings": h_spacings,
        "avg_vertical_spacing": np.mean(v_spacings) if v_spacings else 0,
        "avg_horizontal_spacing": np.mean(h_spacings) if h_spacings else 0,
        "std_vertical_spacing": np.std(v_spacings) if v_spacings else 0,
        "std_horizontal_spacing": np.std(h_spacings) if h_spacings else 0,
    }

    return result_image, grid_info


def visualize_results(original_path, save_path=None):
    """
    Visualize the grid detection results with diagnostic information.

    Parameters:
    -----------
    original_path : str
        Path to the input image
    save_path : str, optional
        Path to save the result image
    """

    # Detect grid lines
    result_img, grid_info = detect_grid_lines(original_path)

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # Original image
    original = cv2.imread(original_path)
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    axes[0].imshow(original_rgb)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Result with grid lines
    axes[1].imshow(result_img)
    axes[1].set_title("Detected Grid Lines")
    axes[1].axis("off")

    # Add grid information as text
    info_text = f"Vertical lines: {grid_info['num_vertical_lines']}\n"
    info_text += f"Horizontal lines: {grid_info['num_horizontal_lines']}\n"
    info_text += f"Avg vertical spacing: {grid_info['avg_vertical_spacing']:.1f} ± {grid_info['std_vertical_spacing']:.1f}\n"
    info_text += f"Avg horizontal spacing: {grid_info['avg_horizontal_spacing']:.1f} ± {grid_info['std_horizontal_spacing']:.1f}"

    fig.text(
        0.5,
        0.02,
        info_text,
        ha="center",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Result saved to {save_path}")

    plt.show()

    return result_img, grid_info


def adaptive_grid_detection(image_path, auto_tune=True):
    """
    Adaptive grid detection that automatically adjusts parameters based on image characteristics.

    Parameters:
    -----------
    image_path : str
        Path to the input image
    auto_tune : bool
        Automatically tune parameters based on image analysis
    """

    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if auto_tune:
        # Analyze image characteristics
        height, width = gray.shape

        # Estimate noise level
        noise_level = np.std(gray - cv2.GaussianBlur(gray, (5, 5), 0))

        # Adjust parameters based on image size and noise
        edge_threshold1 = max(30, min(100, int(noise_level * 2)))
        edge_threshold2 = max(100, min(200, int(noise_level * 5)))
        hough_threshold = max(50, min(height // 10, 200))
        min_line_length = max(20, min(height // 20, width // 20))

        print(f"Auto-tuned parameters:")
        print(f"  Edge thresholds: {edge_threshold1}, {edge_threshold2}")
        print(f"  Hough threshold: {hough_threshold}")
        print(f"  Min line length: {min_line_length}")

        return detect_grid_lines(
            image_path,
            edge_threshold1=edge_threshold1,
            edge_threshold2=edge_threshold2,
            hough_threshold=hough_threshold,
            min_line_length=min_line_length,
        )
    else:
        return detect_grid_lines(image_path)


# Example usage
if __name__ == "__main__":
    # Replace with your image path
    image_path = "images/forest_ruins_bridge/input.png"

    # Method 1: Basic detection with default parameters
    result, info = detect_grid_lines(image_path)

    # Method 2: Visualization with diagnostic information
    # visualize_results(
    #     image_path, save_path="images/forest_ruins_bridge/grid_detected.png"
    # )

    # Method 3: Adaptive detection with auto-tuning
    result, info = adaptive_grid_detection(image_path, auto_tune=True)

    # Display result
    plt.figure(figsize=(12, 8))
    plt.imshow(result)
    plt.title("Pixel Art with Detected Grid Lines")
    plt.axis("off")
    plt.show()

    # Print grid information
    print("\nGrid Detection Results:")
    print(f"Number of vertical lines: {info['num_vertical_lines']}")
    print(f"Number of horizontal lines: {info['num_horizontal_lines']}")
    print(f"Average vertical spacing: {info['avg_vertical_spacing']:.2f} pixels")
    print(f"Average horizontal spacing: {info['avg_horizontal_spacing']:.2f} pixels")
    print(f"Vertical spacing std dev: {info['std_vertical_spacing']:.2f} pixels")
    print(f"Horizontal spacing std dev: {info['std_horizontal_spacing']:.2f} pixels")
