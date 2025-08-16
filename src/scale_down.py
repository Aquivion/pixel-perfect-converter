from pathlib import Path
import sys
from PIL import Image

def scale_down_image(image_path, block_size, output_path):
    """
    Scale down the image to (image_size / block_size) using nearest neighbor method.
    """
    img = Image.open(image_path)
    img = img.convert("RGB")

    # Calculate the new dimensions
    width, height = img.size
    new_width = width // block_size
    new_height = height // block_size

    # Scale down the image using nearest neighbor method
    scaled_img = img.resize((new_width, new_height), Image.NEAREST)

    # Save the scaled image
    scaled_img.save(output_path)
    print(f"Scaled image saved to {output_path}")

def main(block_size):
    input_image_path = "output.png"
    output_image_path = "scaled_output.png"

    scale_down_image(input_image_path, block_size, output_image_path)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scale_down.py <block_size>")
        sys.exit(1)

    block_size = str(sys.argv[1])

    if block_size.isdigit():
        block_size = int(block_size)
    else:
        print("Block size must be a digit. Using default value of 16.")
        block_size = 16

    main(block_size)