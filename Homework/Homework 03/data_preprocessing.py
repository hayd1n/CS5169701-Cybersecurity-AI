import os
import numpy as np
from PIL import Image
from typing import Union


def convert_to_grayscale_image(input_folder: str, output_folder: str) -> None:
    # Iterate through each subfolder
    for parent_dir in os.listdir(input_folder):
        parent_path = os.path.join(input_folder, parent_dir)
        if os.path.isdir(parent_path):
            # Create a corresponding output folder
            output_parent_path = os.path.join(output_folder, parent_dir)
            os.makedirs(output_parent_path, exist_ok=True)

            # Iterate through each file
            for file_name in os.listdir(parent_path):
                file_path = os.path.join(parent_path, file_name)

                if os.path.isfile(file_path):
                    # Read the binary content of the file
                    with open(file_path, "rb") as file:
                        byte_content: bytes = file.read()

                    # Convert the binary content to an 8-bit vector
                    byte_array: np.ndarray = np.frombuffer(byte_content, dtype=np.uint8)

                    # Use the square root of the binary array's length, rounding up, as the side length
                    image_side: int = int(np.ceil(np.sqrt(len(byte_array))))
                    padded_array: np.ndarray = np.pad(
                        byte_array,
                        (0, image_side * image_side - len(byte_array)),
                        mode="constant",
                    )

                    # Convert to a 2D array and generate a grayscale image
                    gray_image_array: np.ndarray = padded_array.reshape(
                        (image_side, image_side)
                    )
                    gray_image: Image.Image = Image.fromarray(gray_image_array, "L")

                    # Save the grayscale image
                    output_image_path: str = os.path.join(
                        output_parent_path, f"{os.path.splitext(file_name)[0]}.png"
                    )
                    gray_image.save(output_image_path)
                    print(f"Saved {output_image_path}")


# Specify input and output folders
input_folder = "C:\\Users\\hayden\\.conda\\envs\\virus_pic\\gray_virus\\PEs"  # Replace with your input folder path
output_folder = "C:\\Users\\hayden\\.conda\\envs\\virus_pic\\gray_virus\\output_image"  # Replace with your output folder path
os.makedirs(output_folder, exist_ok=True)

convert_to_grayscale_image(input_folder, output_folder)
