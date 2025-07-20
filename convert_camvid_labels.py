import os
from PIL import Image
import numpy as np

# --- EXPLICIT MAPPING CONFIGURATION ---
# This dictionary clearly defines the mapping from each RGB color to a
# specific integer label. 'background' is manually set to 0.
COLOR_TO_LABEL_MAP = {
    # RGB Tuple      : Integer Label
    (255, 204, 51): 0,  # background
    (250, 250, 55): 1,  # apple
    (250, 125, 187): 2,  # buda
    (57, 25, 220): 3,   # matrioska
    (176, 128, 192): 4,  # table
    (102, 255, 102): 5,  # tomato
}

# A corresponding list of label names for easy reference and printing.
# The index of each name matches its integer label from the map above.
LABEL_NAMES = [
    'background', # 0
    'apple',      # 1
    'buda',       # 2
    'matrioska',  # 3
    'table',      # 4
    'tomato'      # 5
]


def create_grayscale_labels(input_dir, output_dir, color_map):
    """
    Converts RGB-labeled PNG images to grayscale-labeled images using a predefined
    color map dictionary.

    Args:
        input_dir (str): The path to the directory containing the RGB-labeled PNG images.
        output_dir (str): The path where the grayscale-labeled images will be saved.
        color_map (dict): A dictionary mapping RGB color tuples to integer labels.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    print("\nProcessing files...")
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".png"):
            try:
                img_path = os.path.join(input_dir, filename)
                with Image.open(img_path).convert("RGB") as rgb_image:
                    rgb_data = np.array(rgb_image)

                    # Create a new grayscale image, initializing with the background label (0).
                    grayscale_data = np.full(rgb_data.shape[:2], 0, dtype=np.uint8)

                    # For each color in our map, find where it occurs in the image
                    # and set the corresponding grayscale label.
                    for color, label_index in color_map.items():
                        mask = np.all(rgb_data == np.array(color).reshape(1, 1, 3), axis=2)
                        grayscale_data[mask] = label_index

                    # Convert the NumPy array back to a PIL Image in 'L' (grayscale) mode.
                    grayscale_image = Image.fromarray(grayscale_data, mode='L')

                    # Save the newly created grayscale image.
                    output_path = os.path.join(output_dir, filename)
                    grayscale_image.save(output_path)
                    print(f"  Successfully converted '{filename}'")

            except Exception as e:
                print(f"  Could not process '{filename}'. Reason: {e}")

    print("\nConversion process finished.")
    print("\nThe following integer mapping was used for the labels:")
    for i, label in enumerate(LABEL_NAMES):
        print(f"  {label}: {i}")


if __name__ == '__main__':
    # --- Configuration ---
    # IMPORTANT: You must change these paths to your actual directories.
    input_directory = "defaultannot"  # <-- ** CHANGE THIS **
    output_directory = "masks" # <-- ** CHANGE THIS **

    # To help you test the script, we can create some dummy files.
    # Set this to False if you have your real data ready.
    create_dummy_files = True 

    if create_dummy_files and not os.path.exists(input_directory):
        print(f"Creating a dummy input directory and a test image at: '{input_directory}'")
        os.makedirs(input_directory)
        
        # Create a sample image for testing purposes using the explicit map
        background_color = (255, 204, 51)
        apple_color = (250, 250, 55)
        matrioska_color = (57, 25, 220)
        table_color = (176, 128, 192)

        dummy_rgb_array = np.full((120, 160, 3), background_color, dtype=np.uint8)
        dummy_rgb_array[10:40, 10:50] = apple_color
        dummy_rgb_array[50:80, 60:110] = matrioska_color
        dummy_rgb_array[90:110, 20:140] = table_color
        dummy_image = Image.fromarray(dummy_rgb_array, 'RGB')
        dummy_image.save(os.path.join(input_directory, "sample_label.png"))
        print("A 'sample_label.png' has been created for you to test the script.")

    if os.path.isdir(input_directory):
        # Pass the explicit dictionary to the processing function.
        create_grayscale_labels(input_directory, output_directory, COLOR_TO_LABEL_MAP)
    else:
        print(f"Error: The input directory '{input_directory}' does not exist.")
        print("Please create it and place your PNG label files inside, or adjust the 'input_directory' variable in the script.")