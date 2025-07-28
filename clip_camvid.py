import torch
from PIL import Image
import numpy as np
import open_clip
import os

def create_dummy_data():
    """
    Generates a sample image with three colored squares and a corresponding
    segmentation mask for demonstration purposes.
    """
    # Create directories if they don't exist
    os.makedirs('images', exist_ok=True)
    os.makedirs('masks', exist_ok=True)

    # --- Create a sample image ---
    # Create a 300x300 white background image
    image_array = np.full((300, 300, 3), 255, dtype=np.uint8)
    # Add a red square (R=255, G=0, B=0)
    image_array[50:150, 20:120] = [255, 0, 0]
    # Add a green square (R=0, G=255, B=0)
    image_array[50:150, 180:280] = [0, 255, 0]
    # Add a blue circle (R=0, G=0, B=255)
    center_x, center_y, radius = 225, 225, 40
    y, x = np.ogrid[:300, :300]
    dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    image_array[dist_from_center <= radius] = [0, 0, 255]

    sample_image = Image.fromarray(image_array)
    sample_image.save('images/sample_scene.png')

    # --- Create a corresponding segmentation mask ---
    # CamVid masks use integer class labels.
    # 0: background
    # 1: red square
    # 2: green square
    # 3: blue circle
    mask_array = np.zeros((300, 300), dtype=np.uint8)
    mask_array[50:150, 20:120] = 1  # Class 1
    mask_array[50:150, 180:280] = 2  # Class 2
    mask_array[dist_from_center <= radius] = 3 # Class 3

    sample_mask = Image.fromarray(mask_array)
    sample_mask.save('masks/sample_scene_mask.png')

    print("✅ Dummy data created: 'images/sample_scene.png' and 'masks/sample_scene_mask.png'")
    return 'images/sample_scene.png', 'masks/sample_scene_mask.png'

def segment_objects_from_mask(image_path, mask_path):
    """
    Takes an image and a segmentation mask, and returns a dictionary of
    isolated object images.

    Args:
        image_path (str): Path to the original image.
        mask_path (str): Path to the segmentation mask.

    Returns:
        dict: A dictionary where keys are object class IDs (from the mask)
              and values are PIL Image objects of the isolated objects.
    """
    print("\n🔍 Starting object segmentation...")
    original_image = Image.open(image_path).convert('RGB')
    mask_image = Image.open(mask_path)

    # Convert images to numpy arrays for processing
    original_array = np.array(original_image)
    mask_array = np.array(mask_image)

    # Find the unique non-zero object IDs in the mask
    # Non-zero values represent different objects.
    object_ids = np.unique(mask_array)
    object_ids = object_ids[object_ids != 0] # Exclude background (class 0)

    if len(object_ids) == 0:
        print("No objects found in the mask.")
        return {}

    print(f"Found {len(object_ids)} objects with IDs: {object_ids}")

    segmented_images = {}
    for obj_id in object_ids:
        # Create a binary mask for the current object
        # True where the mask equals the current object ID, False otherwise
        binary_mask = (mask_array == obj_id)

        # Create a new black image array
        black_background = np.zeros_like(original_array)

        # Use the binary mask to copy the object's pixels from the original
        # image onto the black background.
        # np.newaxis is used to align the 2D mask with the 3D image array.
        black_background[binary_mask] = original_array[binary_mask]

        # Convert the array back to a PIL Image
        segmented_image = Image.fromarray(black_background)
        segmented_images[obj_id] = segmented_image

        # Optional: Save the segmented images to disk to inspect them
        os.makedirs('output', exist_ok=True)
        segmented_image.save(f'output/object_{obj_id}.png')
        print(f"  - Saved isolated image for object ID {obj_id} to 'output/object_{obj_id}.png'")

    return segmented_images

def find_best_match(text_query, segmented_images, model, preprocess, tokenizer):
    """
    Compares a text query to a set of segmented images using CLIP and
    finds the best match.

    Args:
        text_query (str): The text description to search for.
        segmented_images (dict): A dictionary of object IDs and their images.
        model: The CLIP model.
        preprocess: The CLIP image preprocessing transform.
        tokenizer: The CLIP text tokenizer.

    Returns:
        tuple: The ID of the best matching object and the similarity probabilities.
    """
    print(f"\n🤖 Running CLIP to find the best match for: '{text_query}'")

    if not segmented_images:
        print("Cannot find match, no segmented images were provided.")
        return None, None

    # Prepare the text input
    text = tokenizer([text_query]).to(device)

    image_tensors = []
    object_ids = list(segmented_images.keys())

    # Preprocess all segmented images
    for obj_id in object_ids:
        image = segmented_images[obj_id]
        preprocessed_image = preprocess(image).unsqueeze(0).to(device)
        image_tensors.append(preprocessed_image)

    # Stack all image tensors into a single batch for efficient processing
    image_batch = torch.cat(image_tensors)

    with torch.no_grad(), torch.cuda.amp.autocast():
        # Encode both the image batch and the text query
        image_features = model.encode_image(image_batch)
        text_features = model.encode_text(text)

        # Normalize the features
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # Calculate similarity
        # The result is a matrix of similarities between each image and each text query.
        # Since we have one text query, we get a [N_images, 1] tensor.
        similarity = (100.0 * text_features @ image_features.T)

        # Use softmax to get probabilities
        probs = similarity.softmax(dim=-1)

    # Convert probabilities to a list for easier inspection
    probs_list = probs.cpu().numpy().flatten().tolist()

    # Print results for each object
    print("\n📊 Similarity Scores:")
    for i, obj_id in enumerate(object_ids):
        print(f"  - Object ID {obj_id}: {probs_list[i]:.4f}")

    # Find the object with the highest probability
    best_match_index = np.argmax(probs_list)
    best_match_id = object_ids[best_match_index]

    return best_match_id, probs_list


if __name__ == '__main__':
    # --- 1. Setup and Data Creation ---
    # Create dummy image and mask for this example
    image_path, mask_path = create_dummy_data()

    # --- 2. Isolate Objects ---
    # This function creates one new image per object found in the mask
    segmented_objects = segment_objects_from_mask(image_path, mask_path)

    # --- 3. Initialize CLIP Model ---
    print("\n🧠 Initializing CLIP model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', device=device)
    model.eval()
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    print("CLIP model loaded successfully.")

    # --- 4. Perform Similarity Search ---
    # Define the text you want to search for in the image
    search_text = "a blue sphere"

    best_id, probabilities = find_best_match(
        text_query=search_text,
        segmented_images=segmented_objects,
        model=model,
        preprocess=preprocess,
        tokenizer=tokenizer
    )

    # --- 5. Report Final Result ---
    if best_id is not None:
        print(f"\n🏆 Best match for '{search_text}' is Object ID: {best_id}")

    # --- Example with a different query ---
    print("\n" + "="*50)
    search_text_2 = "the green block"
    best_id_2, _ = find_best_match(
        text_query=search_text_2,
        segmented_images=segmented_objects,
        model=model,
        preprocess=preprocess,
        tokenizer=tokenizer
    )
    if best_id_2 is not None:
        print(f"\n🏆 Best match for '{search_text_2}' is Object ID: {best_id_2}")
