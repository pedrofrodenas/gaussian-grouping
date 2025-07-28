import cv2
import numpy as np
import glob
import os
import torch
from PIL import Image
import open_clip

# --- 1. Initialize CLIP Model ---
print("🧠 Initializing CLIP model...")
# Use 'cuda' if a GPU is available, otherwise 'cpu'
device = "cuda" if torch.cuda.is_available() else "cpu"
# Load the model, preprocessing pipeline, and tokenizer
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', 
    pretrained='laion2b_s34b_b79k', 
    device=device
)
# Set the model to evaluation mode (important for inference)
model.eval()
print(f"CLIP model loaded successfully on device: {device}")

# --- 2. Data Preprocessing: Glob masks and create a data dictionary ---

# Define paths and file extensions
mask_folder = "data/figuritas/object_mask/"
image_folder = "data/figuritas/images/"
mask_extension = ".png"
image_extension = ".JPG"

print("\n🔍 Starting to search for masks...")
mask_paths = glob.glob(os.path.join(mask_folder, f"*{mask_extension}"))
data_samples = []

# Loop through each found mask path to create our data structure
for mask_path in mask_paths:
    base_filename = os.path.splitext(os.path.basename(mask_path))[0]
    image_path = os.path.join(image_folder, base_filename + image_extension)

    if not os.path.exists(image_path):
        print(f"⚠️ Warning: Image not found for mask: {mask_path}. Skipping.")
        continue

    temp_label_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if temp_label_image is None:
        print(f"⚠️ Warning: Could not read mask {mask_path}. Skipping.")
        continue

    unique_labels = np.unique(temp_label_image).tolist()
    data_samples.append({
        "image_path": image_path,
        "mask_path": mask_path,
        "labels": unique_labels
    })

print(f"✅ Found {len(data_samples)} image/mask pairs.")

# --- 3. Process Images and Generate Embeddings ---

# This dictionary will store the final embeddings, grouped by object ID.
# Key: label_id, Value: numpy array of shape [n_images, embedding_size]
embeddings_by_label = {}

print("\n⚙️ Starting object extraction, preprocessing, and embedding generation...")

# Gather all unique labels from all samples, excluding background (0)
all_labels = set()
for sample in data_samples:
    all_labels.update(sample['labels'])
all_labels.discard(0) # Remove the background label

# --- Main Loop: Iterate over unique labels first ---
for label_id in sorted(list(all_labels)):
    print(f"--- Processing all images for Label ID: {label_id} ---")
    
    # This list will temporarily hold embeddings for the current label
    current_label_embeddings = []

    # Now, iterate over all data samples to find the ones containing this label
    for sample in data_samples:
        if label_id not in sample['labels']:
            continue

        label_image = cv2.imread(sample['mask_path'], cv2.IMREAD_GRAYSCALE)
        original_image = cv2.imread(sample['image_path'])

        if label_image is None or original_image is None:
            print(f"⚠️ Error reading files for label {label_id} in {sample['image_path']}. Skipping.")
            continue

        original_image = cv2.resize(original_image, (label_image.shape[1], label_image.shape[0]))
        mask = np.where(label_image == label_id, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            cropped_object = original_image[y:y+h, x:x+w]
            
            # --- START: CLIP Preprocessing and Embedding ---
            
            # 1. Convert cropped object from BGR (OpenCV) to RGB
            cropped_object_rgb = cv2.cvtColor(cropped_object, cv2.COLOR_BGR2RGB)
            
            # 2. Convert NumPy array to a PIL Image
            pil_image = Image.fromarray(cropped_object_rgb)
            
            # 3. Apply CLIP's preprocessing pipeline. This handles resizing to the correct
            # dimensions (e.g., 224x224), normalization, and tensor conversion.
            image_input = preprocess(pil_image).unsqueeze(0).to(device)

            # 4. Generate the embedding
            with torch.no_grad(), torch.cuda.amp.autocast():
                # Encode the image to get its features (embedding)
                image_features = model.encode_image(image_input)
                # Normalize the features, which is best practice for similarity calculations
                image_features /= image_features.norm(dim=-1, keepdim=True)

            # 5. Move embedding to CPU, convert to NumPy, and remove the batch dimension
            embedding = image_features.cpu().numpy().squeeze()
            current_label_embeddings.append(embedding)
            
            # --- END: CLIP Preprocessing and Embedding ---

    # After processing all images for the current label, stack the embeddings into a single numpy array
    if current_label_embeddings:
        # Use np.vstack to stack the list of 1D arrays into a 2D array
        embeddings_by_label[label_id] = np.vstack(current_label_embeddings)
        print(f"✅ Generated and stored {len(current_label_embeddings)} embeddings for Label ID {label_id}.")


# --- 4. Text-to-Image Matching ---

print("\n--- Starting Text-to-Image Matching ---")

# Get the tokenizer that corresponds to the loaded CLIP model
tokenizer = open_clip.get_tokenizer('ViT-B-32')

def find_best_text_match(text_query, model, tokenizer, embeddings_by_label, device):
    """
    Finds the object ID that best matches a given text query by comparing
    its embedding with the stored image embeddings.

    Args:
        text_query (str): The descriptive text to search for.
        model: The trained CLIP model.
        tokenizer: The CLIP tokenizer.
        embeddings_by_label (dict): A dictionary with label IDs as keys and
                                    a NumPy array of image embeddings as values.
        device (str): The device to run the model on ('cuda' or 'cpu').

    Returns:
        int: The label ID of the best matching object, or None if no match is found.
    """
    print(f"\n🔍 Searching for: '{text_query}'")

    if not embeddings_by_label:
        print("⚠️ Cannot perform search, the embeddings dictionary is empty.")
        return None

    # 1. Tokenize and Encode the Text Query
    # The tokenizer converts the string into a format the model understands.
    text_input = tokenizer([text_query]).to(device)

    with torch.no_grad(), torch.cuda.amp.autocast():
        # Encode the text to get its feature vector (embedding)
        text_features = model.encode_text(text_input)
        # Normalize the features, which is crucial for accurate similarity calculation
        text_features /= text_features.norm(dim=-1, keepdim=True)

    # Move the text embedding to the CPU and convert it to a NumPy array
    text_embedding = text_features.cpu().numpy()

    # 2. Calculate Similarity Scores
    # This dictionary will store the average similarity score for each object ID.
    mean_similarities = {}

    # Iterate through each object's stored image embeddings
    for label_id, image_embeddings in embeddings_by_label.items():
        # Calculate the dot product between the text embedding and all image embeddings
        # for the current label ID. Since all vectors are normalized, this dot product
        # is equivalent to the cosine similarity.
        # Calculation shape: [1, emb_size] @ [emb_size, n_images] -> [1, n_images]
        similarities = np.dot(text_embedding, image_embeddings.T)

        # To get a single representative score for the object, we average the
        # similarities calculated from its different views/images.
        mean_similarities[label_id] = np.mean(similarities)

    # 3. Find and Report the Best Match
    # Find the label_id (key) that has the highest similarity score (value)
    best_match_id = max(mean_similarities, key=mean_similarities.get)
    best_score = mean_similarities[best_match_id]

    print("\n📊 Calculated Mean Similarity Scores:")
    for label_id, score in mean_similarities.items():
        print(f"  - Object ID {label_id}: {score:.4f}")

    print(f"\n🏆 Best match is Object ID: {best_match_id} with a score of {best_score:.4f}")
    
    return best_match_id

# --- Example Usage ---
# You must change this string to describe an object present in your dataset.
search_query = "an tin of tomato" 

# Call the function to find the object that best matches the text query.
best_id = find_best_text_match(
    text_query=search_query,
    model=model,
    tokenizer=tokenizer,
    embeddings_by_label=embeddings_by_label,
    device=device
)

print("hola")