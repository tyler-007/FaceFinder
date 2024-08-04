import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load saved embeddings
revati_embeddings = np.load('revati_embeddings.npy')

# Function to compare embeddings
def compare_embeddings(embedding, saved_embeddings, threshold=0.6):
    # Compute cosine similarity between the new embedding and saved embeddings
    similarities = cosine_similarity([embedding], saved_embeddings)
    
    # Check if the maximum similarity exceeds the threshold
    if np.max(similarities) > threshold:
        return True
    return False

# Example usage
def verify_face(face_image_path):
    try:
        # Get embedding for the face to verify
        embedding = DeepFace.represent(img_path=face_image_path, model_name='VGG-Face')
        embedding = embedding[0]  # Assuming a single embedding
        
        # Compare with saved embeddings
        is_match = compare_embeddings(embedding, revati_embeddings)
        if is_match:
            print("The face matches Revati!")
        else:
            print("The face does not match Revati.")
    except Exception as e:
        print(f"Error verifying face: {e}")

# Call the function with a face image path
verify_face('path_to_new_face_image.jpg')
