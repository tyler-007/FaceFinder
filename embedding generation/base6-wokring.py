import cv2
import numpy as np
import os
from deepface import DeepFace

# Directory containing images of Revati
image_dir = '/Users/aayushjain/codes/projects/personal projects/RevatiFinder/images'
revati_embeddings = []

# Function to get face embedding
def get_face_embedding(image_path):
    try:
        # Use DeepFace to detect and represent the face
        result = DeepFace.represent(img_path=image_path, model_name='VGG-Face', enforce_detection=False, detector_backend='opencv')
        
        # DeepFace.represent returns a list of dictionaries, we want the 'embedding' from the first (and usually only) result
        if result and isinstance(result, list) and len(result) > 0:
            return result[0]['embedding']
        else:
            print(f"No valid embedding generated for {image_path}")
            return None
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

# Process each image
for image_name in os.listdir(image_dir):
    if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue  # Skip non-image files

    image_path = os.path.join(image_dir, image_name)
    
    # Read and display the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Unable to load image {image_path}")
        continue

    cv2.imshow('Current Image', image)
    print(f"Processing {image_name}. Is this Revati? (y/n)")
    key = cv2.waitKey(0)  # Wait for user input
    cv2.destroyAllWindows()

    # If the key pressed is 'y', process the image
    if key == ord('y'):
        # Generate embedding for the image
        embedding = get_face_embedding(image_path)
        if embedding is not None:
            revati_embeddings.append(embedding)
    # Continue to the next image if 'n' or any other key is pressed

# Save the embeddings to a file
if revati_embeddings:
    revati_embeddings = np.array(revati_embeddings)
    np.save('revati_embeddings.npy', revati_embeddings)
    print(f"Saved {len(revati_embeddings)} embeddings to revati_embeddings.npy")
else:
    print("No valid embeddings were generated.")