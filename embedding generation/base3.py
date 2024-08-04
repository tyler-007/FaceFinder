import torch
import cv2
import numpy as np
import os
from deepface import DeepFace

# Load YOLOv5 model
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Directory containing images of Revati
image_dir = '/Users/aayushjain/codes/projects/personal projects/RevatiFinder/images'
revati_embeddings = []

# Function to preprocess images for DeepFace
def preprocess_image(image):
    img = cv2.resize(image, (224, 224))  # Resize to 224x224
    img = img.astype('float32')
    img = (img - 127.5) / 128.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension if needed for DeepFace
    return img

# Function to extract face embeddings using DeepFace
def get_face_embedding(face_image):
    try:
        preprocessed_face = preprocess_image(face_image)
        # Remove batch dimension if required by DeepFace
        preprocessed_face = preprocessed_face.squeeze(axis=0)
        embedding = DeepFace.represent(preprocessed_face, model_name='VGG-Face')
        return embedding
    except Exception as e:
        print(f"Error processing face: {e}")
        return None

# Process each image
for image_name in os.listdir(image_dir):
    image_path = os.path.join(image_dir, image_name)
    image = cv2.imread(image_path)
    
    # Check if the image was loaded correctly
    if image is None:
        print(f"Failed to load image: {image_path}")
        continue

    # Detect faces using YOLOv5
    results = yolo_model(image)
    detections = results.xyxy[0]  # bounding boxes with confidence scores

    for det in detections:
        x1, y1, x2, y2, confidence, _ = det
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        face = image[y1:y2, x1:x2]

        # Display the detected face
        cv2.imshow('Detected Face', face)
        cv2.waitKey(0)  # Press any key to close the window
        cv2.destroyAllWindows()

        # For demonstration, we assume user says 'yes' for simplicity
        response = 'y'  # In a real-world scenario, you might use a GUI or another method for user input

        if response.lower() == 'y':
            # Get the embedding
            embedding = get_face_embedding(face)
            if embedding is not None:
                revati_embeddings.extend(embedding)  # Add embeddings to the list

# Save the embeddings to a file
if revati_embeddings:
    revati_embeddings = np.array(revati_embeddings)
    np.save('revati_embeddings.npy', revati_embeddings)
else:
    print("No valid embeddings were generated.")
