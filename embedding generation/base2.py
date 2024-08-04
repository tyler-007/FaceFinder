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
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    return img



# Process each image
for image_name in os.listdir(image_dir):
    image_path = os.path.join(image_dir, image_name)
    image = cv2.imread(image_path)

    # Detect faces using YOLOv5
    results = yolo_model(image)
    detections = results.xyxy[0]  # bounding boxes with confidence scores

    for det in detections:
        x1, y1, x2, y2, confidence, _ = det
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        face = image[y1:y2, x1:x2]

        # Display the detected face and ask the user if it's Revati
        cv2.imshow('Detected Face', face)
        cv2.waitKey(0)  # Press any key to close the window
        response = input("Is this Revati's face? (y/n): ")
        cv2.destroyAllWindows()

        if response.lower() == 'y':
            # Preprocess and generate embedding
            preprocessed_face = preprocess_image(face)
            print(f"Preprocessed face shape: {preprocessed_face.shape}")            
            embedding = DeepFace.represent(preprocessed_face, model_name='VGG-Face')
            revati_embeddings.append(embedding)

# Save the embeddings to a file
revati_embeddings = np.array(revati_embeddings)
np.save('revati_embeddings.npy', revati_embeddings)
