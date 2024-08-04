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

# Function to preprocess images for saving
def preprocess_and_save_image(image, file_path):
    img = cv2.resize(image, (224, 224))  # Resize to 224x224
    img = img.astype('float32')
    img = (img - 127.5) / 128.0
    cv2.imwrite(file_path, img)  # Save preprocessed image to file

# Function to get face embedding
def get_face_embedding(image_path):
    try:
        # Generate embedding for the preprocessed image
        embeddings = DeepFace.represent(img_path=image_path, model_name='VGG-Face')
        return embeddings
    except Exception as e:
        print(f"Error processing face: {e}")
        return None

# Process each image
for image_name in os.listdir(image_dir):
    image_path = os.path.join(image_dir, image_name)
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Unable to load image {image_path}")
        continue

    # Detect faces using YOLOv5
    results = yolo_model(image)
    detections = results.xyxy[0]  # bounding boxes with confidence scores

    if len(detections) == 0:
        print(f"No faces detected in image {image_name}")
        continue

    for det in detections:
        x1, y1, x2, y2, confidence, _ = det
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        
        # Ensure valid face coordinates
        if x1 < 0 or y1 < 0 or x2 > image.shape[1] or y2 > image.shape[0]:
            print("Detected face coordinates are out of bounds")
            continue

        face = image[y1:y2, x1:x2]

        # Save the detected face as a file
        temp_face_path = 'temp_face.jpg'
        preprocess_and_save_image(face, temp_face_path)

        # Display the detected face and ask the user if it's Revati
        cv2.imshow('Detected Face', face)
        key = cv2.waitKey(0)  # Wait for user input
        cv2.destroyAllWindows()

        # If the key pressed is 'y', process the face
        if key == ord('y'):
            # Generate embedding for the detected face
            embedding = get_face_embedding(temp_face_path)
            if embedding is not None:
                revati_embeddings.extend(embedding)  # Note: DeepFace.represent returns a list
        # Continue to the next face/image if 'n' or any other key is pressed
        elif key == ord('n'):
            continue

# Clean up temporary file
if os.path.exists('temp_face.jpg'):
    os.remove('temp_face.jpg')

# Save the embeddings to a file
if revati_embeddings:
    revati_embeddings = np.array(revati_embeddings)
    np.save('revati_embeddings.npy', revati_embeddings)
    print(f"Saved {len(revati_embeddings)} embeddings to revati_embeddings.npy")
else:
    print("No valid embeddings were generated.")
