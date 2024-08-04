import cv2
import torch
import numpy as np
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity

# Load YOLOv5 model
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Load saved embeddings
revati_embeddings = np.load('revati_embeddings.npy')

# Function to preprocess images for DeepFace
def preprocess_image(image):
    img = cv2.resize(image, (224, 224))  # Resize to 224x224
    img = img.astype('float32')
    img = (img - 127.5) / 128.0
    return img

# Function to compare embeddings
def compare_embeddings(embedding, saved_embeddings, threshold=0.6):
    similarities = cosine_similarity([embedding], saved_embeddings)
    return np.max(similarities) > threshold

# Function to get face embedding
def get_face_embedding(face_image):
    try:
        preprocessed_face = preprocess_image(face_image)
        if preprocessed_face.shape != (224, 224, 3):
            print(f"Unexpected face shape: {preprocessed_face.shape}")
            return None
        
        embedding = DeepFace.represent([preprocessed_face], model_name='VGG-Face')
        return embedding[0]  # Assuming a single embedding
    except Exception as e:
        print(f"Error processing face: {e}")
        return None

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    # Detect faces using YOLOv5
    results = yolo_model(frame)
    detections = results.xyxy[0]  # bounding boxes with confidence scores
    
    for det in detections:
        x1, y1, x2, y2, confidence, _ = det
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        
        # Ensure valid face coordinates
        if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
            continue
        
        face = frame[y1:y2, x1:x2]
        
        # Get embedding for the detected face
        embedding = get_face_embedding(face)
        if embedding is not None:
            is_match = compare_embeddings(embedding, revati_embeddings)
            color = (0, 255, 0) if is_match else (0, 0, 255)
            label = "Match" if is_match else "No Match"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    # Display the frame
    cv2.imshow('Webcam Feed', frame)
    
    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
