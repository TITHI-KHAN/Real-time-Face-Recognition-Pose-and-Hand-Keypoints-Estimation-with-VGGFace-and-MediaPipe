import cv2
import os
import numpy as np
import mediapipe as mp
from scipy.spatial.distance import cosine
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from keras.preprocessing import image


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True)
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)


model = VGGFace(model='vgg16', include_top=False, input_shape=(224, 224, 3), pooling='avg')

def extract_features(img):
    img = cv2.resize(img, (224, 224))  # Resize image to expected size
    img = img.astype('float32')  # Convert to float32
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = preprocess_input(img)  # Preprocess image
    features = model.predict(img)  # Extract features
    return features


known_faces = {}
training_data_path = 'training_data'
for dir_name in os.listdir(training_data_path):
    subject_path = os.path.join(training_data_path, dir_name)
    if not os.path.isdir(subject_path):
        continue

    face_images = os.listdir(subject_path)
    if face_images:
        img_path = os.path.join(subject_path, face_images[0])
        img = cv2.imread(img_path)
        features = extract_features(img)
        known_faces[dir_name] = features

video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Pose and hand keypoints detection
    results_pose = pose.process(frame_rgb)
    results_hands = hands.process(frame_rgb)

    # Draw the pose and hand landmarks
    if results_pose.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    
    face_roi = frame  
    captured_features = extract_features(face_roi)


    min_dist = float('inf')
    identity = "Unknown"


    for name, features in known_faces.items():
        dist = cosine(features.flatten(), captured_features.flatten())
        if dist < min_dist:
            min_dist = dist
            identity = name

    label_text = f"{identity}, {min_dist:.2f}"
    cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Real-time Face Recognition, Pose and Hand Estimation", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pose.close()
hands.close()
video_capture.release()
cv2.destroyAllWindows()
