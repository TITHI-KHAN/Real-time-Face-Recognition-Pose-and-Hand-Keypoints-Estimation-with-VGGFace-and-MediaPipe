# Real-time Face Recognition, Pose and Hand Keypoints Estimation with VGGFace and MediaPipe

## To run the code, please either use "Visual Studio" or "Jupyter Notebook from Anaconda Navigator".

### Thank you.

<br>

## Code Explanation:

1. **Importing Libraries**: The code imports necessary libraries including `cv2` for OpenCV operations, `os` for file operations, `numpy` for numerical computations, `mediapipe` for pose and hand keypoint estimation, and specific modules from `keras_vggface` and `keras.preprocessing` for working with the VGGFace model and preprocessing images.

2. **Loading MediaPipe Models**: It loads the `Pose` and `Hands` models from MediaPipe for pose and hand keypoint estimation.

3. **Loading VGGFace Model**: It loads the VGGFace model with VGG16 architecture (`model='vgg16'`) for feature extraction.

4. **Feature Extraction**: It defines a function `extract_features()` to preprocess and extract features from a given image using the loaded VGGFace model.

5. **Loading Known Faces**: It iterates through the directories in the 'training_data' directory, reads the first image of each person, extracts features from the image using the VGGFace model, and stores the features along with the person's name in the `known_faces` dictionary.

6. **Capturing Video and Processing Frames**: It initializes a video capture object using `cv2.VideoCapture(0)` to capture frames from the default camera (index 0). It then continuously captures frames from the video feed.

7. **Pose and Hand Keypoints Detection**: It processes each frame for pose and hand keypoints detection using the MediaPipe models. It draws landmarks and connections for pose and hand keypoints on the frame using `mp_drawing.draw_landmarks()`.

8. **Face Recognition**: It extracts features from the detected face region in each frame and calculates the cosine similarity with known faces to recognize the identity of each detected face.

9. **Displaying Results**: It displays the processed frame with landmarks, hand keypoints, and predicted identities along with cosine distances using `cv2.imshow()`.

10. **Exiting**: The program exits the loop if the 'q' key is pressed, releases resources used for pose and hand keypoints estimation, video capture, and closes all OpenCV windows.

## Key Points:
- Utilizes VGGFace model pre-trained on VGG16 architecture for feature extraction.
- Uses MediaPipe for real-time pose and hand keypoints estimation.
- Processes video frames for face recognition, pose estimation, and hand keypoints estimation simultaneously.
- Draws landmarks and connections for pose and hand keypoints on the frames.
- Recognizes faces and displays predicted identities along with cosine distances.
- Allows quitting the application by pressing the 'q' key.
