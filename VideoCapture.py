import cv2 as cv
import numpy as np
import torch
import torch.nn.functional as F
import mediapipe as mp
import HandDetection as hd
import GestureModel

def main():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils
    model = GestureModel.GestureModel('rotated_gesture_model.pth')

    while True:
        mode = input("Enter mode (0 for webcam, 1 for video file): ")
        try:
            mode = int(mode)
            break
        except:
            print("Invalid input. Please enter 0 or 1.")
            continue
    
    if mode == 0:
        # Initialize the video-feed
        video_feed = cv.VideoCapture(0)
        hand_detector = hd.HandDetector()
        while True:
            found_next_frame, frame = video_feed.read()
            if not found_next_frame:
                break

            # Returns an image of the hand0 detected in the frame
            hand = hand_detector.DetectHands(frame)
            if(hand is not None):
                


                # Normalizes the frame to 28x28 to match the input size of the model
                normalized_frame = downsample_and_pad(hand, target_size=(28, 28))
                
                # Convert to graycale and normalize
                normalized_frame_grayscaled = cv.cvtColor(normalized_frame, cv.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            
                predictions = model.predict(normalized_frame_grayscaled)
                predicted_class = np.argmax(predictions)  # Get the class with highest probability

                # Define class names (assuming you have these in a list, update accordingly)
                class_names = ['Open Palm', 'OK Sign', 'Pointing', 'Peace Sign']  # Replace with your actual class names
                
                # Get the predicted class label
                predicted_label = class_names[predicted_class]
                
                # Display the predicted label on the frame
                cv.putText(frame, f'Predicted: {predicted_label}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)


                cv.imshow('Video Playback', frame)
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
            else:

                cv.putText(frame, f'Hand Not Detected', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)

                cv.imshow('Video Playback', frame)
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break


    if mode == 1:
        video_feed = cv.VideoCapture('test_video.mp4')
        while True:
            found_next_frame, frame = video_feed.read()
            if not found_next_frame:
                print("End of video or failed to read")
                break
            
            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            cv.rectangle(frame, (300, 100), (500, 300), (0, 255, 0), 2)
            
            sift = cv.SIFT_create()
            keypoints = sift.detect(frame, None)
            frame_with_keypoints = cv.drawKeypoints(frame, keypoints, frame, color=(0, 255, 0), flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            
            cv.imshow('Video Playback', frame_with_keypoints)
            if cv.waitKey(25) & 0xFF == ord('q'):
                break
    
    video_feed.release()
    cv.destroyAllWindows()

def downsample_and_pad(image, target_size=(28, 28)):
    # Resize the image to exactly 28x28
    resized_image = cv.resize(image, target_size, interpolation=cv.INTER_AREA)
    return resized_image

def apply_softmax(logits):
    """
    Apply softmax function to logits to convert them into probabilities.

    Parameters:
    logits (numpy.ndarray or torch.Tensor): The raw logits output by the model.

    Returns:
    torch.Tensor: The probabilities after applying softmax.
    """
    if isinstance(logits, np.ndarray):
        logits = torch.tensor(logits, dtype=torch.float32)  # Convert numpy array to torch tensor if necessary
    probabilities = F.softmax(logits, dim=-1)  # Apply softmax along the last dimension
    return probabilities

if __name__ == "__main__":
    main()
