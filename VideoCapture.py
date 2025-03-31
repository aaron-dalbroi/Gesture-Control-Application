import cv2 as cv
import mediapipe as mp

def main():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils

    while True:
        mode = input("Enter mode (0 for webcam, 1 for video file): ")
        try:
            mode = int(mode)
            break
        except:
            print("Invalid input. Please enter 0 or 1.")
            continue
    
    if mode == 0:
        video_feed = cv.VideoCapture(0)

        while True:
            found_next_frame, frame = video_feed.read()
            if not found_next_frame:
                break
            
            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            
            x_min, y_min = float('inf'), float('inf')
            x_max, y_max = float('-inf'), float('-inf')
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                                        # Create bounding box around the hand

                    for landmark in hand_landmarks.landmark:
                        x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                        x_min, y_min = min(x, x_min), min(y, y_min)
                        x_max, y_max = max(x, x_max), max(y, y_max)
                    cv.rectangle(frame, (x_min - 25, y_min - 25), (x_max + 25, y_max + 25), (0, 255, 0), 2)
 
            
            if(x_min < x_max and y_min < y_max and x_min > 0 and y_min > 0 and x_max < frame.shape[1] and y_max < frame.shape[0]):
                cv.imshow('Hand Detection', frame[y_min:y_max, x_min:x_max])
            else:
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

if __name__ == "__main__":
    main()
