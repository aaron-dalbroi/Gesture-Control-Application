import cv2 as cv
import mediapipe as mp
import HandDetection as hd


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
        # Initialize the video-feed
        video_feed = cv.VideoCapture(0)
        hand_detector = hd.HandDetector()
        while True:
            found_next_frame, frame = video_feed.read()
            if not found_next_frame:
                break
            
            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            frame_rgb = hand_detector.DetectHands(frame)


            frame_rgb = cv.flip(frame_rgb, 1) 
            cv.imshow('Video Playback', frame_rgb)
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
