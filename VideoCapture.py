import cv2 as cv

def main():
    

    while(1):
        mode = input("Enter mode (0 for webcam, 1 for video file): ")
        try:
            mode = int(mode)
            break
        except:
            print("Invalid input. Please enter 0 or 1.")
            continue
    
    if(mode == 0):
        video_feed = cv.VideoCapture(0)

        while True:
            found_next_frame, frame = video_feed.read()

            if not found_next_frame:
                break
            

            # Applying SIFT detector
            sift = cv.SIFT_create()
            keypoints = sift.detect(frame, None)
            
            # Marking the keypoint on the image using circles
            frame_with_keypoints = cv.drawKeypoints(frame ,
                                keypoints ,
                                frame ,
                                color=(0, 255, 0),
                                flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            

            cv.imshow('Video Playback', frame_with_keypoints)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    if(mode == 1):
        video_feed = cv.VideoCapture('test_video.mp4')  # Replace with your video file path

        while True:
            found_next_frame, frame = video_feed.read()

            if not found_next_frame:
                print("End of video or failed to read")
                break
            
            # Applying SIFT detector
            sift = cv.SIFT_create()
            keypoints = sift.detect(frame, None)
            
            # Marking the keypoint on the image using circles
            frame_with_keypoints = cv.drawKeypoints(frame ,
                                keypoints ,
                                frame_with_keypoints ,
                                flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            

            cv.imshow('Video Playback', frame_with_keypoints)

            # Press 'q' to exit
            if cv.waitKey(25) & 0xFF == ord('q'):  # Check every 25ms for key press
                break

    video_feed.release()
    cv.destroyAllWindows()

                    



if __name__ == "__main__":
    main()