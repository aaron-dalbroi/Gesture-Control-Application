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

            gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            cv.imshow('Grayscale Feed', gray_frame)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    if(mode == 1):
        video_feed = cv.VideoCapture('test_video.mp4')  # Replace with your video file path

        while True:
            found_next_frame, frame = video_feed.read()

            if not found_next_frame:
                print("End of video or failed to read")
                break

            cv.imshow('Video Playback', frame)

            # Press 'q' to exit
            if cv.waitKey(25) & 0xFF == ord('q'):  # Check every 25ms for key press
                break

    video_feed.release()
    cv.destroyAllWindows()

                    



if __name__ == "__main__":
    main()