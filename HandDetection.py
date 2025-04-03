import cv2 as cv
import mediapipe as mp

class HandDetector:
    def __init__(self):


        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_draw = mp.solutions.drawing_utils
        
        # The current bounding box coordinates
        self.bb_x_min = float('inf')
        self.bb_y_min = float('inf')
        self.bb_x_max = float('-inf')
        self.bb_y_max = float('-inf')

    def DetectHands(self,frame):
        
        
        # Reset the bounding box coordinates for each frame
        self.bb_x_min = float('inf')
        self.bb_y_min = float('inf')
        self.bb_x_max = float('-inf')
        self.bb_y_max = float('-inf')
        
        # Process the frame to detect hands
        results = self.hands.process(frame)
            
        # If hands are detected, find a bounding box around the hand
        if results.multi_hand_landmarks:
            
            # Check each hand landmark and see if it produces the largest bounding box
            for hand_landmarks in results.multi_hand_landmarks:

                for landmark in hand_landmarks.landmark:
                    x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                    self.bb_x_min, self.bb_y_min = min(x, self.bb_x_min), min(y, self.bb_y_min)
                    self.bb_x_max, self.bb_y_max = max(x, self.bb_x_max), max(y, self.bb_y_max)
                
                self.ExpandBoundingBox()

                # If our bounding box is within the frame, we can proceed with processing
                if(self.ValidateBoundingBox(frame)):
                    
                    # Draw a bounding box around the detected hand
                    cv.rectangle(frame, 
                                (self.bb_x_min, self.bb_y_min), 
                                (self.bb_x_max, self.bb_y_max), 
                                (0, 255, 0), 2)  # Green box, thickness=2


                    # Isolate the region of interest (ROI) around the hand
                    cropped_frame = frame[self.bb_y_min:self.bb_y_max, self.bb_x_min:self.bb_x_max]
                    
                    # cropped_frame = cv.cvtColor(cropped_frame, cv.COLOR_BGR2GRAY)

                    # Perform morphological opening to remove noise
                    #kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
                    #cropped_frame = cv.morphologyEx(cropped_frame, cv.MORPH_OPEN, kernel)

                    # cropped_frame = cv.threshold(cropped_frame, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
                    # cropped_frame = 255 - cropped_frame
                    # Convert edges to RGB so we can insert it back into the original frame
                    # edges_colored = cv.cvtColor(cropped_frame, cv.COLOR_GRAY2RGB)
                    

                    # Insert the cropped image back into the original frame
                    # frame[self.bb_y_min:self.bb_y_max , self.bb_x_min:self.bb_x_max] = edges_colored

                    return cropped_frame
        return None
    
    def ExpandBoundingBox(self):
        # Expand the bounding box by a given percent
        width_padding = (round(self.bb_x_max - self.bb_x_min) * 0.07)
        height_padding = (round(self.bb_y_max - self.bb_y_min) * 0.07)
        
        
        self.bb_x_min -= int(width_padding)
        self.bb_y_min -= int(height_padding)
        self.bb_x_max += int(width_padding)
        self.bb_y_max += int(height_padding)

        if(self.bb_x_max - self.bb_x_min > self.bb_y_max - self.bb_y_min):
            # If the width is greater than the height, we need to adjust the height to be equal to the width
            height_padding = (self.bb_x_max - self.bb_x_min) - (self.bb_y_max - self.bb_y_min)
            self.bb_y_min -= int(height_padding / 2)
            self.bb_y_max += int(height_padding / 2)
        elif(self.bb_y_max - self.bb_y_min > self.bb_x_max - self.bb_x_min):
            # If the height is greater than the width, we need to adjust the width to be equal to the height
            width_padding = (self.bb_y_max - self.bb_y_min) - (self.bb_x_max - self.bb_x_min)
            self.bb_x_min -= int(width_padding / 2)
            self.bb_x_max += int(width_padding / 2)


    
    def ValidateBoundingBox(self, frame):
            
            # If the bounding box is outside the frame, it is not valid. return False
            if(self.bb_x_min < 0 or
                self.bb_y_min < 0 or
                self.bb_x_max > frame.shape[1] or
                self.bb_y_max > frame.shape[0]):
                
                return False
            else:
                return True
            
    