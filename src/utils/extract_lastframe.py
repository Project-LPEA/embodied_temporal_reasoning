import cv2 as cv

def extract_last_frame(video_path):
        """ 
        Function to extract last frame from input video

        """
        input_list = []
        
        cap = cv.VideoCapture(video_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            last_frame = frame
        cap.release()

        if last_frame is not None:
            cv.imwrite("last_frame.jpg", last_frame)
        else:
            print("No frames found in the video.")

import cv2 as cv


