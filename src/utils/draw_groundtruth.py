import cv2 as cv
from PIL import Image
import numpy as np

def draw_groundtruth(options_path, output_path, bbox_coords, ground_truth_coords):
             
        """ Function to draw bounding box coordinates on image

        Returns:
            image with grounded object of interest and bounding box coordinates
            
        """

        with Image.open(options_path) as img:
            # Convert the image to RGB if it's not already in that mode
                # img = img.convert('RGB')
                # Convert the image to a numpy array
                image_array = np.array(img)
                for row in image_array:
                    for i in range(len(row)):
                        row[i] = (row[i][2], row[i][1], row[i][0]) 

        save_img = output_path
        
        color1 = (0,255,0)
        color2 = (255,0, 0)
        thickness = 2
        x1, y1, x2, y2 = ground_truth_coords
        x1, y1, x2, y2 = round(x1), round(y1), round(x2), round(y2)
        cv.rectangle(image_array, (x1, y1), (x2, y2), color1, thickness)

        x1, y1, x2, y2 = bbox_coords
        cv.rectangle(image_array, (int(x1), int(y1)), (int(x2), int(y2)), color2, thickness)
        print("Size of frame : ", np.size(image_array))
        cv.imwrite(save_img, image_array)