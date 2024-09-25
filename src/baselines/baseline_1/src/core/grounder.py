#!/usr/bin/env python3   

# import numpy
import socket
# import rospy
# from std_msgs.msg import String
# import openai
import cv2 as cv
import pickle
# from sensor_msgs.msg import Image, CompressedImage
# from cv_bridge import CvBridge
# import sys
import numpy as np
# import time
import json
# from PIL import Image
import ast
from openai import OpenAI
# import os




def send_msg_to_server(image, recv_query, output_path):
        
        """ Function to send query and image to CogVLM

        Returns:
            scaled_bbox_coords: grounded bounding box coordinates as per scaled image
        """
        
        # for row in image:
        #     for i in range(len(row)):
        #         row[i] = (row[i][2], row[i][1], row[i][0]) 

        print("Calling the COGVLM server")
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(("10.237.20.209", 65433))
        print("Sending Query: ", recv_query)
        msg = {"image_array": image, "query": recv_query}
        byte_msg = pickle.dumps(msg)
        s.sendall(byte_msg)
        s.shutdown(socket.SHUT_WR)
        data = b""
        while True:
            packet = s.recv(1024)
            if not packet:
                break
            data += packet
        data = pickle.loads(data)

        print("Before resize:", data)
       
        scaled_bbox_coords = filter_object_name_list_from_dino(image.shape, image, data, output_path)
        
        return scaled_bbox_coords


def draw_bbox(image_size, image_array, bbox_coords, output_path):
        
        """ Function to draw bounding box coordinates on image

        Returns:
            image with grounded object of interest and bounding box coordinates
        """

        color = (255,0,0)
        thickness = 2
        print(image_size)
        image_x = image_size[1]
        image_y = image_size[0]
        print("Image coord", (image_x, image_y))
        x1, y1, x2, y2 = bbox_coords 
        x1 = int((x1/1000) * (image_x))
        y1 = int((y1/1000) * (image_y))
        x2 = int((x2/1000) * (image_x))
        y2 = int((y2/1000) * (image_y))
        scaled_bbox_coords = [x1, y1, x2, y2]
        image_array = np.array(image_array)
        cv.rectangle(image_array, (x1, y1), (x2, y2), color, thickness)
        cv.imwrite(output_path, image_array)
        return scaled_bbox_coords


def filter_object_name_list_from_dino(image_size, image_array, bbox_string, output_path):
        

        """ Function to return coordinates in correct format

        Returns:
            bounding box coordinates in a list (corrected format)
        """

        sys_prompt = "Suppose I have a list: [[168,015,421,465]]. Output in the following format: [[168,15,421,465]]. Only output the list, nothing else."
        prompt = bbox_string

        client = OpenAI(

            api_key = "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",    
        )
        
        completion = client.chat.completions.create(
        # Use GPT 3.5 as the LLM
        model="gpt-3.5-turbo",
        # Pre-define conversation messages for the possible roles 
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt}
        ]
        )
 
        corrected_list = completion.choices[0].message.content.strip()
        bbox_coords = ast.literal_eval(corrected_list)[0]
        scaled_bbox_coords = draw_bbox(image_size, image_array, bbox_coords, output_path)
        print("Scaled Bbox Coords: ", scaled_bbox_coords)
        return scaled_bbox_coords


def extract_last_frame(video_path):
        
        """ 
        Function to extract last frame from input video

        """

        print(video_path)
        
        input_list = []
        
        cap = cv.VideoCapture(video_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            input_list.append(frame)
        cap.release()
        return input_list[-1]


# if __name__ == "__main__":

#     final = {}

#     for i in [2]:

#         with open("/home/nivi_nath/tmp_reason_ws/src/driver_codes/fm_temporal_reasoning/data/dataset_1/info.json") as f:
#             data = json.load(f)

#         with open("/home/nivi_nath/tmp_reason_ws/src/driver_codes/fm_temporal_reasoning/data/dataset_1/ground_truth/validator.json") as f:
#             instructions = json.load(f)

#         video_file = "/home/nivi_nath/tmp_reason_ws/src/driver_codes/fm_temporal_reasoning/data/dataset_1/videos/"

            
#         output_path = "/home/nivi_nath/tmp_reason_ws/src/driver_codes/fm_temporal_reasoning/data/dataset_1/ground_truth/images/" + str(i) + ".png" 

#         datum = data[str(i)]

#         video_path = video_file + datum['video_path']

#         #query = instructions[str(i)]['instruction']
        
#         query = "all cups"

#         print(query)

#         last_frame = extract_last_frame(video_path)

#         scaled_bbox = send_msg_to_server(last_frame, query, output_path)

#         final[str(i)] = scaled_bbox

#     print(final)