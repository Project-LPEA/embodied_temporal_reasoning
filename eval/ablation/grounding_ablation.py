#!/usr/bin/env python3   

import numpy
import socket
import rospy
from std_msgs.msg import String
import openai
import cv2 as cv
import pickle
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import sys
import numpy as np
import time
import json
from PIL import Image
import ast
from openai import OpenAI
# import core.bbox


def send_msg_to_server(image_path, recv_query):
        
        image = Image.open(image_path)
        image = np.array(image)

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
        #print(data.keys())
        print("Data", data)
        filter_object_name_list_from_dino(image.shape, image, data)
        # print("Corrected List: ", corrected_list)
        # bbox = ast.literal_eval(corrected_list)[0]
        # print("BBox: ", bbox)
        # return bbox

        # BBox.draw_bbox(image_size, image_array, bbox, "/home/nivi_nath/catkin_ws/src/tmp_reason/src/driver_codes/bbox_frame4.jpg")

        
        print("Got the response")
        return data

def draw_bbox(image_size, image_array, bbox_coords, output_path):
        color = (255,0,0)
        thickness = 2
        print(image_size)
        image_x = image_size[1]
        image_y = image_size[0]
        x1, y1, x2, y2 = bbox_coords 
        x1 = int((x1/1000) * (image_x))
        y1 = int((y1/1000) * (image_y))
        x2 = int((x2/1000) * (image_x))
        y2 = int((y2/1000) * (image_y))
        print(([x1, y1, x2, y2]))
        print(type(image_array))
        image_array = np.array(image_array)
        cv.rectangle(image_array, (x1, y1), (x2, y2), color, thickness)
        save_img = output_path
        cv.imwrite(save_img, image_array)


def filter_object_name_list_from_dino(image_size, image_array, bbox_string):

        # openai.api_key = "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"
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
        print(completion.choices[0].message.content.strip())  
        corrected_list = completion.choices[0].message.content.strip()
    
        #corrected_list = self.filter_object_name_list_from_dino(data)
        print("Corrected List: ", corrected_list)
        bbox_coords = ast.literal_eval(corrected_list)[0]
        print("BBox: ", bbox_coords)

        draw_bbox(image_size, image_array, bbox_coords, "/home/nivi_nath/tmp_reason_ws/src/driver_codes/fm_temporal_reasoning/src/eval/ground_truth/final/62.png")
        
        
if __name__ == '__main__':
      
      send_msg_to_server("/home/nivi_nath/tmp_reason_ws/src/driver_codes/fm_temporal_reasoning/data/dataset_1/last_frames/33_last_frame.jpg", "center cloth")