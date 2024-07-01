#!/usr/bin/env python3

""" 

Helper file for test_run.py
This file contains functions to execute entire pipeline step by step

"""

import os
import os   
import sys
sys.path.append('/home/nivi_nath/tmp_reason_ws/src/driver_codes/fm_temporal_reasoning/')
import cv2 as cv
import json
import io
from src.core import video_understander
from functools import partial
from src.core import validator
from src.core import parser
import json
import pathlib
from src.core import grounder
import numpy as np



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
            input_list.append(frame)
        cap.release()
        return input_list[-1]

def slow_video(video_path, slow_video_path):
    cap = cv.VideoCapture(video_path)

    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))

    path = slow_video_path
    fourcc = cv.VideoWriter_fourcc(*'mp4v')

    output = cv.VideoWriter(path, fourcc, 10, (width, height))

    while True:
        ret, frame = cap.read()
        if type(frame) == type(None):
            break
        cv.imshow("frame",frame)
        # print("here")
        output.write(frame)
        k = cv.waitKey(24)
        if k==ord("q"):
            break

    cap.release()
    output.release()
    cv.destroyAllWindows()


def draw_groundtruth(image_array, output_path, bbox_coords, ground_truth_coords):
             
        """ Function to draw bounding box coordinates on image

        Returns:
            image with grounded object of interest and bounding box coordinates
        """

        image_array = np.array(image_array)
        save_img = output_path
        
        color1 = (0,255,0)
        color2 = (255,0, 0)
        thickness = 2
        x1, y1, x2, y2 = ground_truth_coords
        cv.rectangle(image_array, (x1, y1), (x2, y2), color1, thickness)

        x1, y1, x2, y2 = bbox_coords
        cv.rectangle(image_array, (x1, y1), (x2, y2), color2, thickness)
        cv.imwrite(save_img, image_array)


    
def test_pipeline(video_path, instruction, output_path, params, prompts, pipeline_output, bbox_gt):

    """ Function to execute entire pipeline step by step

    Returns:
        Final bounding box coordinates of grounded object
    """

    slowed_video_path = '/home/nivi_nath/tmp_reason_ws/src/driver_codes/fm_temporal_reasoning/data/dataset_1/videos/' + 'slowed' + '1.mp4'
    
    # slow_video(video_path, slowed_video_path)
   

    ##########################################################################

    last_frame = extract_last_frame(video_path)
    

    
    os.system('scp ' + video_path + ' niveditha@10.237.23.193:/home/niveditha/VideoChat2/video_chat2/dataset/input_video.mp4')

   
    parser_output = eval(parser.to_llm(instruction, params, prompts))

    with open(pipeline_output, 'a') as file: file.write("Instruction:" + instruction + "\n" + "Parser1 Output" + str(parser_output) + "\n" + "\n ")

    modified_prompt = parser_output["past"]

    videoLLM_output = validator.conversation(params, modified_prompt, last_frame, prompts, pipeline_output)
    
    cogvlm_query = eval(parser.to_cogVLM(videoLLM_output, params, prompts))

    print("Parser 2 Output: ", cogvlm_query)
    
    cogvlm_query = cogvlm_query["output"]

    cogvlm_query = parser.to_cogVLM(videoLLM_output, params, prompts)
    cogvlm_query = eval(cogvlm_query)['output'] 


    with open(pipeline_output, 'a') as file: 

        file.write("Parser2 Input: " + videoLLM_output + "\n" + "Parser2 Output: " + str(cogvlm_query) + "\n" + "\n ")
    
    


    bbox_coords = grounder.send_msg_to_server(last_frame,cogvlm_query, output_path)
    draw_groundtruth(last_frame, output_path, bbox_coords, bbox_gt )

    with open(pipeline_output, 'a') as file: 

        file.write("Bbox coordinates " + str(bbox_coords) + "\n ")

    return bbox_coords

