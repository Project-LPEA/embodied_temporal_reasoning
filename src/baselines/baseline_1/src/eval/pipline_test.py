#!/usr/bin/env python3

""" 

Helper file for test_run.py
This file contains functions to execute entire pipeline step by step

"""

import os
import os   
import sys
import cv2 as cv
import json
import io
from ..core import parser, grounder, validator
from ..utils import cogvlm2_client
import json
import numpy as np
import time



def extract_last_frame(video_path, output_path, integer):
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
        
        cv.imwrite(output_path,input_list[integer])
        return input_list[integer]

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
        x1, y1, x2, y2 = round(x1), round(y1), round(x2), round(y2)
        cv.rectangle(image_array, (x1, y1), (x2, y2), color1, thickness)

        x1, y1, x2, y2 = bbox_coords
        cv.rectangle(image_array, (x1, y1), (x2, y2), color2, thickness)
        cv.imwrite(save_img, image_array)


    
def test_pipeline(video_path, instruction, output_path, params, prompts, pipeline_output, bbox_gt, images_folder_path):

    """ Function to execute entire pipeline step by step

    Returns:
        Final bounding box coordinates of grounded object
    """

    initial_frame_path = output_path[:-4] + "first.png"
    gpt_image_path = output_path[:-4] + "final.png"
    last_frame = extract_last_frame(video_path, gpt_image_path,-1)
    initial_frame = extract_last_frame(video_path, initial_frame_path,0)

    for row in initial_frame:
            for i in range(len(row)):
                row[i] = (row[i][2], row[i][1], row[i][0])


    os.system('scp ' + video_path + ' ' + params["server_video_path"])

    parser_output = json.loads(parser.to_llm(instruction, params, prompts))
    with open(pipeline_output, 'a') as file: file.write("Instruction:" + instruction + "\n" + "Parser1 Output" + str(parser_output) + "\n" + "\n ")
    modified_prompt = parser_output["past"] + ". Give the object, object properties, and its exact relative spatial location(in words) from camera's point of view. "
    with open(pipeline_output, 'a') as file: file.write("Modifies Prompt: " + modified_prompt + "\n" )

    tr_start = time.time()
    

        
    if params['baseline_number']== 1:
        videoLLM_output = cogvlm2_client.send_query_to_server(modified_prompt)

    if params['approach']== 2:
        videoLLM_output = validator.conversation(params, modified_prompt, last_frame, gpt_image_path, prompts, pipeline_output)


    cogvlm_query = parser.to_cogVLM(videoLLM_output, params, prompts)
    print("Parser 2 Output: ", cogvlm_query)
    with open(pipeline_output, 'a') as file: file.write("LLaVA output: " + videoLLM_output + "\n" + "TR time: " + str(tr_time) + "\n" + 
                                                        "Parser2 Output: " + str(cogvlm_query) + "\n" + "\n ")

    grounding_start = time.time()
    bbox_coords = grounder.send_msg_to_server(last_frame,cogvlm_query, output_path)
    grounding_end = time.time()
    grounding_time = grounding_end-grounding_start
    with open(pipeline_output, 'a') as file: 
        file.write("Grounding time: " + str(grounding_time) + "\n")
    draw_groundtruth(last_frame, output_path, bbox_coords, bbox_gt )
    with open(pipeline_output, 'a') as file: file.write("Bbox coordinates " + str(bbox_coords) + "\n ")
    return bbox_coords
