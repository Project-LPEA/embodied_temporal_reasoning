#!/usr/bin/env python3

""" 

Helper file for test_run.py
This file contains functions to execute entire pipeline step by step

"""

import os
import sys

from ..utils import sam2_client, cogvlm2_client, draw_groundtruth, interval_extracter
import json
from ..core import image_understander, parsers, options_generator
import numpy as np
import re
import shutil
import time 

class Color:
    """ 
    Function to colour output in cli
    """
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    RESET = '\033[0m'

#Takes in video and temporal question and outputs the interval where the interaction happens
def event_localization(video_path, parser_output, partial_obs, interval, mask_bboxes, dino_index,pipeline_output, params):

    if not partial_obs:
        os.system('scp ' + video_path + ' sandeep@10.237.20.209:/home/sandeep/CogVLM2/video_demo/test.mp4')

        prompt_CogVLM2 = parser_output["temporal_question"]
        response_CogVLM2 = cogvlm2_client.send_query_to_server(prompt_CogVLM2, params)  
        print("CogVLM2 response: ", response_CogVLM2)
        time_stamp = re.findall(r'\d+', response_CogVLM2)
        time_stamp = int(time_stamp[0]) 
        interval_extracter_output = interval_extracter.interval_extracter(time_stamp, video_path)
        print("Interval before subsampling")
        interval_extracter_output = interval_extracter_output[3::4]
        with open(pipeline_output, 'a') as file: 
            file.write("Time Instant by CogVLM2: " + str(time_stamp) + "\n"\
                       "Interval Extracted: " + str(interval_extracter_output) + "\n")


    else: 
        interval_extracter_output = []
        all_keys = list(mask_bboxes.keys())
        start = 0
        end = start+1
        for i in range(len(all_keys)-1, -2,-1):
            if mask_bboxes[all_keys[i]] != []: 
                start_index = int(max(0, i-2))
                start = all_keys[start_index] + int(interval[dino_index])
                end_index = int(min(i+2, all_keys[-1]))
                end = int(all_keys[end_index]) + int(interval[dino_index])
                break 
        
        print("start and end:", start, end)
        for i in range(start, end +1):
            interval_extracter_output.append(f"{int(i):05d}")
        # interval_extracter_output = interval_extracter_output[3::4]
        with open(pipeline_output, 'a') as file: 
            file.write("PO Interval Extracted: " + str(interval_extracter_output) + "\n")

    return interval_extracter_output

def class_detection(params, prompts, parser_output, images_folder_path, options_path, interval, partial_obs, object_class, pipeline_output):

    if partial_obs: prompt_class_detector = "Where did " + object_class + " go? Give the object that partially or completely hid the " + object_class + ". Return only the object that hid the " + object_class + ". Return that object in 1 word."
    else: prompt_class_detector = str(prompts['prompt_ImageUnderstander1']) + parser_output["interaction"] + str(prompts['prompt_ImageUnderstander2']) + parser_output["object"] + str(prompts['prompt_ObjectExtractor']) + ". Return that object in 1 word."
    object_class = image_understander.image_understander(params, prompt_class_detector, options_path, interval, images_folder_path)
    # with open(dino_input_txt, 'a') as file:
    #     file.write(object_class)
    with open(pipeline_output, 'a') as file: 
            file.write("Class: " + str(object_class) + "\n")

    return [object_class, prompt_class_detector]

def grounding(params, object_class, images_folder_path, options_path, interval, prompt_class_detector, dino_index,pipeline_output):

    last_frame_path = images_folder_path + f"{int(interval[dino_index]):05d}" + '.jpg'
    print("Input to Dino: ",object_class)
    all_bboxes = options_generator.dino(object_class, last_frame_path, options_path, bbox_threshold=0.2)
    object_label = ""
    prompt_object_detector = prompt_class_detector + " Return the corresponding object label number visible in the last frame. Always return object label number. If the object does not have a label or a bounding box in the last frame, reply with a NO."

    if len(all_bboxes) > 1:
        object_label = image_understander.image_understander(params, prompt_object_detector, options_path, interval, images_folder_path)

    with open(pipeline_output, 'a') as file: 
            file.write("All bboxes: " + str(all_bboxes) + "\n"\
                       "Object Label: " + str(object_label) + "\n")
    return [all_bboxes, object_label]


def tracker(interval, dino_index, images_folder_path, grounding_output, total_frames, pipeline_output, params):

    for i in range(int(interval[0]), int(interval[dino_index])-1):
        os.remove(images_folder_path + f"{int(i):05d}" + ".jpg")

    print("total frames: ", total_frames)
    print("interval dino index: ", int(interval[dino_index]))

    length_frames_dir = total_frames - int(interval[dino_index]) + 1

    # To sample every fifth frame for SAM2
    sampled_frames_dir = images_folder_path[:-7]  + 'sampled_frames_dir/'
    os.makedirs(sampled_frames_dir)

    if length_frames_dir >= 5:
        for i in range(int(interval[dino_index]), total_frames-1, 5):
            src = images_folder_path + f"{int(i):05d}" + '.jpg'
            dst = sampled_frames_dir + f"{int(i):05d}" + '.jpg'
            shutil.copy2(src, dst)  # Copy the image
    else: 
        for filename in os.listdir(images_folder_path):
            if filename.endswith('.jpg'):  # Adjust the file extension as needed
                source_file = os.path.join(images_folder_path, filename)
                destination_file = os.path.join(sampled_frames_dir, filename)
                shutil.copy2(source_file, destination_file)

    entries = os.listdir(sampled_frames_dir)
    file_count = sum(1 for entry in entries if os.path.isfile(os.path.join(sampled_frames_dir, entry)))
    
    remote_server = 'niveditha@10.237.23.193'
    remote_dir = '/home/niveditha/Grounded-SAM-2/notebooks/videos/test/'

    remote_command = f"ssh {remote_server} 'rm -rf {remote_dir} && mkdir -p {remote_dir}'"

    # Execute the command to prepare the remote directory
    os.system(remote_command)
    os.system('scp -rq ' + sampled_frames_dir + ' niveditha@10.237.23.193:/home/niveditha/Grounded-SAM-2/notebooks/videos/test/frames/' )
    shutil.rmtree(sampled_frames_dir)

    point = (0,0)
    bbox = np.array(grounding_output, dtype=np.float32)
    mask_bboxes = sam2_client.send_query_to_server(bbox ,point, file_count, params)
    _, last_bbox = mask_bboxes.popitem()

    with open(pipeline_output, 'a') as file: 
            file.write("Mask bboxes: " + str(mask_bboxes) + "\n"\
                       "Last bbox: " + str(last_bbox) + "\n")

    return [last_bbox, mask_bboxes]

def test_pipeline(video_path, instruction, output_path, images_folder_path, params, prompts, pipeline_output, bbox_gt):

    parser_output = json.loads(parsers.to_lvlm(instruction, params, prompts))
    options_path = images_folder_path[:-7]  + 'with_boxes.png'
    last_bbox = []
    partial_obs = False
    interval = []
    mask_bboxes = []
    object_class = ""


    # This loop encourages Partial Observability
    while len(last_bbox) == 0:

        dino_index = -1

        el_start = time.time()
        interval = event_localization(video_path, parser_output, partial_obs, interval, mask_bboxes, dino_index,pipeline_output, params)
        print("Interval: ", interval)
        total_frames = image_understander.split_video(video_path, images_folder_path, interval[0])
        el_end = time.time()
        el_time = el_end - el_start
        
        with open(pipeline_output, 'a') as file: 
            file.write("EL time: " + str(el_time) + "\n")

        # This loop ensures grounded output is not None  
        # and ensures that the grounder grounds the correct objectof-interest by:
    
        #   (1) going a few frames back to retrieve clear image
        #   (2) changing input to grounder by calling class detector once in 4 times
        #   (3) if not None and doesn't ground correct object, repeat the whole process by going back a few frames
        # Maximum tries for inner loop = 5
        # Maximum tries for outer loop = 3

        
        total_chances = 1
        object_label = "NO"

        stage2_start = time.time()
        while "NO" in object_label and total_chances <= 3:

            if os.path.exists(options_path):
                os.remove(options_path)

            dino_index = -1
            all_bboxes = {}
            class_grounding_chance = 4

            while len(all_bboxes)==0 and class_grounding_chance <= 14:

                dino_index = dino_index - 1

                if class_grounding_chance%4==0 or dino_index == -len(interval):

                    [object_class, prompt_class_detector] = class_detection(params, prompts, parser_output, images_folder_path, options_path, interval, partial_obs, object_class,pipeline_output)
                    dino_index = -1

                [all_bboxes, object_label] = grounding(params, object_class, images_folder_path, options_path, interval, prompt_class_detector, dino_index,pipeline_output)
                class_grounding_chance += 1

            total_chances += 1

        stage2_end = time.time()
        stage2_time = stage2_end-stage2_start
        with open(pipeline_output, 'a') as file: 
            file.write("Stage 2 time: " + str(stage2_time) + "\n")


        # Extracting the input for Tracker
        if len(all_bboxes) == 0: grounding_output =[0, 0, 0, 0]
        elif len(all_bboxes) == 1:grounding_output = list(all_bboxes.values())[0]
        else: 
            object_label = re.findall(r'\d+', object_label)
            grounding_output = all_bboxes[ "Object " + str(object_label[0])]

        #Tracking
        tracking_start = time.time()
        [last_bbox, mask_bboxes] = tracker(interval, dino_index, images_folder_path, grounding_output, total_frames,pipeline_output, params)
        tracking_end = time.time()
        tracking_time = tracking_end-tracking_start
        with open(pipeline_output, 'a') as file: 
            file.write("Tracking time: " + str(tracking_time) + "\n")

        if last_bbox == []: partial_obs = True
        else: partial_obs = False

    
    final_bbox = last_bbox[-1]
    last_frame_video = images_folder_path + f"{int(total_frames-1):05d}" + '.jpg'
    draw_groundtruth.draw_groundtruth(last_frame_video, output_path, final_bbox, bbox_gt)
    # shutil.rmtree(images_folder_path)
    
    return final_bbox
 

        

        
        


