#!/usr/bin/env python3


"""Run file to execute the temporal reasoning pipeline. The parameters
of the pipeline can be set by passing arguments to this file.
"""

# Library imports
import argparse
import yaml
import os
import pathlib
# from . import ros_whisper
import rospy
import multiprocessing

import os   
import sys
sys.path.append('/home/nivi_nath/tmp_reason_ws/src/driver_codes/fm_temporal_reasoning/')
import cv2
import numpy as np
# import core.parsers as parsers
import time
import rospy
import json
from std_msgs.msg import String
from PIL import Image
import sensor_msgs
import io
from cv_bridge import CvBridge  
from src.core import video_understanding
np.float = np.float64
import ros_numpy
from functools import partial
from src.core import validator
from src.core import parsing
import json

from src.core import grounding

from signal import signal,SIGPIPE, SIG_DFL
signal(SIGPIPE, SIG_DFL)
#from src.utils import ros_whisper


#Remove or Modify later
class Color:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    RESET = '\033[0m'

def load_params():
    """Loads the parameter

    Returns:
        dict: parameters of the localization pipeline
    """
    # current_dir_path = pathlib.Path.cwd()
    params = {}
    params_path = "/home/nivi_nath/tmp_reason_ws/src/driver_codes/fm_temporal_reasoning/config/params.yaml"
    with open(params_path) as stream:
        try:
            params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return params

def load_prompts():
    """Loads the prompts

    Returns:
        dict: parameters of the localization pipeline
    """
    # current_dir_path = pathlib.Path.cwd()
    prompts = {}
    prompts_path = "/home/nivi_nath/tmp_reason_ws/src/driver_codes/fm_temporal_reasoning/config/prompts.yaml"
    with open(prompts_path) as stream:
        try:
            prompts = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return prompts

class TR_pipeline_testing():

    def __init__(self):

        self.input_list = []

        
    
    def extract_last_frame(self, video_path):
        cap = cv2.VideoCapture(video_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            self.input_list.append(frame)
        cap.release()
        return self.input_list[-1]

    
    def test_pipeline(self, video_path, instruction, output_path, params, prompts):

        last_frame = self.extract_last_frame(video_path)
        
       
        os.system('scp ' + video_path + ' helium:/home/niveditha/VideoChat2/video_chat2/dataset/input_video.mp4')

      
        print("inside TR usage")
        print("user prompt", instruction)
        parser_output = eval(parsing.to_llm(instruction, params, prompts))
        print("Parser Output", parser_output)

        # parser_output = parser_output['ground_truth']
        print(parser_output)
        modified_prompt = "Identify " + parser_output["past"]


        videoLLM_output = validator.conversation(params, modified_prompt, last_frame)
        
        cogvlm_query = parsing.to_cogVLM(videoLLM_output, params, prompts)

        print("CogVLM Query:", cogvlm_query)
        action = parser_output["present"]
        location = ''

        grounding.send_msg_to_server(last_frame,cogvlm_query, output_path)

        # return({"action": action, "object": cogvlm_query, "object_context": location})

    
if __name__ == '__main__':

    prompts = load_prompts()
    params = load_params()

    video_folder = "/home/nivi_nath/tmp_reason_ws/src/driver_codes/fm_temporal_reasoning/data/dataset_1/videos/"
    output_folder = "/home/nivi_nath/tmp_reason_ws/src/driver_codes/fm_temporal_reasoning/output/dataset_1/final/"

    with open('/home/nivi_nath/tmp_reason_ws/src/driver_codes/fm_temporal_reasoning/data/dataset_1/info.json') as f:
        data = json.load(f)
    
    for i in range(45, len(data)+1):

        datum = data[str(i)]

        video_path = os.path.join(video_folder, datum['video_path'])
        instruction = datum['instruction']
        output_path = os.path.join(output_folder, str(i) + ".png")
        print(os.path.join(output_folder, str(i)+ ".png"))
        pipeline = TR_pipeline_testing()
        pipeline.test_pipeline(video_path, instruction, output_path, params, prompts)





        
