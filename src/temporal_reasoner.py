#!/usr/bin/env python3

""" 
Helper file of run.py (the main execution file)
This file contains functions to execute entire temporal reasoning pipeline step by step

"""

import os   
import cv2
import numpy as np
import time
import rospy
import json
from std_msgs.msg import String
from PIL import Image
import sensor_msgs
import io
from cv_bridge import CvBridge  
from src.core import video_understander
np.float = np.float64
import ros_numpy
from functools import partial
from src.core import validator
from src.core import parser
from src.core import grounder
from signal import signal,SIGPIPE, SIG_DFL
signal(SIGPIPE, SIG_DFL)

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

class TemporalReasoner():

    def __init__(self):

        """ 
        Initializes ros node and defines other elements of pipeline

        """

        rospy.init_node('TR', anonymous=False)
    
        self.input_list = []
        self.count = 0
        self.video_flag = False
        self.start_time = time.time()
        self.is_video_saved = False
        self.pub = rospy.Publisher('/plan', String, queue_size=10)        
        self.start_time = time.time()

    
    def subscribers(self, params, prompts):
        """ 
        Function to subscribe to ros nodes (audio and camera modules)
        """

        rospy.Subscriber("/camera/color/image_raw", sensor_msgs.msg.Image, partial(self.img_callback, params=params))

        rospy.Subscriber("/speech_recognition/unified_language", String, partial(self.instr_callback, params=params, prompts = prompts))        
    
    def save_video(self, frames_list, params):

        """ 
        Function to save incoming camera stream as video and send it to server

        """

        print("Action Started!")
        self.is_video_saved = True
        width = np.shape(frames_list[1])[1]
        height = np.shape(frames_list[1])[0]

        # video_mp4_path = 'input_video.mp4'
        video_mp4_path = os.path.join(params['output_folder_path'], "input_video.mp4")
        video_avi_path = os.path.join(params['output_folder_path'], "input_video.avi")

        if os.path.exists(video_mp4_path):
            os.remove(video_mp4_path)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        video = cv2.VideoWriter(video_avi_path, fourcc, 25, (width, height))

        for j in range(0,len(frames_list)):
            img = frames_list[j]
            video.write(img)
        
        os.system("ffmpeg -i {input} -ac 2 -b:v 2000k -c:a aac -c:v libx264 -b:a 160k -vprofile high -bf 0 -strict experimental -f mp4 {output}".format(input = video_avi_path, output = video_mp4_path))
        video.release()
        print("Action Recorded")
       
        os.system('scp ' + video_mp4_path + ' niveditha@10.237.23.193:/home/niveditha/VideoChat2/video_chat2/dataset/input_video.mp4')
        

    def img_callback(self, msg, params):

        """ 
        Function to set a bound to recorder video length
        """


        # width = params["width"]
        # height = params["height"]
        color_image = ros_numpy.numpify(msg) 

        # data = np.array(Image.fromarray(color_image).resize((width, height)))
        data = np.array(Image.fromarray(color_image))
        data = data[...,::-1]

        
        if (time.time() - self.start_time) < params['video_length']:
            self.input_list.append(data)

        elif not(self.is_video_saved):
            self.save_video(self.input_list, params)


    def instr_callback(self, msg, params, prompts):
       
       """ 
       Function to publish to audio ros node
       """

       data=msg
       response = self.usage(data, params, prompts) 
       self.pub.publish(json.dumps(response))

    def usage(self, user_prompt, params, prompts):

        """ Function to execute temporal reasoner pipeline step by step

        Returns:
            Final bounding box coordinates of grounded object of interest
        """

        parser_output = eval(parser.to_llm(user_prompt, params, prompts))
        modified_prompt = parser_output["past"]

        if os.path.exists(os.path.join(params['output_folder_path'], 'output.txt')):
            os.remove(os.path.join(params['output_folder_path'], 'output.txt'))

        with open(os.path.join(params['output_folder_path'], 'output.txt'), 'a') as file:

            file.write("Parser 1 Output: " + str(parser_output) + "\n\n" )
        

        videoLLM_output = validator.conversation(params, modified_prompt, self.input_list[-1], prompts, os.path.join(params['output_folder_path'], 'output.txt'))
        cogvlm_query = eval(parser.to_cogVLM(videoLLM_output, params, prompts))
        cogvlm_query = cogvlm_query['output']
        bbox_coords = grounder.send_msg_to_server(self.input_list[-1],cogvlm_query, os.path.join(params['output_folder_path'], 'output.png'))

        with open(os.path.join(params['output_folder_path'], 'output.txt'), 'a') as file:

            file.write(
                        
                       "Parser 2 Output: " + str(cogvlm_query) + "\n\n" \

                        "Grounding Output: " + str(bbox_coords)

                       )

        return({"bbox_coords": bbox_coords })
