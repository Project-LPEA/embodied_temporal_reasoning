#!/usr/bin/env python3
import os   
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

class TemporalReasoner():

    def __init__(self):
        print("inside TR init")
        rospy.init_node('TR', anonymous=False)
        print("rospy init TR")
        self.input_list = []
        self.count = 0
        self.video_flag = False
        self.start_time = time.time()
        self.is_video_saved = False
        self.pub = rospy.Publisher('/plan', String, queue_size=10)        
        self.start_time = time.time()

    
    def subscribers(self, params, prompts):

        print("reached subs")

        rospy.Subscriber("/camera/color/image_raw", sensor_msgs.msg.Image, partial(self.img_callback, params=params))

        rospy.Subscriber("/speech_recognition/unified_language", String, partial(self.instr_callback, params=params, prompts = prompts))
        # self.instr_callback('Pick the object that I just placed', params, prompts)
        
    
    def save_video(self, frames_list, params):

        print("Action Started!")
        self.is_video_saved = True
        width = np.shape(frames_list[1])[1]
        height = np.shape(frames_list[1])[0]

        video_file_path = 'input_video.mp4'
        if os.path.exists(video_file_path):
            os.remove(video_file_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        video = cv2.VideoWriter('input_video.avi', fourcc, 25, (width, height))

        for j in range(0,len(frames_list)):
            img = frames_list[j]
            video.write(img)
        
        os.system("ffmpeg -i {input} -ac 2 -b:v 2000k -c:a aac -c:v libx264 -b:a 160k -vprofile high -bf 0 -strict experimental -f mp4 {output}.mp4".format(input = '/home/nivi_nath/tmp_reason_ws/src/driver_codes/fm_temporal_reasoning/input_video.avi', output = '/home/nivi_nath/tmp_reason_ws/src/driver_codes/fm_temporal_reasoning/input_video'))
        video.release()
        print("Action Recorded")
        #self.export_vid()
        #print("copy started")
        os.system('scp /home/nivi_nath/tmp_reason_ws/src/driver_codes/fm_temporal_reasoning/input_video.mp4 helium:/home/niveditha/VideoChat2/video_chat2/dataset/input_video.mp4')
        #print("copy ended")

      
        

    def img_callback(self, msg, params):
        #print("inside TR img_callback")
        width = params["width"]
        height = params["height"]
        color_image = ros_numpy.numpify(msg) 

        data = np.array(Image.fromarray(color_image).resize((width, height)))
        data = data[...,::-1]

        
        if (time.time() - self.start_time) < 5:
            self.input_list.append(data)

        elif not(self.is_video_saved):
            self.save_video(self.input_list, params)

        

    def instr_callback(self, msg, params, prompts):
       print("msg", msg)
    #    data = msg.data
    
       data=msg
       print("instruction: ", data)
       response = self.usage(data, params, prompts) 
       self.pub.publish(json.dumps(response))

    def usage(self, user_prompt, params, prompts):
        print("inside TR usage")
        print("user prompt", user_prompt)
        parser_output = eval(parsing.to_llm(user_prompt, params, prompts))
        print("Parser Output", parser_output)

        # parser_output = parser_output['ground_truth']
        print(parser_output)
        modified_prompt = "Identify " + parser_output["past"]


        videoLLM_output = validator.conversation(params, modified_prompt, self.input_list[-1])
        
        cogvlm_query = parsing.to_cogVLM(videoLLM_output, params, prompts)

        print("CogVLM Query:", cogvlm_query)
        action = parser_output["present"]
        location = ''

        grounding.send_msg_to_server(self.input_list[-1],cogvlm_query)

        return({"action": action, "object": cogvlm_query, "object_context": location})
