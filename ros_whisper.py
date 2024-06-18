#!/usr/bin/env python3

# -*- coding: utf-8 -*-

import sys
import queue
import socket
import pickle
import sounddevice as sd
from mmap import MAP_SHARED

import rospy
import numpy as np
from std_msgs.msg import String

from matplotlib import pyplot as plt
from colorama import Fore, Style

class RosWhisper():
    """Class which interfaces the whisper model and the ROS system. It
    establishes TCP connection to the remote server. This module accepts 
    the audio signal, sends it to the whisper model and publishes the
    transcripted text
    """

    def __init__(self):
        """Default constructor

        Raises:
            ValueError: if no input device is not found
        """

        self.rate = rospy.Rate(100)

        # IP addr and port number of the server running the whisper model
        self.host = "10.237.20.209"
        #self.host = "192.168.131.11"
        self.port = 65430
        self.data_buffer = []
        self.q = queue.Queue()

        # Get the number of the input device
        self.input_dev_num = sd.query_hostapis()[0]['default_input_device']
        if self.input_dev_num == -1:
            rospy.logfatal('No input device found')
            raise ValueError('No input device found, device number == -1')

        # Get the information about the chosen input device
        device_info = sd.query_devices(self.input_dev_num, 'input')

        # soundfile expects an int, sounddevice provides a float:
        self.samplerate = int(device_info['default_samplerate'])
        rospy.set_param('vosk/sample_rate', self.samplerate)
        self.pub = rospy.Publisher("/speech_recognition/unified_language", \
                                    String, queue_size=3)
        rospy.on_shutdown(self.cleanup)
    
    def cleanup(self):
        """function to be called when the node is terminated
        """
        rospy.logwarn("Shutting down whisper speech recognition node...")
    
    def stream_callback(self, indata, frames, time, status):
        """Callback function for the RawInputStream function to read 
        the
         audio signal coming from the input devices
        """
        if status:
            print(status, file=sys.stderr)  
        self.q.put(bytes(indata))
        
    def speech_recognize(self): 
        """Function which chunks the audio signals and sends to the 
        whisper model running remotely on the server
        """   
        try:
            with sd.RawInputStream(samplerate=self.samplerate, blocksize=16000, \
                                device=self.input_dev_num, dtype='int16', \
                               channels=1, callback=self.stream_callback):
                rospy.logdebug('Started recording')
                print(Fore.GREEN, "\nROS Whisper is ready\n", Style.RESET_ALL)
                while not rospy.is_shutdown():
                    data = self.q.get()
                    self.data_buffer.append(np.frombuffer(data, dtype=np.int16))
                    if np.all(self.data_buffer[-1] == 0) and \
                            len(self.data_buffer) != 1:
                        self.data_buffer = np.array(self.data_buffer)
                        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        s.connect((self.host, self.port))
                        byte_msg = pickle.dumps(self.data_buffer.flatten())
                        print("Sending the request to the Whisper server")
                        s.sendall(byte_msg)
                        s.shutdown(socket.SHUT_WR)
                        data = b""
                        while True:
                            packet = s.recv(1024)
                            if not packet:
                                break
                            data += packet
                        data = pickle.loads(data)
                        print("Transcribed text: ", data)
                        self.publish_msg(data)
                        self.data_buffer = []
                        continue
                    if np.all(self.data_buffer[-1] == 0) and \
                            len(self.data_buffer) == 1:
                        self.data_buffer = []

        except Exception as e:
            exit(type(e).__name__ + ': ' + str(e))
        except KeyboardInterrupt:
            rospy.loginfo("Stopping the whisper speech recognition node...")
            rospy.sleep(1)
            print("node terminated")

    def publish_msg(self, text):
        """Function which published the transcribed text as a ROS message

        Args:
            text (str): transcribed text 
        """
        self.pub.publish(text)
        return


if __name__ == '__main__':
    try:
        rospy.init_node('nlp', anonymous=False)
        rec = RosWhisper()
        rec.speech_recognize()
    except (KeyboardInterrupt, rospy.ROSInterruptException) as e:
        rospy.logfatal("Error occurred! Stopping the vosk speech recognition node...")
        rospy.sleep(1)
        print("node terminated")