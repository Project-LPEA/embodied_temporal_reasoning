#Sending Query to the VideoChat2 server

import cv2, imutils, socket
import numpy as np
import time 
import base64
import socket
import rospy
import pickle
import sys
import json

def client_socket(params):

    """ 
    Function to set up connection with video understander LVLM
    """

    print("Calling the VideoChat2 server")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((params['host_ip'], params['port']))

    return s

def send_query_to_server(query, s):

    """ 
    Function to send query to video understander LVLM
    """

    msg = {"query": query}
    byte_msg = pickle.dumps(msg)
    s.sendall(byte_msg)
    s.shutdown(socket.SHUT_WR)

    data = b""

    time.sleep(0.2)

    while True:
        packet = s.recv(2048)   
        print("Packet:", packet)
        if not packet:
            break
        data += packet


    data = pickle.loads(data)

    DATA = data

    print("Sending query from VideoChat2:", DATA)

    return data