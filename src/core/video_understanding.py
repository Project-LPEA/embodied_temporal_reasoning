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

    print("Calling the VideoChat2 server")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((params['host_ip'], params['port']))

    return s

def send_query_to_server(query, s):

    msg = {"query": query}
    print("Reached before pickling")
    byte_msg = pickle.dumps(msg)
    print("Sending the query thru sockets")
    s.sendall(byte_msg)
    print("Done sending the query")
    s.shutdown(socket.SHUT_WR)

    data = b""

    time.sleep(0.2)

    while True:
        packet = s.recv(2048)   
        print("Packet:", packet)
        if not packet:
            break
        data += packet

    print("Data: ", data)

    data = pickle.loads(data)
    print(data)

    DATA = data

    print("Sending query from VideoChat2:", DATA)
    print("Got the response")

    return data