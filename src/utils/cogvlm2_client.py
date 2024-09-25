import socket
import numpy as np
import time 
import pickle

# class CogVLM2_Client():

# def __init__(query):

#     self.host = "10.237.20.209"
#     self.port = 65490

#     self.host_name = socket.gethostname()

#     # self.send_query_to_server(query)
#     # print(host_ip)
#     # print("sent total no. of frames")

def send_query_to_server(query, params):

    host = params["cogvlm2_host_ip"]
    port = params["cogvlm2_port"]

    
    print("Calling the CogVLM2 server")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))

    msg = {"query": query}
    print("Reached before pickling")
    byte_msg = pickle.dumps(msg)
    print("Sending the query thru sockets")
    s.sendall(byte_msg)
    print("Done sending the user query")
    s.shutdown(socket.SHUT_WR)

    time.sleep(0.9) 

    data = b""

    time.sleep(0.9) 
    
    while True:
        print("inside")
        packet = s.recv(1024)   
        print("Packet:", packet)
        if not packet:
            break
        data += packet

    # print("Data: ", data)

    data = pickle.loads(data)
    #print(data.keys())
    print(data)

    DATA = data

    print("Received from CogVLM2:", DATA)
    print("Got the response")

    return data
            
# if __name__ == "__main__":

#     query = "describe the video"
    
#     example = CogVLM2_Client(query)
