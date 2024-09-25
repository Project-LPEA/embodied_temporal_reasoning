import base64
import requests
from PIL import Image, ImageDraw
from colorama import init, Fore, Style
import os
from src.core import parsers
import cv2 as cv
import math
import json

def split_video(video_path, output_folder, first_frame_no):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)    # Open the video file

    cap = cv.VideoCapture(video_path)    # Get the total number of frames in the video
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))    # Calculate the interval between frames to extract, ensuring it's at least 1
    
    frame_interval = 1 # Initialize frame count

    frame_count = 0
    extracted_count = 0    

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break      
        if frame_count % frame_interval == 0 : 
            frame_filename = os.path.join(output_folder, f"{extracted_count:05d}.jpg")
            if extracted_count >= int(first_frame_no):
                cv.imwrite(frame_filename, frame)
        
            extracted_count += 1   
            
        frame_count += 1 
    cap.release()

    return frame_count

def frame_counter(video_path, output_folder, time_stamps, first_frame_no):

    saved_count = split_video(video_path, output_folder, first_frame_no)

    new_time_stamps = []
    if saved_count != 100:
        for i in time_stamps:    
            new_time_stamps.append(math.floor(int(i)*saved_count/100))

        new_time_stamps = [f"{num:04}" for num in new_time_stamps]
        return new_time_stamps

    else :
        time_stamps = [f"{num:04}" for num in time_stamps]
        return time_stamps
    
def image_understander(params, prompt_GPT, options_path, time_stamps, images_folder_path):

    # OpenAI API Key
    api_key = params['openai_api_key']

    # Prepare base64 encoded images
    base64_images = []
    image_hashmap = {}

    time_stamps = sorted(time_stamps)

    for timestamp in time_stamps:
        if str(timestamp) not in image_hashmap: 

            image_path = images_folder_path + f"{int(timestamp):05d}" + '.jpg' 
            
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                base64_images.append(base64_image)
            image_hashmap[str(timestamp)] = 1

    if os.path.exists(options_path):
        print("OPTIONS ADDED")
        with open(options_path, "rb") as image_file:
            
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")
            base64_images.append(base64_image)
            

    headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
                
            }
   
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt_GPT
                }
            ]
        }
    ]

    for base64_image in base64_images:
        messages[0]["content"].append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        })

    payload = {
        "model": "gpt-4o",
        #"response_format" : { "type": "json_object" },
        "messages": messages,
        "max_tokens": 1000
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    response_json = response.json()        
    messages = response_json["choices"][0]["message"]["content"]
    # test_json = json.loads(messages)
    return messages
    
    
    # while True:
    #     try:
            
    #     except:
    #         print("GPT error")





