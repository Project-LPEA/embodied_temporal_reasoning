import base64
import requests
from src.core import video_understanding
from PIL import Image, ImageDraw

class Color:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    RESET = '\033[0m'

def validator(params, prompt_GPT, last_frame):  
        
        # for row in last_frame:
        #     for i in range(len(row)):
        #         # Swap Green and Blue channels
        #         row[i] = (row[i][2], row[i][1], row[i][0])



        # OpenAI API Key
        api_key = params['openai_api_key']

        image = Image.fromarray(last_frame)

        image.save("/home/nivi_nath/tmp_reason_ws/src/driver_codes/fm_temporal_reasoning/output/dataset_1/test/test.png")

        #get the current frame from robot instead of path or last frame from the video !?
        # generated_image_path = "/home/nivi_nath/tmp_reason_ws/src/driver_codes/fm_temporal_reasoning/output/dataset_1/test/last_frame4.jpg"

        generated_image_path = "/home/nivi_nath/tmp_reason_ws/src/driver_codes/fm_temporal_reasoning/output/dataset_1/test/test.png"

        with open(generated_image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        #saving the last frame as an image
    
        


        headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}"
                    
                }
        
        payload = {
                    "model": "gpt-4o",
                    "messages": [
                        {
                        "role": "user",
                        "content": [
                            {
                            "type": "text",
                            "text": prompt_GPT
                            },
                            {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                            }
                    ]
                        }
                    ],
                    "max_tokens": 300

                }
        
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

        if response.status_code == 200:
            response_json = response.json()
            
            messages = response_json["choices"][0]["message"]["content"]
            return messages
            
        else:
            print("Error:", response.status_code)

def conversation(params, modified_prompt, last_frame):
    # example_output_LVLM = "the object that was dropped was a blue color cloth in the centre of the table next to a white colored bottle"
    # prompt_GPT = "There is a phrase grounding model that needs to uniquely ground a specific object which will help a robot interact with that object. Given this image, is the context enough of information to uniquely identify an object? Reply with a YES if that object can be distinguished uniquely with the given context. If not, ask more questions as an MCQ to uniquely identify the object using its spatial location or other properties. If object is not present, ask more questions to get the right object. Example - Context: The object that was placed was a bottle. Question: Which one was it? The rightmost, second from right, third from left, right and second from top, right and bottommost, or?"    
    # prompt_LVLM = "Identify the object that was dropped. The video is from a robot's perspective. Return all answers from the robot's perspective. "
    prompt_GPT = "There is a phrase grounding model that needs to uniquely ground a specific object which will help a robot interact with that object.\
                 Given this image, is the context enough of information to uniquely identify an object? Reply with a YES if that object can be distinguished uniquely with the given context. \
                 If not, ask more questions to uniquely identify the object using its spatial location or other properties. If object is not present, ask more questions to get the right object. \
                 Example - Context: The object that was placed was a bottle. Question: Which one was it? The rightmost, second from right, third from left, right and second from top, right and bottommost, or?"    
  

    #Changes required! 
    prompt_LVLM = modified_prompt +  ". The video is from a robot's perspective. Return all answers from the robot's perspective. Do not ask questions"

    flag = False

    history = ""

    count = 0

    while(not flag) :

        # videoLLM_output = prompt_LVLM

        socket = video_understanding.client_socket(params)

        lvlm_output = video_understanding.send_query_to_server(prompt_LVLM, socket)

        if history == "":
            prompt_GPT_main = "Context : " + lvlm_output + "Prompt:" + prompt_GPT
        
        else:
            # prompt_GPT_main = "Consider " + history + "as your knowledge and memory." + "Context : " + lvlm_output + "Prompt:" + prompt_GPT
            prompt_GPT_main = "Context : " + lvlm_output + "Prompt:" + prompt_GPT

        print("prompt GPT main" , prompt_GPT_main)
        gpt_output = validator(params,prompt_GPT_main, last_frame)
        prompt_LVLM = gpt_output + ". The video is from a robot's perspective. Return all answers from the robot's perspective. If options are given, just pick one from those and do not ask questions. "       
        print(Color.GREEN + "Prompt LVLM" + prompt_LVLM + Color.RESET)
        print(Color.RED + "LVLM output: "  + lvlm_output + Color.RESET)
        print(Color.BLUE + "GPT output: "  + gpt_output + Color.RESET)
        
        count = count + 1
        
        if 'YES' in gpt_output or count >= 5:
            flag = True
            # prompt_LVLM = prompt_LVLM

            # output_list += [lvlm_output]
        
        # history += "Video LLM output :" + lvlm_output + "GPT output :" + gpt_output
        history +=  lvlm_output 


    return lvlm_output
    # return lvlm_output + " " + gpt_output