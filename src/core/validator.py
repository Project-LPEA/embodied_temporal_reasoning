import base64
import requests
from src.core import video_understander
from PIL import Image, ImageDraw
from colorama import init, Fore, Style
import os
from src.core import parser



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
        
        """ Function to call the validator LVLM

        Returns:
            Output of validator LVLM
        """
        

        # OpenAI API Key
        api_key = params['openai_api_key']

        image = Image.fromarray(last_frame)

        image.save("/home/nivi_nath/tmp_reason_ws/src/driver_codes/fm_temporal_reasoning/output/test.png")


        generated_image_path = "/home/nivi_nath/tmp_reason_ws/src/driver_codes/fm_temporal_reasoning/output/test.png"

        with open(generated_image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        if os.path.exists(generated_image_path):
            os.remove(generated_image_path)

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

def conversation(params, modified_prompt, last_frame, prompts, conversation_output):

    """ Function to set up conversation between video understander and validator

    Returns:
        Object of interest concluded from conversation to be sent for grounding
    """

    # prompt_GPT = prompts['baseprompt_llm']  

    for row in last_frame:
        for i in range(len(row)):
            row[i] = (row[i][2], row[i][1], row[i][0]) 
  

    #Changes required! 
    prompt_LVLM = modified_prompt +  prompts['baseprompt_lvlm']

    flag = False

    history = ""

    count = 0

    while(not flag) :


        socket = video_understander.client_socket(params)

        lvlm_output = video_understander.send_query_to_server(prompt_LVLM, socket)

        
        lvlm_output = eval(parser.to_cogVLM(lvlm_output, params, prompts))['output']

        if history == "":
            # prompt_GPT_main = "CONTEXT : " + lvlm_output + "Prompt:" + prompt_GPT
            prompt_GPT_main = prompts['baseprompt_llm_1'] + lvlm_output + prompts['baseprompt_llm_2'] 
        
        else:
            prompt_GPT_main = "Consider " + history + "as your knowledge and memory." + prompts['baseprompt_llm_1'] + lvlm_output + prompts['baseprompt_llm_2'] 
            # "Consider " + history + "as your knowledge and memory."
            # prompt_GPT_main = "CONTEXT : " + lvlm_output + "Prompt:" + prompt_GPT

        print("prompt GPT main" , prompt_GPT_main)
        gpt_output = validator(params,prompt_GPT_main, last_frame)

        print(Color.RED + "GPT Prompt: ", prompt_GPT_main)

        # print(Color.RED + "LVLM output: "  + lvlm_output + Color.RESET)
        # print(Color.BLUE + "GPT output: "  + gpt_output + Color.RESET)


        count = count + 1

        with open(conversation_output, 'a') as file: 

            init()

            lvlm_output_color = "LVLM output: "  + lvlm_output
            
            gpt_output_color = "GPT output: "  + gpt_output 

            file.write("Count: " + str(count) + "\n" + \
                         lvlm_output_color + "\n" + \
                        gpt_output_color + "\n\n" 
                       )

        

        
        if 'YES' in gpt_output or count>=7:
            flag = True
            # prompt_LVLM = prompt_LVLM

            # output_list += [lvlm_output]
        else:
            gpt_output = parser.question_extractor(gpt_output, params, prompts)
            prompt_LVLM = gpt_output + prompts['baseprompt_lvlm']     

        
        history += "Video LLM output :" + lvlm_output + "GPT output :" + gpt_output
        # history +=  lvlm_output 


    return lvlm_output + gpt_output
    # return lvlm_output + " " + gpt_output