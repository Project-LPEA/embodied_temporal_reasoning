import base64
import requests
from PIL import Image, ImageDraw
from colorama import init, Fore, Style
import os
import parser
from ..utils import cogvlm2_client




class Color:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    RESET = '\033[0m'

def validator(params, prompt_GPT, last_frame, image_path):  
        
        """ Function to call the validator LVLM

        Returns:
            Output of validator LVLM
        """
        

        # OpenAI API Key
        api_key = params['openai_api_key']

        image = Image.fromarray(last_frame)

        # image.save("/home/nivi_nath/tmp_reason_ws/src/driver_codes/fm_temporal_reasoning/output/test.png")
        image.save(image_path)


        generated_image_path = image_path

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
            print("response json: ", response_json)
            messages = response_json["choices"][0]["message"]["content"]
            print("Messages: ", messages)
            return messages
            
        else:
            print("Error:", response.status_code)

def conversation(params, modified_prompt, last_frame, image_path, prompts, conversation_output):

    """ Function to set up conversation between video understander and validator

    Returns:
        Object of interest concluded from conversation to be sent for grounding
    """

    prompt_LVLM =  modified_prompt

    flag = False

    history = ""

    count = 0

    while(not flag) :

        lvlm_output = cogvlm2_client.send_query_to_server(prompt_LVLM)
        
        lvlm_output = parser.to_cogVLM(lvlm_output, params, prompts)

        if history == "":
            prompt_GPT_main = prompts['baseprompt_llm_1'] + lvlm_output + prompts['baseprompt_llm_2'] 
        else:
            prompt_GPT_main = "Consider " + history + "as your knowledge and memory." + prompts['baseprompt_llm_1'] + lvlm_output + prompts['baseprompt_llm_2'] 
    

        print("prompt GPT main" , prompt_GPT_main)
        gpt_output = validator(params,prompt_GPT_main, last_frame, image_path)

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

        else:
            gpt_output = parser.question_extractor(gpt_output, params, prompts)
            prompt_LVLM = modified_prompt + ". " + gpt_output
        
        print("Prompt LVLM:", prompt_LVLM)

        
        history += "Video LLM output :" + lvlm_output + "GPT output :" + gpt_output


    return lvlm_output