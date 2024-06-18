import os
import subprocess
import base64
import requests
import yaml
#from src.utils import prompts_loader


# Past, Present and Future parser
def to_llm(human_input, params, prompts):


    api_key = params['openai_api_key']
    # print("prompt loader", prompts_loader.prompt_loader("prompt_parser1", params))
    # prompt_GPT = str(prompts_loader.prompt_loader("prompt_parser1", {})) + str("human instruction: ") + str(human_input)
    print("prompt loader", prompts["prompt_parser1"])
    prompt_GPT = str(prompts["prompt_parser1"]) + str("human instruction: ") + str(human_input)
    
    
    headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
                
            }
    
    payload = {
                "model": "gpt-4",
                "messages": [
                    {
                    "role": "user",
                    "content": [
                        {
                        "type": "text",
                        "text": prompt_GPT
                        },
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

# To extract cogvlm prompt from videollm output
def to_cogVLM(videoLLM_output, params, prompts):
    api_key = params["openai_api_key"]

    # prompt_GPT = videoLLM_output + prompts_loader.prompt_loader("prompt_parser2",params)
    prompt_GPT = videoLLM_output + prompts["prompt_parser2"]

    headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
                
            }
    
    payload = {
                "model": "gpt-4",
                "messages": [
                    {
                    "role": "user",
                    "content": [
                        {
                        "type": "text",
                        "text": prompt_GPT
                        },
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




        

        


    