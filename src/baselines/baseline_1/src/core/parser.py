import os
import subprocess
import base64
import requests
import yaml



# Past, Present and Future parser
def to_llm(human_input, params, prompts):
    
    """ Parser 1 to manipulate human instruction for our pipeline

    Returns:
        Prompt for video understander and command/ action for robot
    """

    api_key = params['openai_api_key']
    
    prompt_GPT = str(prompts["prompt_parser1"]) + str("human instruction: ") + str(human_input)
    
    
    headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
                
            }
    
    payload = {
                "model": "gpt-3.5-turbo",
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
                "max_tokens": 1000

            }
    
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    #if response.status_code == 200:
    response_json = response.json()
    print("Response json : ", response_json)
        
    messages = response_json["choices"][0]["message"]["content"]
        
    return messages
    # else:
    #     print("Error:", response.status_code)
        

# To extract cogvlm prompt from videollm output
def to_cogVLM(videoLLM_output, params, prompts):

    """ Parser 2 to extract object of interest from output of video understander

    Returns:
        Prompt for Grounder
    """
    
    api_key = params["openai_api_key"]

    prompt_GPT = str(prompts["prompt_parser2"]) + str("Prompt : ") + str(videoLLM_output)

    headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
                
            }
    
    payload = {
                "model": "gpt-4o-mini",
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
                "max_tokens": 1000

            }
    
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    # if response.status_code == 200:
    response_json = response.json()
    messages = response_json["choices"][0]["message"]["content"]
        
        
    return messages
    # else:
    #     print("Error:", response.status_code)



def question_extractor(gpt_output, params, prompts):

    """ Parser 2 to extract object of interest from output of video understander

    Returns:
        Prompt for Grounder
    """
    
    api_key = params["openai_api_key"]

    prompt_GPT = str(prompts["prompt_questionExtractor"]) + str("Text : ") + str(gpt_output)

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

    # if response.status_code == 200:
    response_json = response.json()
    
    messages = response_json["choices"][0]["message"]["content"]
    
    
    return messages
    # else:
    #     print("Error:", response.status_code)
        

        


    