#!/usr/bin/env python3

"""Run file to execute the temporal reasoning pipeline using LITA. The parameters
of the pipeline can be set by passing arguments to this file.
"""

# Library imports
import argparse
import yaml
import pathlib
from src import temporal_reasoner
from src.temporal_reasoner import TemporalReasoner
import rospy


def load_params():
    """Loads the parameter

    Returns:
        dict: parameters of the localization pipeline
    """
    current_dir_path = pathlib.Path.cwd()
    params = {}
    params_path = str(current_dir_path) + "/config/params.yaml"
    with open(params_path) as stream:
        try:
            params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return params

def load_prompts():
    """Loads the prompts

    Returns:
        dict: parameters of the localization pipeline
    """
    current_dir_path = pathlib.Path.cwd()
    prompts = {}
    prompts_path = str(current_dir_path) + "/config/prompts.yaml"
    with open(prompts_path) as stream:
        try:
            prompts = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return prompts


def update_params(args):
    """Function to update the parameters

    Args:
        args (argparse): arguments from CLI
    """
    if args.openai_api_key != None:
        params['openai_api_key'] = args.openai_api_key
    if args.host_ip != None:
        params['host_ip'] = args.host_ip
    if args.port != None:
        params['port'] = args.port
    if args.video_length != None:
        params['video_length'] = args.video_length
    if args.output_folder_path != None:
        params['output_folder_path'] = args.output_folder_path
    if args.pipeline_path != None:
        params['pipeline_path'] = args.pipeline_path
    


if __name__ == "__main__":
    """
    main function which accepts the arguments and sets appropriate variables
    """

    prompts = load_prompts()
    params = load_params()
    parser = argparse.ArgumentParser()

    parser.add_argument("--openai_api_key")
    parser.add_argument("--host_ip")
    parser.add_argument("--port")
    parser.add_argument("--video_length")
    parser.add_argument("--output_folder_path")
    parser.add_argument("--pipeline_path")
    
    args = parser.parse_args()
    update_params(args)

    # First run whisper
    # Run the pipeline


    temporal_reasoner = TemporalReasoner()
    temporal_reasoner.subscribers(params, prompts)
    rospy.spin()




        