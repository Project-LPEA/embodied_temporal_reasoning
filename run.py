#!/usr/bin/env python3

"""Run file to execute the temporal reasoning pipeline. The parameters
of the pipeline can be set by passing arguments to this file.
"""

# Library imports
import argparse
import yaml
import os
import pathlib
from src import temporal_reasoning
# from . import ros_whisper
import rospy
import multiprocessing

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
    if args.width != None:
        params['width'] = args.width
    if args.height != None:
        params['height'] = args.height



if __name__ == "__main__":
    """main function which accepts the arguments and sets appropriate variables
    """
    print("Running Main")
    prompts = load_prompts()
    params = load_params()
    parser = argparse.ArgumentParser()

    parser.add_argument("--openai_api_key")
    parser.add_argument("--host_ip")
    parser.add_argument("--port")
    parser.add_argument("--width")
    parser.add_argument("--height", \
                        help="height of frame")
    
    args = parser.parse_args()
    update_params(args)

    # First run whisper
    # Run the pipeline

    print("ros node TR done")
    temporal_reasoner = temporal_reasoning.TemporalReasoner()
    print("Temporal Reasoning done")
    temporal_reasoner.subscribers(params, prompts)
    print("Sub done")
    rospy.spin()




        