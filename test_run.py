#!/usr/bin/env python3

"""

Run file to test the temporal reasoning pipeline on a fixed dataset. 
The parameters of the pipeline can be set by passing arguments to this file.

"""

# Library imports
import argparse
import yaml
import os, shutil
import pathlib
import rospy
import multiprocessing
import os   
import sys
sys.path.append('/home/nivi_nath/tmp_reason_ws/src/driver_codes/fm_temporal_reasoning/')
import cv2
import json
import io
from cv_bridge import CvBridge  
import json
import pathlib
from src.eval import pipline_test
from src.eval import iou_calculation




current_dir_path = pathlib.Path.cwd()


def load_params():
    """Loads the parameter

    Returns:
        dict: parameters of the localization pipeline
    """
    params = {}
    params_path =  str(current_dir_path) + "/config/params.yaml" 
    with open(params_path) as stream:
        try:
            params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return params


def update_params(args):
    """Function to update the parameters

    Args:
        args (argparse): arguments from CLI
    """
    if args.component != None:
            params['component'] = args.component
    

def load_prompts():
    """Loads the prompts

    Returns:
        dict: parameters of the localization pipeline
    """
    prompts = {}
    prompts_path = str(current_dir_path) + "/config/prompts.yaml"
    with open(prompts_path) as stream:
        try:
            prompts = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return prompts


    
if __name__ == '__main__':

    """
    
    main function which accepts the arguments and sets appropriate variables

    """

    prompts = load_prompts()
    params = load_params()

    print(prompts)

    video_folder = str(current_dir_path) + "/data/dataset_1/videos/"
    output_folder = str(current_dir_path) + "/output/dataset_1/final/"

    with open(str(current_dir_path) + '/data/dataset_1/info.json') as f:
        data = json.load(f)

    with open(str(current_dir_path) + '/data/dataset_1/ground_truth/final.json') as f:
        bbox_groundtruths = json.load(f)

    correct_output = 41

    for i in range(62, len(data) + 1):
        
        folder_path = str(current_dir_path) + "/output/dataset1_outputs/" + str(i)

        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)

        output_folder = os.makedirs(folder_path)

        pipeline_output = folder_path + '/' + str(i) + '.txt'
        image_output = folder_path + '/' + str(i) + '.png'
       

        datum = data[str(i)]

        video_path = video_folder + datum['video_path']


        instruction = datum['instruction']
        bbox_gt = bbox_groundtruths[str(i)]

        bbox_coords = pipline_test.test_pipeline(video_path, instruction, image_output, params, prompts, pipeline_output, bbox_gt)

        
        iou = iou_calculation.bb_intersection_over_union(bbox_coords, bbox_gt)



        if iou >= 0.7:  correct_output = correct_output + 1
        print("IOU:", iou)

        with open(pipeline_output, 'a') as file: 

            file.write("IOU: " + str(iou) + "\n" \
                       
                        "Number of correct cases: " + str(correct_output) + "\n" \
                        
                        "Accuracy: " + str((correct_output/i) * 100) + "%") 


    








        
