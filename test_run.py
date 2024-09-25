#!/usr/bin/env python3

"""

Run file to test the temporal reasoning pipeline on a fixed dataset. 
The parameters of the pipeline can be set by passing arguments to this file.

"""

# Library imports
import yaml
import os, shutil
import pathlib
import os   
import json
import json
import pathlib
from .src.eval import tr_test,iou_calculation, fact1
import time




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
            params['event_localizer'] = args.event_localizer
    

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

    # dino_input_folder = "/home/nivi_nath/tmp_reason_ws/src/driver_codes/CogVLM2/Dino_Input/"

    prompts = load_prompts()
    params = load_params()

    dataset_folder = params["dataset_folder_path"]
    # output_folder = str(current_dir_path) + "/output/dataset1_outputs/final/"

    video_folder = dataset_folder + "/videos/"

    with open(dataset_folder + "/info.json") as f:
        data = json.load(f)

    with open(dataset_folder + '/ground_truth.json') as f:
        bbox_groundtruths = json.load(f)
    
    with open(dataset_folder + '/ground_truth/validator.json') as f:
        validator_data = json.load(f)

    correct_output = 0
    for i in range(1,156):    

        output_folder = str(current_dir_path) + "/output/" + str(i)

        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)

        images_folder_path = output_folder + "/frames/"

        if os.path.exists(images_folder_path):
            shutil.rmtree(images_folder_path)

        os.makedirs(images_folder_path)

        pipeline_output = output_folder + '/' + str(i) + '.txt'
        image_output = output_folder + '/' + str(i) + '.png'

        # getting the corresponding video and instruction
        datum = data[str(i)]
        video_path = video_folder + datum['video_path']
        instruction = datum['instruction']
        time_stamp = validator_data[str(i)]['Expected time instant']

        print("Time instance for datum " + str(i) + " : ", time_stamp)

        bbox_gt = bbox_groundtruths[str(i)]

        if params["factorization"] == 1:

            print("Running factorization 1!")
            output_folder = str(current_dir_path) + "/output_fact1/dataset" + str(dataset_number) + "_outputs_fact1/" + str(i)
            images_folder_path = output_folder + "/frames/"
            if os.path.exists(images_folder_path):
                 shutil.rmtree(images_folder_path)
            os.makedirs(images_folder_path)
            pipeline_output = output_folder + '/' + str(i) + '.txt'
            image_output = output_folder + '/' + str(i) + '.png'
            bbox_coords = fact1.test_pipeline(time_stamp, video_path, instruction, image_output, images_folder_path, params, prompts, pipeline_output, bbox_gt)

        else:
        #running the pipeline
            pipeline_start_time = time.time()
            bbox_coords = tr_test.test_pipeline(video_path, instruction, image_output, images_folder_path, params, prompts, pipeline_output, bbox_gt)
            print("bbox_cords final : ", bbox_coords)
            pipeline_end_time = time.time()
            total_time = pipeline_end_time-pipeline_start_time

        #Evaluation of pipeline
        iou = iou_calculation.bb_intersection_over_union(bbox_coords, bbox_gt)
        if iou >= 0.7:  correct_output = correct_output + 1
        else: incorrect_cases += [int(i)]
        # total_time_for_all_cases += total_time
        with open(pipeline_output, 'a') as file: 
            file.write("IOU: " + str(iou) + "\n" \
                        "Number of correct cases: " + str(correct_output) + "\n" \
                        # "Total Time: " + str(total_time) + "\n" \
                        "Accuracy: " + str((correct_output/i) * 100) + "%"+ "\n" \
                        # "Avg Time: " + str(total_time_for_all_cases/i) + "\n" \
                        "Incorrect Cases: " + str(incorrect_cases))  


         


