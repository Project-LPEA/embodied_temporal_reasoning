from dds_cloudapi_sdk import Config

from dds_cloudapi_sdk import Client

from dds_cloudapi_sdk import DetectionTask

from dds_cloudapi_sdk import TextPrompt

from dds_cloudapi_sdk import DetectionModel

from dds_cloudapi_sdk import DetectionTarget

import os    

from PIL import Image, ImageDraw, ImageFont

import random

import cv2 as cv

import webcolors

from copy import deepcopy



def dino(object_of_interest, image_path, options_path, bbox_threshold):

    # Convert image to PNG and upload it

    input_path = image_path

    print(input_path)

    converted_path = os.path.splitext(input_path)[0] + '.png'

    # Convert JPEG to PNG

    with Image.open(input_path) as img:

        img.save(converted_path, format='PNG')



    # Initialize the config

    token = "345ed33943d1b9498e1e10b1fa2f1576"

    while True:

        try:

            config = Config(token)

            # Initialize the client

            client = Client(config)



            # Upload the PNG image to DDS server

            image_url = client.upload_file(converted_path)

            print(f"Uploaded image URL: {image_url}")  # Debugging print



            # Run the detection task

            task = DetectionTask(

                image_url=image_url,

                prompts=[TextPrompt(text=object_of_interest)],

                targets=[DetectionTarget.BBox],  # Detect bounding boxes

                model=DetectionModel.GDino1_5_Pro,  # Use GroundingDino-1.5-Pro model,

            )



            client.run_task(task)

            

            break

        except Exception as e:

            print(f"GDINO error: {e}")

            

            

    result = task.result

    all_bboxes = {}

    print("result objects: ", result)

    # Check if result is valid

    if result and result.objects:

        objects = result.objects  # List of detected objects

        print("Objects inside OG: ", objects)



        # Open the original image to draw bounding boxes

        

        color_image = cv.imread(converted_path) 

        

        # List of colors to use for bounding boxes

        colors = ["red", "green", "blue", "purple", "magenta", "orange", "cyan", "pink"]

        colors = [webcolors.name_to_rgb(color_name) for color_name in colors]

        

        index = 0 

        for idx, obj in enumerate(objects):

            print("idx, obj: ", idx, obj)

            print("Object category and obj score: ", obj.category, obj.score)

            print("Bbox threshold: ", bbox_threshold)

            print("Object of interest: ", object_of_interest)

            if str(obj.category).lower() == str(object_of_interest).lower() and obj.score >= bbox_threshold:

                bbox = obj.bbox

                bbox = list(map(int, bbox))

                print("bbox : ", bbox)

                # Choose a color for the bounding box

                

                color = (colors)[index % len(colors)]

                cv.rectangle(color_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 5)

                # Draw the label text

                label = f"Object {idx + 1}"

                all_bboxes[str(label)] = bbox

                font = cv.FONT_HERSHEY_SIMPLEX

                font_scale = 1

                font_thickness = 2

                text_size = cv.getTextSize(label, font, font_scale, font_thickness)[0]

                (text_width, text_height), baseline = cv.getTextSize(label, font, font_scale, font_thickness)


                # Calculate text position to ensure it doesn't get cut off

                text_x = max(bbox[0], 0)

                text_y = max(bbox[1] - 5, 0)

                if text_x + text_size[0] > color_image.shape[1]:

                    text_x = color_image.shape[1] - text_size[0] - 5

                if text_y - text_size[1] < 0:

                    text_y = bbox[3] + text_size[1] + 5



                rect_start = (text_x, text_y - text_height - baseline)  # Top-left corner

                rect_end = (text_x + text_width, text_y + baseline)



                # Draw the filled rectangle

                cv.rectangle(color_image, rect_start, rect_end, color, thickness=cv.FILLED)



                # Draw the text on the image

                cv.putText(color_image, label, (int(text_x), int(text_y)), font, font_scale, (255,255,255), font_thickness)



            index +=1



        # Save the image with bounding boxes

        output_path = options_path

        # color_image = color_image.convert('RGB')

        # color_image.save(output_path)

        cv.imwrite(output_path, color_image)

            

        print(f"Image saved with bounding boxes at {output_path}")

        all_bboxes = deepcopy(all_bboxes)

    else:

        print("No result found or failed to run the detection task.")

    return all_bboxes






def extract_last_frame(video_path):

        """ 

        Function to extract last frame from input video



        """

        input_list = []

        

        cap = cv.VideoCapture(video_path)

        while True:

            ret, frame = cap.read()

            if not ret:

                break

            last_frame = frame

        cap.release()



        if last_frame is not None:

            cv.imwrite("last_frame.jpg", last_frame)

        else:

            print("No frames found in the video.")



# if __name__ == "__main__":



    # extract_last_frame("/home/nivi_nath/tmp_reason_ws/src/driver_codes/CogVLM2/data/dataset_2/videos/91.mp4")



    # all_bboxes = dino("cup", "/media/nivi_nath/Extreme SSD/Pipeline_outputs_3/dataset2_outputs/78/frames/00074.jpg", "/media/nivi_nath/Extreme SSD/Pipeline_outputs_3/dataset2_outputs/78/frames/with_boxes.pmg", bbox_threshold=0.2)

    # print(str(all_bboxes))

