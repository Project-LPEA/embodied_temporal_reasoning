import cv2 as cv

def interval_extracter(time_stamp, video_path):

    
    # Open the video file
    cap = cv.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    fps = cap.get(cv.CAP_PROP_FPS)
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    
    # Calculate the frame number range for the specified timestamp
    frame_number = int((time_stamp - 1) * fps)
    start_frame = frame_number + 1
    end_frame = start_frame + 2*fps 
    
    # Adjust end_frame if it exceeds the total number of frames
    if end_frame > total_frames:
        end_frame = total_frames
    print("Total frames: ", total_frames)

    # Set the video to start from the desired frame
    cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)

    current_frame = start_frame
    frame_numbers = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or current_frame > end_frame:
            break
        
        # Save the frame
        # frame_filename = os.path.join(images_folder_path, f"{str(interval[-1]):05d}.jpg")
        # cv.imwrite(frame_filename, frame)
        # print(f'Saved {frame_filename}')

        if current_frame > 0:
        
            frame_numbers.append(f"{int(current_frame):05d}")
        
        current_frame += 1

    cap.release()

    return frame_numbers