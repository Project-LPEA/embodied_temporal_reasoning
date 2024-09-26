# embodied_temporal_reasoning
Repository for project on temporal reasoning for intelligent human robot collaboration.

This project assists robots to perform temporal reasoning over the past, and carry out human-instructions in the present, in a generalized manner via Foundational Models.

## Requirements
1. ### Whisper Model
   * Download Whisper model from <https://github.com/openai/whisper>, and set up server file.
   * In the client file, `ros_whisper.py`, set `self.host` and `self.port` to server's IP address and corresponding port number used by the server.

2. ### CogVLM2 Model
   *  Download CogVLM2 from <https://github.com/THUDM/CogVLM2>, and set up server file.
   *  In `config/params.yaml`, set `cogvlm2_host_ip` and `cogvlm2_port` to server's IP address and corresponding port number used by the server.

3. ### SAM2 Model
   *  Download CogVLM2 from <https://github.com/IDEA-Research/Grounded-SAM-2>, and set up server file.
   *  In `config/params.yaml`, set `sam2_host_ip` and `sam2_port` to server's IP address and corresponding port number used by the server.

4. ### Dataset
   * Download the Dataset from <https://drive.google.com/drive/folders/1c78MIOhFKuIKvPrMw79iLxZvnk47zg0X?usp=sharing>.
   * Set the `dataset_folder_path` in `config/params.yaml` and in `baseline_params.yaml`.

5. ### CogVLM - grounding model for Baselines
   *  Download CogVLM from <https://github.com/THUDM/CogVLM>, and set up server file.
   *  In `config/baseline_params.yaml`, set `cogvlm_host_ip` and `cogvlm_port` to server's IP address and corresponding port number used by the server.

6. ### Recording Real-time Video
   * Set up realsense camera to record input video
   * Set `video_length` in `config/params.yaml` to the maximum length of video that you want to record
   * Video is stored in `output/run_output/input_video.mp4
  
## Running the code
1. ### Testing Pipeline
   * Set `openai_api_key` and `pipeline_path` variable in the `config/params.yaml`
   * Run `test_run.py` for running pipeline on dataset. See output for each datapoint in `output/dataset/` folder.
   * To test with real-time data, record a video using `realsense`, convert input instruction using `ros_whisper.py`. 
