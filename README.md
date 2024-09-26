# embodied_temporal_reasoning
Repository for project on temporal reasoning for intelligent human robot collaboration.

This project assists robots to perform temporal reasoning over the past, and carry out human-instructions in the present, in a generalized manner via Foundational Models.

# Requirements
1. # Whisper Model
   * The Whisper model needs to be downloaded locally on a server from <https://github.com/openai/whisper>, and the server file needs to be set up.
   * In the client file, `ros_whisper.py`, update the following variable: `self.host` and `self.port` to server's IP address and corresponding port number used by the server.

2. # CogVLM2 Model
   * CogVLM2 
