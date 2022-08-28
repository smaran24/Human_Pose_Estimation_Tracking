## Human_Pose_Estimation_Tracking
Human Pose Estimation and Tracking with unique ID

This project has been developed in python using human pose estimation and tracking models and can be executed in openVINO environment.

#### Overview
This project is the combination of two different objectives. It works combinely the following:
###### - Human pose estimation: The project uses open pose estimation technique out several pose estimation ethods. This pose contains 18 key points of human i.e., ears, eyes, nose, neck, shoulders, elbows, wrists, hips, knees, and ankles to detect a multiperson 2D human pose. human-pose-estimation-0001 is implemented which can be downloaded from OpenVINO Open Model Zoo.
###### - Tracks the human with unique ID: It tracks multiple human by using centroid tracking algorithm. It identifies and localize the objects of interest i.e., human by drawing bounding boxes around them in each frame generating a unique ID for each detected human, tracking them as they move around in a video while maintaining the ID assignment.


#### Prerequisites to run program
- OpenVINO
- OpenCV
- Microsoft Visual Studio 2019

#### How to Run:
- Activate openVINO environment executing "openvino_env\Scripts\activate" in command prompt.
- Navigate to the directory of the files:  cd <directory>
- Run command: python pose_estimation.py

The Program videos in the Test_Videos has been considered to test the model. Output is saved in the Output_Videos.
