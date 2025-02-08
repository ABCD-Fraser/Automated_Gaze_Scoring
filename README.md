# Automated Gaze Orientation Estimation with the L2CS-Net model. 

Overview

This repository provides an example pipeline for processing and annotating videos using the L2CS gaze estimation model. The script extracts frames from input videos, applies gaze estimation, normalizes scores, and annotates the frames with visual markers. The processed videos are saved for inspection, and frame-by-frame annotations are exported as a CSV file for further analysis.

Repository Structure:

.
├── input
│   ├── video/          # Contains three example video files
│   ├── scoring_data.csv # Corresponding scoring file for the videos
├── output
│   ├── L2CS_annotation/    # Folder for L2CS-annotated videos
│   ├── normalised_annotation/ # Folder videos annotated with normalised gaeze estimateion and relative target location
│   ├── l2cs_data.csv # Output file with frame-wise analysis
├──  models/            # Folder for L2CS model weights
├── run_example_L2CS_pipeline.py  # Main script for processing videos




## Requirements

Ensure you have the following dependencies installed:

OpenCV (cv2)
Torch (torch)
Pandas (pandas)
tqdm (tqdm)


## Usage
Example videos are stored in the input/video/ folder.

Ensure scoring_data.csv is present in input/ with participant scoring information.

Run the main processing script: python process_videos.py

The processed videos and corresponding annotations will be saved in the output/ directory.

## Functionality
Frame Extraction & Processing: Extracts frames from input videos.

Gaze Estimation: Uses L2CS to estimate gaze direction.

Annotation: Annotates frames with gaze direction markers. Outputs the L2CS annotated videos.

Normalisation: Normalises pitch values for visualisation. Outputs videos annotated with normalised L2CS gaze 
estimation and relative location of target. 

CSV Output: Saves frame-wise gaze estimations for further analysis.


### Notes

The model weights should be placed inside the models/ directory.

The script defaults to using a CPU; modify it to use a GPU if available.



License

This project is open-source under the MIT License.
