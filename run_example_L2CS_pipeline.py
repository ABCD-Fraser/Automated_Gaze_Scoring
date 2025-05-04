
"""
Script Overview:
This script processes and annotates videos using the L2CS gaze estimation model.
It extracts frames from videos, applies gaze estimation, normalises scores, and annotates the frames with visual markers. 
Annotated videos are saved for inspection.
Frame by frame annotations are saved to a CSV file for further analysis.
"""

# Import required libraries
import os 
import time  
import cv2  
import pathlib  
import torch  
import csv  
import pandas as pd  
from l2cs import Pipeline, render  
from tqdm import tqdm  


def write_video(frames, fpath):
    
    """
    Writes a sequence of frames to a video file.

    Args:
        frames (list of numpy.ndarray): A list of frames (images) to be written to the video. 
                                        Each frame should be a numpy array with shape (height, width, channels).
        fpath (str): The file path where the output video will be saved.

    Returns:
        None
    """
    print(f'Writing L2CS annotated video to: {fpath}')
    height, width, layers = frames[0].shape  # Get the dimensions of the frames
    output_video = cv2.VideoWriter(
        fpath, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height)  # Save as MP4 with 30 FPS
    )
    for img in frames:
        output_video.write(img)  # Write each frame to the output video
    output_video.release()  # Release the video writer


def process_l2cs_video(input_video_path, output_video_path, participant_rows, torch_device='cpu'):
    """
    Process a video to estimate gaze using the L2CS pipeline and annotate the frames.
    Args:
        input_video_path (str): Path to the input video file.
        output_video_path (str): Path to save the output annotated video file.
        participant_rows (pd.DataFrame): DataFrame containing participant data to be combined with gaze estimation results.
    Returns:
        frame_results_list (list): List of results from gaze estimation for each frame.
        output_video_path (str): Path to the saved annotated video file.
        participant_rows (pd.DataFrame): Updated DataFrame with L2CS scores and classifications.
    """

    # Initialise the L2CS pipeline
    gaze_pipeline = Pipeline(
        weights=pathlib.Path('models/Gaze360/L2CSNet_gaze360.pkl'),  # Path to model weights
        arch='ResNet50',  # Model architecture
        device=torch.device(torch_device)  # Device to run the model (CPU in this case)
    )



        # Load video and initialise variables for frame processing
    video = cv2.VideoCapture(input_video_path)  # Open video file
    frames = []  # Store processed frames
    frame_results_list = []  # Store results from gaze estimation

    # set up progress bar and inisialise frame index and list for pitch values
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total number of frames
    progress_bar = tqdm(total=total_frames, desc=f"Annotating {input_video_path}", unit="frame")  # Initialise progress bar
    
    

    # Process each frame in the video
    while video.isOpened():
        ret, frame = video.read()  # Read the next frame
        
        if not ret:  # Break if no frame is returned
            break

        try:
            img = frame.copy()  # Create a copy of the frame for processing

            # Perform gaze estimation if the frame is valid
            if frame is not None:
                frame_results = gaze_pipeline.step(frame)  # Process the frame through the pipeline
                frame_results_list.append(frame_results)  # Append results to the list
            else:
                continue
            
            # Annotate the frame with the L2CS score
            img = render(img, frame_results)  # Annotate the frame with results
            frames.append(img)  # Add the annotated frame to the list

            # Update the progress bar
            progress_bar.update(1)  

        except Exception as e:  # Handle exceptions during frame processing
            print(f"Error processing frame: {e}")

    # Release resources and close the progress bar
    video.release()  
    progress_bar.close()  

    # Combine norm data with pitch data
    pitch_values = [f.pitch[0] for f in frame_results_list]  # Extract pitch values
    pitch_class = ['left' if x < 0 else 'right' for x in pitch_values]  # Classify pitch as left/right

    temp_df = pd.DataFrame({'l2cs_score': pitch_values, 'l2cs_class': pitch_class})
    participant_rows = participant_rows.reset_index(drop=True)
    participant_rows = pd.concat([participant_rows, temp_df], axis=1)

    # Save the annotated frames as a video
    write_video(frames, output_video_path)

    return frame_results_list, output_video_path, participant_rows


def normalised_video_annotation(input_video_path, output_video_path, participant_rows):
    
    """
    Annotates a video with normalised pitch values and true locations.
    Parameters:
        input_video_path (str): Path to the input video file.
        frame_results_list (list): List of frame results for each frame in the video.
        output_video_path (str): Path to save the annotated output video.
        participant_rows (pd.DataFrame): DataFrame containing participant data, including 'l2cs_score' and 'true_location'.
    Returns:
        pd.DataFrame: Updated DataFrame with normalised pitch values.
        str: Path to the annotated output video.
    """
    
    # Reprocess for adding detailed annotations
    video = cv2.VideoCapture(input_video_path)  # Reopen the processed video
    frame_index = 0  # Reset frame index
    frames = []  # Reset frames list

    pitch_values = participant_rows['l2cs_score']  # Extract pitch values
    positive_values = [value for value in pitch_values if value > 0]  # Filter positive pitch values
    negative_values = [value for value in pitch_values if value < 0]  # Filter negative pitch values

    # normalise positive and negative pitch values
    if positive_values:
        max_positive = max(positive_values)

    if negative_values:
        min_negative = min(negative_values)

    pitch_norm = [
        (value / max_positive if value > 0 else value / abs(min_negative))
        for value in pitch_values
    ]


    # Combine norm data with pitch data
    temp_df = pd.DataFrame({'l2cs_norm': pitch_norm})
    participant_rows = participant_rows.reset_index(drop=True)
    participant_rows = pd.concat([participant_rows, temp_df], axis=1)

    # Set up progress bar for annotation
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = tqdm(total=total_frames, desc=f"Adding normalised annotations to {input_video_path}", unit="frame") 

    # Process each frame in the video
    while video.isOpened():
        ret, frame = video.read()  # Read the next frame
        if not ret:  # Break if no frame is returned
            break
        try:
            
            # Extract normalised pitch and true locatiomn for the current frame
            norm_pitch = participant_rows['l2cs_norm'][frame_index]  
            true_location = participant_rows['true_location'][frame_index] 

            # Copy frame for annotation
            img = frame.copy()  

            # Draw annotations
            l2cs_x = int(img.shape[1] * (1 - norm_pitch) / 2)  # X coordinate for normalised pitch
            l2cs_y = int(img.shape[0] / 2)  # Y coordinate for normalised pitch
            cv2.circle(img, (l2cs_x, l2cs_y), radius=5, color=l2cs_colour, thickness=-1)  # Red dot

            true_x = int(img.shape[1] * (1 - true_location) / 2)  # X coordinate for true location
            true_y = int(img.shape[0] / 2)  # Y coordinate for true location
            cv2.circle(img, (true_x, true_y), radius=5, color=true_location_colour, thickness=-1)  # Blue dot

            # Append annotated frame to the list
            frames.append(img)  

            # Update progress bar
            progress_bar.update(1)

        # Handle exceptions during frame processing
        except Exception as e: 
            print(f"Error processing frame: {e}")

        # Increment frame index
        frame_index += 1 

    # Release resources and close the progress bar
    video.release() 
    progress_bar.close() 

    # Save the fully annotated video
    write_video(frames, output_video_path)

    return participant_rows, output_video_path

# Set paths for input and output directories
input_dir = os.path.join("input", "video")  # Directory containing input videos
output_dir = os.path.join("output") # Directory for saving output files
L2CS_dir = os.path.join("output", "L2CS_annotation") # Directory for L2CS annotated output
norm_dir = os.path.join("output", "normalised_annotation")  # Directory for normalised annotated output
dir_list = [output_dir, L2CS_dir, norm_dir]  # List of directories to create

for dir in dir_list: # Create directories if they do not exist
    os.makedirs(dir, exist_ok=True)  

# Initialise the output DataFrame and set path for output CSV
output_df = pd.DataFrame()  
output_df_path = os.path.join(output_dir, 'l2cs_data.csv')  

# Record the start time of the script
start_time = time.time()  

# Parameters for annotation
annotate_norm = True  # Flag to control whether normalisation is annotated
l2cs_colour = (0, 0, 255)  # Red color for L2CS annotations
true_location_colour = (255, 0, 0)  # Blue color for true location annotations

# Load the data for human scoring and target location from CVSV file
score_data_path = os.path.join("input", "scoring_data.csv") 
score_data = pd.read_csv(score_data_path)  

# List files in video directory
video_files = os.listdir(input_dir) 

# Set the GPU index for CUDA enabled GPU for PyTorch. If not GPU is availble set the value to 'cpu'
GPU_index = 'cpu' 

# loop through each video file in the input directory
video_file_count = 0
for video_file in video_files:
    
    try:
        # Increment video count and extract participant ID 
        video_file_count += 1
        video_name = os.path.splitext(video_file)[0]  # Get the video name without extension
        participant_id = video_name[:4]  # Extract participant ID (first 4 characters)


        # Set individual paths for this video file
        input_video_path = os.path.join(input_dir, video_file)  # Path for input video
        l2cs_annotated_path = os.path.join(L2CS_dir, f"{video_name}_L2CS.mp4") # Path for L2CS annotated video
        norm_annotated_path = os.path.join(norm_dir, f"{video_name}_norm.mp4")  # Path for normalised annotated video

        # Filter score_data for the current participant
        participant_rows = score_data[score_data['participant'] == participant_id]

        
        print(f"Processing video {video_file_count}/{len(video_files)}: {video_file}")

        # Process the video to annotate with L2CS gaze estimation
        frame_results_list, l2cs_annotated_path, participant_rows = process_l2cs_video(input_video_path, 
                                                                                       l2cs_annotated_path, participant_rows, torch_device=GPU_index)

        
        # Annotate the video with normalised pitch values and true location
        if annotate_norm:
            
            # Annotate the video with normalised pitch values and true location
            participant_rows, norm_annotated_path = normalised_video_annotation(l2cs_annotated_path,
                                                                                norm_annotated_path, participant_rows)
        
        
        # Append the participant data to the output DataFrame
        output_df = pd.concat([output_df, participant_rows], axis=0)

    # Handle exceptions during video processing
    except Exception as e: 
        print(f"Error processing {input_video_path}: {e}")

# Save the output DataFrame to a CSV file
output_df.to_csv(output_df_path, index=False)

# Print total execution time
print(f"Finished processing all videos. Total time: {time.time() - start_time:.2f} seconds")
