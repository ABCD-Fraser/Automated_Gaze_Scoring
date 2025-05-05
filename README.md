# Automated Gaze Orientation Estimation with the L2CS-Net model. 


This repository provides an example pipeline for processing and annotating videos using the L2CS gaze estimation model. The script extracts frames from input videos, applies gaze estimation, normalises scores, and annotates the frames with visual markers. The processed videos are saved for inspection, and frame-by-frame annotations are exported as a CSV file for further analysis.


## Repository Structure
```
.
├── input/
│   ├── video/          # Contains three example video files
│   ├── scoring_data.csv # Corresponding scoring file for the videos
├── output/
│   ├── L2CS_annotation/    # Folder for L2CS-annotated videos
│   ├── normalised_annotation/ # Folder for normalised annotated videos
│   ├── l2cs_data.csv # Output file with frame-wise analysis
├── process_videos.py  # Main script for processing videos
└── models/            # Folder for L2CS model weights

```

## Requirements
To run this project, ensure the following dependencies are installed:
- Python (tested with version 3.13.3)
- OpenCV: `cv2` (tested with version 4.11.0.86)
- PyTorch: `torch` (tested with version 2.7.0) 
- Pandas: `pandas` (tested with version 2.2.3)
- tqdm: `tqdm` (tested with version 4.67.1)

Additionally, you will need to install the L2CS-Net package for gaze estimation.

### Installation
You can install the required dependencies (with tested versions) and the [L2CS-Net package](https://github.com/Ahmednull/L2CS-Net) using the following commands:

```sh
pip install opencv-python==4.11.0.86 torch==2.7.0 pandas==2.2.3 tqdm==4.67.1

pip install git+https://github.com/edavalosanaya/L2CS-Net.git@main
```

## Example Usage

To demonstrate the functionality of the script a set of exmple videos and associated scoring have been provided. All example videos have been provided by researchers invovled in the study, and they have provided consent for their use as examples. 

1. Ensure example input videos are inside the `input/video/` directory.
2. Ensure `scoring_data.csv` is present in `input/` directory with example participant scoring information.
3. Run the main processing script:
   ```sh
   python run_example_L2CS_pipeline.py
   ```
4. The processed videos and corresponding annotations will be saved in the `output/` directory.

## Functionality
- **Frame Extraction & Processing:** Extracts frames from input videos.
- **Gaze Estimation:** Uses L2CS-Net to estimate gaze direction.
- **Annotation:** Annotates frames with gaze direction markers. Outputs the L2CS annotated videos.
- **Normalisation:** Normalises pitch values for visualisation. Outputs videos annotated with normalised L2CS gaze 
estimation and relative location of target. 
- **CSV Output:** Saves frame-wise gaze estimations for further analysis.

## Notes
- The model weights have been provided and are located inside the `models/` directory.
- The model weights have been provided by [Abdelrahman, Hempel, Khalifa, Al-Hamadi, 2023](https://ieeexplore.ieee.org/document/10372944) and can be found on the [L2CS-Net repo](https://github.com/Ahmednull/L2CS-Net).  
- The script defaults to using the CPU. If you have a CUDA enabled GPU, this can be activated by setting the `GPU_index` variable in the `run_example_l2cs_pipeiline.py` to the GPU device id. 

## License
This project is open-source under the MIT License.

