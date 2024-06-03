# Depth Estimation from Optical Flow in Agricultural Fields

## Project Overview

This project aims to develop a robust and cost-effective system for estimating depth in agricultural fields using optical flow techniques. The primary objective is to utilize a single RGB camera mounted on a vehicle to measure the depth of objects such as plant canopies, trunks, and fruits, facilitating various agronomic analyses.

## Key Features

- **Optical Flow Calculation**: Utilizes the Lucas-Kanade method for calculating optical flow, tracking the movement of pixels between consecutive frames to estimate motion.
- **Depth Estimation Model**: Based on the displacement of tracked points in the image plane, the depth (`Z`) is estimated using the relationship `ΔY / δy = Z / fy`, where `ΔY` is the displacement in the real-world coordinates and `δy` is the displacement in the image plane.
- **Camera Calibration**: Ensures accurate depth estimation by correcting for lens distortion using a chessboard calibration method.
- **Experimental Validation**: Employs Aruco markers to validate the system in controlled environments before deployment in the field.

## Statistical and Computer Vision Concepts

### Optical Flow

Optical flow refers to the pattern of apparent motion of objects in a visual scene caused by the relative motion between the observer (camera) and the scene. The Lucas-Kanade method is used to estimate the optical flow by solving a set of linear equations that relate pixel intensities between consecutive frames.

### Depth Estimation

Depth estimation in this project is achieved by analyzing the displacement of pixels in the image plane. The focal lengths `fx` and `fy` are assumed to be equal (for square pixels), allowing us to simplify the depth calculation. For non-square pixels, the focal lengths are considered separately.

### Camera Calibration

To correct lens distortions, a chessboard pattern is used for camera calibration. This process involves capturing multiple images of the pattern, calculating the intrinsic parameters (focal lengths, principal points), and the distortion coefficients to correct the images.

### Error Minimization and Model Fitting

The project employs non-linear least squares methods to fit the depth estimation model to the collected data. The model's performance is evaluated using the `R^2` coefficient, which quantifies the proportion of variance in the observed data explained by the model. Techniques such as moving average filters are used to enhance the robustness of the optical flow metric, balancing between model uncertainty and latency.

## Getting Started

### Prerequisites

- Python 3.x
- OpenCV
- NumPy
- SciPy
- Aruco Library

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/bernardolanza93/DepthFromOpticalFlow.git
    cd DepthFromOpticalFlow
    ```

2. Install the required libraries:
    ```sh
    pip install -r requirements.txt
    ```

### Usage

1. Calibrate the camera using the provided calibration script:
    ```sh
    python calibrate_camera.py
    ```

2. Run the depth estimation process:
    ```sh
    python depth_multi_marker_optical_flow.py
    ```
   


# Image Analyzer Project

## Function Descriptions

### `imaga_analizer_raw`
This function processes raw video files, detecting and analyzing Aruco markers to extract their positional data. The results are saved for subsequent processing.

### `convert_position_to_speed`
This function converts the positional data obtained from the raw videos into speed data (optical flow).

## Experimental Model Fitting

This section fits the experimental model to the data and generates graphs to visualize the results.


```
EXPERIMENTAL_MODEL_FITTING = 0
if EXPERIMENTAL_MODEL_FITTING:
file_path_1 = 'dati_of/all_points_big_fix_speed.xlsx'
show_result_ex_file(file_path_1)
windowing_vs_uncertanty(file_path_1)
```


### `show_result_ex_file`
This function fits the experimental model to the data in the provided Excel file and generates graphs based on the reference external velocities.

### `windowing_vs_uncertanty`
This function analyzes the impact of different window sizes on the uncertainty of the model.

## Experimental K's Results Evaluation

This section analyzes the experimental constant values obtained from the model fitting at different reference speeds.

```
EXP_Ks_RESULTS_EVALUATION = 0
if EXP_Ks_RESULTS_EVALUATION:
constant_analisis()
```

### `constant_analisis`
This function performs a detailed analysis of the experimental constant values derived from the model evaluation at various reference speeds.

## Raw Robot and Optical Flow Data Validation

This section involves the synchronization and validation of raw robot data with optical flow and depth information.
```
RAW_ROBOT_AND_RAW_OPTICAL_FLOW_VALIDATION = 1
if RAW_ROBOT_AND_RAW_OPTICAL_FLOW_VALIDATION:
x_s, vy_s = synchro_data_v_v_e_z("results_raw.xlsx")
merge_dataset_extr_int(x_s, vy_s)
```

### `synchro_data_v_v_e_z`
This function synchronizes the external velocities of the robot to ensure a reliable robot path.

### `merge_dataset_extr_int`
This function merges the robot's raw data with optical flow and depth information based on timestamps, ensuring synchronized validation.


# Additional Functions

### `find_signal_boundaries`
Identifies significant changes in a signal based on a threshold.

### `calculate_theo_model_and_analyze`
Fits a theoretical model to the data and calculates various metrics to evaluate the model's performance.

## Function Descriptions

### `imaga_analizer_raw()`
Processes raw video files, detecting and analyzing Aruco markers to extract their positional data. The results are saved for subsequent processing.

### `convert_position_to_speed()`
Converts the positional data obtained from the raw videos into speed data (optical flow).

### `show_result_ex_file(file_path)`
Fits the experimental model to the data in the provided Excel file and generates graphs based on the reference external velocities.

### `windowing_vs_uncertanty(file_path)`
Analyzes the impact of different window sizes on the uncertainty of the model.

### `constant_analisis()`
Performs a detailed analysis of the experimental constant values derived from the model evaluation at various reference speeds.

### `synchro_data_v_v_e_z(file_raw_optics)`
Synchronizes the external velocities of the robot to ensure a reliable robot path.

### `merge_dataset_extr_int(x, y)`
Merges the robot's raw data with optical flow and depth information based on timestamps, ensuring synchronized validation.

### `find_signal_boundaries(signal, threshold)`
Identifies significant changes in a signal based on a threshold.

### `calculate_theo_model_and_analyze(n_df, n_vy, win, vlim)`
Fits a theoretical model to the data and calculates various metrics to evaluate the model's performance.



## Contributing

We welcome contributions to improve the project. Please fork the repository and submit pull requests.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

- This project was developed by [bernardolanza93],  and supported by the Department of Industrial and Mechanical Engineering, University of Brescia.
- Special thanks to the contributors of the OpenCV and Aruco libraries for their invaluable tools and resources.

