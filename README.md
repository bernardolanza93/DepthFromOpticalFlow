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
    git clone https://github.com/yourusername/depth-estimation-agriculture.git
    cd depth-estimation-agriculture
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
    python depth_estimation.py
    ```

## Contributing

We welcome contributions to improve the project. Please fork the repository and submit pull requests.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

- This project was developed by [Your Name], [Collaborators], and supported by the Department of Industrial and Mechanical Engineering, University of Brescia.
- Special thanks to the contributors of the OpenCV and Aruco libraries for their invaluable tools and resources.

