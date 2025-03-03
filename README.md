# Bottle_AnoDet
A real-time anomaly detection project from an RTSP camera feed, featuring a pre-trained model and image classification using EfficientNetV2-S.
<img src="https://skillicons.dev/icons?i=python" />

## Table of contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Usage](#usage)
4. [License](#license)
5. [Contact](#contact)

## Overview
The Bottle_AnoDet project is designed for real-time anomaly detection from an RTSP camera feed. It loads a pre-trained model, captures and processes frames, applies region-of-interest cropping and masking, and performs real-time inference. Additionally, the project includes the training files of the image classification model using EfficientNetV2-S for the task and testing the model on a dataset, evaluating its performance based on class-wise accuracy and misclassified images.

## Installation
To install the project, follow these steps:
```bash
# Install Python and necessary libraries
pip install -r "requirements.txt"
```
Make sure to install the required libraries from the requirements file, such as PyTorch and Torchvision, to run the project.

## Usage
The script continuously processes frames and displays the prediction. The user can exit the live video feed by pressing 'q'. This project also includes a demo video creator that showcases the live classification. For model testing, it evaluates the model's performance on the test dataset, displaying test loss, overall accuracy, class-wise accuracy and misclassified images for further analysis.  

## License
This project is under no license.

## Contact
For any questions or feedback, please reach out to:
- **Email**: [imv.university@gmail.com](mailto:imv.university@gmail.com)
- **GitHub Profile**: [imendezval](https://github.com/imendezval), [asumak2003](https://github.com/asumak2003)
- **LinkedIn**: [inigo-miguel-mendez-valero](https://www.linkedin.com/in/i%C3%B1igo-miguel-m%C3%A9ndez-valero-4ba3732b1/)
Feel free to open an issue on GitHub or contact us in any way if you have any queries or suggestions.