# Bottle_AnoDet
A real-time anomaly detection project from an RTSP camera feed, featuring a pre-trained model and image classification using EfficientNetV2-S.

<img src="https://skillicons.dev/icons?i=python" /> <img src="https://skillicons.dev/icons?i=pytorch" />

## Table of contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Structure](#structure)
4. [Usage](#usage)
5. [Models](#models)
6. [Contact](#contact)

## Overview
The Bottle_AnoDet project is designed for real-time anomaly detection of live footage of a production chain located at the Hochschule Heilbronn laboratory, using an RTSP camera feed: a pre-trained model is loaded, frames are captured and preprocessed through region-of-interest cropping and masking, and real-time inference is then performed. A labeled image of the production chain can be seen below:


<div align="center">
  <img src="https://github.com/asumak2003/Bottle_AnoDet/raw/main/imgs/exs/production_chain.png">
</div>

Additionally, this repository also includes the whole used for training dataset, and all the files used to preprocess and prepare the recollected images for input to the network, as well as the training files of the image classification model, using EfficientNetV2-S pretrained on ImageNet and a class-weighted loss, and the testing files, evaluating its performance based on class-wise accuracy and displaying misclassified images.

The objective is to detect anomalies in the production chain, and classify them into one of 4 classes: no anomaly, no lid, fallen before and fallen after (the corner).

<div align="center">
  <img src="https://github.com/asumak2003/Bottle_AnoDet/raw/main/imgs/exs/classes.png">
</div>

## Installation
To install the project, follow these steps:
```bash
# Clone repository
git clone https://github.com/asumak2003/Bottle_AnoDet
# Install Python and necessary libraries
pip install -r "requirements.txt"
```
Make sure to install the necessary libraries from the requirements file, such as PyTorch and Torchvision, to run the project.

Apart from the above Python libraries, it is also necessary to install the FFmpeg framework to be able to run the [live_classification.py](./live_classification.py) file on the live footage of the RTSP camera. To do this:

+ Go to https://www.gyan.dev/ffmpeg/builds/
+ Download "ffmpeg-git-full.7z" or "ffmpeg-release-full.7z"
+ Extract it and place it in C:\ffmpeg
+ Add C:\ffmpeg\bin to your system's PATH

## Structure
Below, the structure of the repository is represented as a tree diagram. It is worth noting that not all files are displayed, only the most essential ones that are actually required for the mantainance and further development of the project.

```bash
Bottle_AnoDet
├───EffNet_fine_tune.py
├───EffNet_test.py
├───scratch_model.py
├───live_classification.py
├───models
├───imgs
│   ├───bin_mask_opt.jpg
│   ├───data_loader
│   │   ├───fallen_after
│   │   ├───fallen_before
│   │   ├───no_anomaly
│   │   └───no_deckel
│   └───empty_rail
│       └───augumented
├───img_prep_utils
│   ├───img_agumentation.py
│   ├───img_prep.py
│   └───duplicate_removal
└───wrong_screwed_lid
    └───data
```

## Usage

#### Main Files
In the main folder, the files used for the training and testing and usage of the EfficientNet model can be found.
+ The [scratch_model.py](./scratch_model.py) file simply showcases an attempt to train a model from scratch on our data. 
+ The file [EffNet_fine_tune.py](./EffNet_fine_tune.py) is the main responsible for fine tuning the EfficientNet model, pretrained on ImageNet, on our dataset. It outputs a model and multiple graphs displaying the progress of the model during training. 
+ The output model can then be tested using [EffNet_test.py](./EffNet_test.py), displaying test loss, overall accuracy, class-wise accuracy and misclassified images for further analysis.
+ The file [live_classification.py](./live_classification.py) is used for direct inference on the live footage of the RTSP camera, by using one of the several models, found under the "models" folder. The script continuously processes frames and displays the prediction. An aditional algorithm runs on top of the model, only outputting an error signal upon detection of an anomaly in 5 consecutive frames, reducing false positives greatly. The user can exit the live video feed by pressing 'q'. A screenshot of the final prototype can be seen below:

<div align="center">
  <img src="https://github.com/asumak2003/Bottle_AnoDet/raw/main/imgs/exs/prototype.jpg">
</div>

#### Dataset and Image Preprocessing
Under the [imgs](./imgs/) folder, one may find the binary mask, augumented images, and the dataset (organised according to the requirements of the the PyTorch DataLoader function).

All files responsible for the preprocessing of images, mainly used to prepare images to be added to the dataset, are found under the [img_prep_utils](./img_prep_utils/) folder:

+ The [img_agumentation.py](./img_prep_utils/img_agumentation.py) file is used to create augumented images using brightness and contrast changes. 
+ Furthermore, the [duplicate_removal](./img_prep_utils/duplicate_removal/) folder has multiple files showcasing the different techniques explored for removing duplicated frames, where there has been no changes in the production chain from one image to the other.
+ The [img_prep.py](./img_prep_utils/img_prep.py) file has multiple useful functions, including the sampling of videos, cropping and masking of images and duplicate removal. Below, the workflow of the cropping and masking can be observed:

<div align="center">
  <img src="https://github.com/asumak2003/Bottle_AnoDet/raw/main/imgs/exs/crop_and_mask.png">
</div>


#### Wrongly Screwed Lid
Finally, in the [wrong_screwed_lid](./wrong_screwed_lid/) folder, files attempting to detect when a lid has been misplaced can be found. Many different techniques were explored in an attempt to to detect this instance. The first step was always to detect the circumference of both the inner and outer ellipses of the lid of the bottle, and then multiple features were analysed, including but not limited to: the total area of the ellipse in the image, the eccentricity of the ellipse, the angle between the major and minor axis of the ellipse...

<div align="center">
  <img src="https://github.com/asumak2003/Bottle_AnoDet/raw/main/imgs/exs/wrong_screwed_lid.png">
</div>

Using a lot of data, it was concluded that this task was not possible with our current equipment. No feature could be found that separated the anomalies from the normal class. The analysis performed on the collected data can be found under the [data](./wrong_screwed_lid/data/) folder.

## Models
Over the course of the project, 8 different models were trained, all of which can be found under the [models](./models/) folder. It was a iterative process of tiral-and-improvement, consisting of training a model, identifying it's issues, and rectifying and retraining accordingly. For example, the imbalanced dataset resulted in much higher accuracy in classes with more images, and therefore a class-weighted loss was implemented.

In a later stage of the development process, it was discovered that, although the camera could save images in a 1280x720 resolution, the live footage was limited to 640x368. At this point, the whole dataset had already been collected with a 1280x720 resolution, so another model had to be trained adjusting the resolution at the input.

Below is a table summarising some of the models and their performance:

| Model Nr.     | Unfrozen Layers| Resolution|Epochs| Class-weighted Loss| Accuracy|
|:-------------:|:-------------:|:----------:|:----:|:------------------:|:-------:|
|4              | Classification| 1280x720|50| No| 90.8%|
| 5             | Classification and Blocks  6 + 7| 1280x720|15| No| 98.8%|
|6              | Classification and Blocks  6 + 7| 1280x720|15| Yes| 99.7%|
| 8             | Classification and Blocks  6 + 7| 640x368|15| Yes| 99.4%|

## Contact
For any questions or feedback, please reach out to:
- **Email**: [imendezval@stud.hs-heilbronn.de](mailto:imendezval@stud.hs-heilbronn.de), [asumak@stud.hs-heilbronn.de](mailto:asumak@stud.hs-heilbronn.de)
- **GitHub Profile**: [imendezval](https://github.com/imendezval), [asumak2003](https://github.com/asumak2003)
- **LinkedIn**: [inigo-miguel-mendez-valero](https://www.linkedin.com/in/i%C3%B1igo-miguel-m%C3%A9ndez-valero-4ba3732b1/), [arian-sumak](https://www.linkedin.com/in/arian-sumak-6b5b8925a/)

Feel free to open an issue on GitHub or contact us in any way if you have any queries or suggestions.
