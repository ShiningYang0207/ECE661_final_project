# Image Synthesis Using A single Robust Classifier


In this project, we performed several image synthesis tasks (image generation, image inpainting, super resolution, image-to-image translation, and sketch-to-image translation) using a pre-trained adversarially robust classifier released by Santurkar et al (2019). Due to the limitation in computation resource, we used ImageNette dataset, which is a subset of 10 easily classified classes from Imagenet. In line with the paper, we found that the features learned by a basic classifier are sufficient for all these tasks, provided this classifier is adversarially robust. The idea for these augmentations, is taking the original adversarial model and using L2 FGSM attacks in various means to achieve the synthesis tasks.


Steps to run this code:

1) Clone this repository


2)Within this folder clone the robustness repo found here: https://github.com/MadryLab/robustness_applications


3) Create a Models folder within the robustness_applications folder.


4) Download the pretrained weights for the adversarially trained Imagenet Classifer found here: http://andrewilyas.com/ImageNet.pt

5) Download and save the Imagenette dataset into the following paths robustness_applications/dataset/val and robustness_applications/dataset/train

6) Modify robustness_applications/robustness/tools/user_constants.py to point to where the datasets are saved

7) Either move the files in the attacks folder into the robustness_application folder or ensure that the python notebook you are running is pointing to the right attack file for each corresponding image synthesis tasks

8) Run the python notebook for the desired synthesis tasks but change the paths to match those on your instance.


## Contributors

| Name | Reference |
|---- | ----|
|Abhijith Tammanagari | [GitHub Profile](https://github.com/23abhijith)|
|Shining Yang | [GitHub Profile](https://github.com/ShiningYang0207)|
|Minjung Lee |[GitHub Profile]()|
