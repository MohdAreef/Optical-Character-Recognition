# Optical-Character-Recognition
This repository contains the code and resources for an Optical Character Recognition (OCR) project developed during my internship at Emagia Corporations. The OCR model is designed to recognize handwritten words with an accuracy of 82%.
## Model Architecture
CNN (ResNet101V2): Utilized ResNet101V2 pretrained model for feature extraction, leveraging Transfer Learning to focus on local patterns and textures within handwritten characters.
RNN: Employed Recurrent Neural Networks to process the sequential features extracted by the CNN, capturing the dependencies between characters and enhancing robustness to handwriting variations.
##Key Features
Feature Extraction: ResNet101V2 pretrained model for efficient and effective feature extraction.
Sequential Processing: RNN for handling the sequence of characters, improving recognition accuracy for handwritten text.
