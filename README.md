# Plant Disease Classification App

This app classifies plant diseases from leaf images using a deep convolutional neural network model.

## Overview

<img title="Plant Disease Classification App Overview" alt="Overview of the app" src="/image/app_overview.png">



The app allows users to:

- Upload an image of a plant leaf
- Classify disease in the leaf (if any)
- View the predicted disease name and treatment recommendation

It uses a CNN model trained on plant leaf images to identify 14 different diseases.

## Usage

### Dependencies

- Python 3.7+
- TensorFlow 2.0+
- OpenCV
- Streamlit
- Numpy
- Pandas
- PIL

### Requirements
To install the dependencies for this app, you will need the following packages:

- numpy - For numerical processing
- opencv-python - For image processing
- pandas - For data manipulation
- Pillow - For image handling
- streamlit - For the web app
- tensorflow- For the deep learning model

  
To install, run:

```bash
pip install -r requirements.txt
```
This will install the required packages.



### Run the app

```bash
streamlit run app.py
```
## Use the app

1 .Upload an image of a diseased plant leaf
2 .Hit submit
3 .View the predicted disease name and treatment recommendation

## Model

The app uses a convolutional neural network model trained on the PlantVillage dataset.
The model file is located at models/plant_disease_model.h5.

It can classify between 14 different plant diseases.

## Customization
-  The plant diseases and treatment recommendations are configured in data.py. Add or remove diseases by modifying this file.
-  Retrain the model on new data to handle more plant disease classes by running train.py with an updated dataset.

## Credits
- The model is trained on the PlantVillage dataset.
- Built with Streamlit and TensorFlow.





