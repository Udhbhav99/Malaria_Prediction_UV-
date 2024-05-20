# Malaria Prediction Model Project Summary

The objective of this project is to develop a machine learning model that can accurately predict the presence of malaria from cell images. This involves distinguishing between parasitized (infected) and uninfected (healthy) cells.

### Data Source:

The dataset consists of images stored in two folders:
  - Parasitized: Contains images of cells infected with malaria.
  - Uninfected: Contains images of healthy cells.

Python: The programming language used for data processing, model training, and evaluation.

NumPy: Used to convert image data into numpy arrays for efficient processing and manipulation.

Keras with TensorFlow backend: Utilized for building and training the deep learning model.

VGG19 Pretrained Model: Leveraged with ImageNet weights for transfer learning to improve model performance and reduce training time.

1) Data Preprocessing:
  - Load images from the Parasitized and Uninfected folders.
  Convert images to numpy arrays for easier manipulation and processing.
  Normalize and preprocess the image data to make it suitable for model input.

  - Split the dataset into training and testing sets to evaluate model performance.
  Ensure a balanced split to maintain the integrity of the dataset.

2) Model Training:

  - Use the VGG19 model pretrained on ImageNet as the base model.
  Fine-tune the VGG19 model on the malaria cell image dataset to adapt it to the specific task of malaria detection.

  - Implement additional layers as needed to improve model accuracy.

  - Evaluate the trained model on the test set to measure its performance. 
