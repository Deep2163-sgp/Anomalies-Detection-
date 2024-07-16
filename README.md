# Video Anomaly Detection 

## Abstract
This project aims to detect anomalies in videos using a combination of Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) networks. Anomalies in videos can include activities such as fighting, stealing, and other suspicious behaviors. By leveraging deep learning techniques, this project provides an efficient method to identify and classify such activities, which can be highly beneficial in security and surveillance systems.

## Project Structure
1. **Anomalies(cnn+lstm).ipynb**: Notebook file for training and testing anomalies. This Jupyter Notebook contains the code to preprocess the dataset, build the CNN-LSTM model, and evaluate its performance.
2. **Anomalies.h5**: Pre-trained model to load and test on videos. This file contains the weights of the trained model, allowing for quick testing and validation on new video samples.

## Dataset
The dataset used for this project is the UCF Crime Dataset, which includes various types of anomalous activities in videos. The dataset can be accessed and downloaded from the following link:
[UCF Crime Dataset on Kaggle](https://www.kaggle.com/datasets/odins0n/ucf-crime-dataset)

## Usage
1. **Training the Model**:
   - Open the `Anomalies(cnn+lstm).ipynb` notebook.
   - Follow the instructions to preprocess the dataset.
   - Train the model by running the provided cells.
   - Save the trained model for future use.

2. **Testing the Model**:
   - Load the pre-trained model `Anomalies.h5`.
   - Use the provided functions in the notebook to test the model on new video samples.
   - Evaluate the model's performance based on the results.

## Requirements
- Python 3.x
- TensorFlow
- Keras
- OpenCV
- NumPy
- Pandas
- Jupyter Notebook

Install the required packages using pip:
```bash
pip install tensorflow keras opencv-python numpy pandas jupyter
