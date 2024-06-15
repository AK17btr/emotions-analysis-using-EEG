# EEG Emotion Analysis

## Project Overview

EEG Emotion Analysis aims to predict emotional states using electroencephalography (EEG) data. EEG measures electrical activity in the brain through electrodes placed on the scalp, capturing changes in 
brain activity that correspond to different emotional states.

## Table of Contents
- [Dataset](#dataset)
- [Features](#features)
- [Data Processing](#data-processing)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)


## Dataset

The dataset contains 1000 trials, each a 4-second EEG signal labeled with the corresponding emotional state (happiness, sadness, anger, fear). The dataset includes:
- **Time Domain Features**: Mean, standard deviation, peak-to-peak amplitude, root mean square.
- **Frequency Domain Features**: Power spectral density (PSD), mean frequency, dominant frequency.
- **Time-Frequency Domain Features**: Hjorth parameters, wavelet transform, short-time Fourier transform.
- **Spatial Domain Features**: Laplacian of the EEG signal, coherence between different electrodes.

## Features

### Time Domain
- **Mean**: Average value of the EEG signal.
- **Standard Deviation**: Measure of the amount of variation in the EEG signal.
- **Peak-to-Peak Amplitude**: Difference between the maximum and minimum values of the EEG signal.
- **Root Mean Square**: Square root of the mean of the squared values of the EEG signal.

### Frequency Domain
- **Power Spectral Density (PSD)**: Power of the EEG signal as a function of frequency.
- **Mean Frequency**: Average frequency of the EEG signal.
- **Dominant Frequency**: Frequency with the highest power in the EEG signal.

### Time-Frequency Domain
- **Hjorth Parameters**: Characterize the shape of the EEG signal's power spectrum.
- **Wavelet Transform**: Decomposes the EEG signal into a series of wavelets.
- **Short-Time Fourier Transform**: Decomposes the EEG signal into frequency components over a short time window.

### Spatial Domain
- **Laplacian of the EEG Signal**: Measure of the spatial variation of the EEG signal.
- **Coherence Between Electrodes**: Correlation between EEG signals at different electrodes.

## Data Processing

1. **Import Libraries**: Essential libraries for data processing and model building.
2. **Load Dataset**: Load and inspect the dataset.
3. **Feature Selection**: Select necessary features for model training.
4. **Data Splitting**: Split data into training and testing sets.
5. **Label Encoding**: Encode categorical labels into numerical values.
6. **Reshape Data**: Reshape data for model compatibility (3D arrays).

## Model Architecture

The model uses a neural network with the following architecture:
- **Input Layer**: Accepts input in the shape of (number of features, 1).
- **GRU Layer**: Gated Recurrent Unit layer with 512 units.
- **Flatten Layer**: Converts the GRU output into a one-dimensional vector.
- **Dense Layer**: Fully connected layer with 3 output units and softmax activation for classification.

## Installation

To install the necessary dependencies, run:
```bash
pip install -r requirements.txt

## Workflow

Step 1: Load and preprocess the EEG dataset.
Step 2: Extract relevant features from the EEG signals.
Step 3: Train the GRU-based neural network model on the training data.
Step 4: Evaluate the model on the test data to obtain performance metrics.
Step 5: Use the trained model to predict emotional states from new EEG data

## Usage

This project predicts emotional states from EEG data. By processing and analyzing EEG signals,
the model classifies these signals into one of three categories: negative, neutral, and positive emotions.
This can be useful in various applications such as mental health monitoring, brain-computer interfaces, and user experience research.

Results
The model's performance is evaluated using accuracy, confusion matrix, and classification report. These metrics help in assessing how well the model can classify EEG signals into the correct emotional states.
