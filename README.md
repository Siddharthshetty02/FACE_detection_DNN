# FACE_detection_DNN
## 📌 Overview

This project implements real-time face detection using OpenCV's Deep Neural Network (DNN) module. It utilizes a pre-trained Caffe model for detecting faces in video streams from a webcam.

## 🚀 Features

Uses Deep Learning-based face detection (more accurate than Haar cascades)

Detects faces in real-time using OpenCV's DNN module

Supports both CPU and GPU acceleration for faster detection

Works with pre-trained Caffe model

## 🛠️ Requirements

Ensure you have the following dependencies installed:

Python 3.x

OpenCV 4.x

NumPy

You can install them using:

pip install opencv-python numpy

## 📥 Model Files

Download the required model files:

Deploy Prototxt

Caffe Model

Save these files in the project directory.

## 📜 Usage

1️⃣ Clone the Repository

git clone https://github.com/Siddharthshetty02/FACE_detection_DNN.git
cd FACE_detection_DNN

2️⃣ Run the Face Detection Script

python DNN_detection_model.py

## 🔍 How It Works

The script loads the Caffe model for face detection.

It captures video from the webcam.

Each frame is processed using OpenCV's DNN module.

If faces are detected, bounding boxes are drawn around them.

The processed video stream is displayed in real-time.

Press 'q' to exit.

## 📸 Example Output

If a face is detected, a bounding box is drawn around it, along with a confidence score.

⚡ Performance Optimization

Use GPU acceleration: OpenCV DNN supports CUDA if configured properly.

Reduce the confidence threshold to detect more faces (default: 0.5).

## 🛠️ Troubleshooting

If you get an error "Can't open deploy.prototxt", check if the file paths are correct.

Make sure OpenCV is properly installed using pip show opencv-python.
