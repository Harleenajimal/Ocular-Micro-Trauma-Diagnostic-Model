# Ocular-Micro-Trauma-Diagnostic-Model
AI-Powered Ocular Micro-Trauma Detection System Using Real-Time CNN-LSTM Analysis of Saccadic Eye Dynamics
A real-time intelligent diagnostic platform that detects subtle ocular micro-trauma patterns using deep learning, computer vision, and live eye-movement analytics.

This project combines a hybrid CNN-LSTM architecture with browser-based eye tracking to analyze saccadic motion, velocity, acceleration, and temporal gaze dynamics in real time. The system continuously evaluates ocular behavior from a live webcam feed and predicts whether eye movement patterns resemble a healthy baseline or a strained/post-impact state.

Designed as a research-driven health-tech prototype, the platform demonstrates how non-contact ocular biometrics can be leveraged for early neurological and fatigue-related screening applications.

Features
Real-time ocular movement tracking using webcam
Live gaze velocity and acceleration analysis
Hybrid CNN-LSTM deep learning architecture
Browser-native eye tracking using MediaPipe Face Mesh
Live prediction API powered by Flask
Dynamic frontend dashboard with animated diagnostics
Real-time Chart.js visualization of eye movement signals
Glassmorphism-inspired premium UI
End-to-end inference pipeline from webcam → AI model → live prediction

System Architecture
Webcam Feed
     ↓
MediaPipe Face Mesh
     ↓
Eye Landmark Extraction
     ↓
Velocity & Acceleration Signal Processing
     ↓
50-Frame Sequential Window Buffer
     ↓
CNN-LSTM Deep Learning Model
     ↓
Live Diagnostic Prediction
     ↓
Interactive Visualization Dashboard

Deep Learning Pipeline
Model Architecture

The system utilizes a hybrid CNN-LSTM architecture:

CNN Layer

Extracts spatial-temporal motion patterns from ocular movement signals.

LSTM Layer

Learns sequential dependencies and temporal gaze dynamics across continuous eye movement windows.

Input Signals
Eye velocity
Eye acceleration
Sequential gaze movement patterns
Performance
Experimental Subset Evaluation
Metric	Value
Accuracy	66%
ROC-AUC	0.7289

This initial result validated the feasibility of detecting strain-related ocular patterns using sequential eye-movement analysis.

Full-Scale Training Pipeline

After scaling to over 15 million ocular sequences and optimizing the inference pipeline:

Metric	Value:
Accuracy	99%
ROC-AUC	0.9997

Improvements Included:
Large-scale dataset training
CNN-LSTM optimization
Stable inference tuning
Removal of unstable Batch Normalization layers
Improved temporal generalization
Enhanced preprocessing pipeline

Technologies Used:
Artificial Intelligence & ML
TensorFlow / Keras
CNN-LSTM Hybrid Networks
Sequential Signal Processing
Computer Vision
MediaPipe Face Mesh
Real-time eye landmark tracking

Backend:
Python
Flask REST API

Frontend:
HTML5
CSS3
JavaScript
Chart.js

Research Disclaimer

This project is a research and educational prototype and is not intended for clinical diagnosis or medical decision-making.

The current system demonstrates proof-of-concept feasibility for detecting strain-related ocular movement patterns using AI-driven temporal analysis. Clinical validation and subject-independent testing are required before real-world deployment.

Future Improvements
Subject-independent validation
Multi-class trauma severity detection
Blink instability analysis
Ocular jerk profiling
Temporal risk scoring
Edge-device optimization
Mobile deployment
Explainable AI visualization
Eye trajectory heatmaps
Author

Developed as an advanced AI + Computer Vision research project focused on real-time ocular micro-trauma diagnostics using deep learning and sequential gaze analytics.
