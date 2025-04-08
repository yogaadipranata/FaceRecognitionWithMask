## Face Recognition with Mask using Convolutional Neural Network & Haar Cascade

A real-time face mask detection system built with Python, using OpenCV, TensorFlow, Keras, CNN (MobileNetV2), and Haar Cascade for face detection. Developed as a college assignment, this project classifies whether a person is wearing a mask or not in live video. The model was trained on a Kaggle dataset via Google Colab and demonstrates the integration of deep learning with computer vision for real-world safety applications.

### 🛠️ Tech Stack
* Python
* OpenCV
* CNN-MobileNetV2
* Haar Cascade
* Google Colab
* Jupyter Notebook

## 🚀 Usage

### 💻 Run Locally
1. Clone the repository:
```https://github.com/yogaadipranata/FaceRecognitionWithMask.git```
2. Navigate to the project directory:
```cd FaceRecognitionWithMask```
3. Launch Jupyter Notebook.
4. Open mask-detector.ipynb and run the cells step-by-step.

### 🔗 Model Overview
* Architecture: MobileNetV2 (for feature extraction) + custom CNN head
* Face Detection: Haar Cascade Classifier (OpenCV)
* Frameworks: TensorFlow & Keras
* Evaluation Metrics: Accuracy, Precision, Recall

### 📊 Results
Based on testing, the program achieved an accuracy rate of 73%. The key strengths observed during evaluation include:
* Accurately detects faces even when users wear accessories such as hats or glasses.
* Capable of detecting multiple faces simultaneously.
* Successfully identifies individuals not wearing a mask, even when their face is partially covered by a hand.
* Real-time face detection performance.
* Does not require high-spec hardware since the model runs via Jupyter Notebook.

## 🤝 Contributing Guidelines
1. Fork the project
2. Create a feature branch (```git checkout -b ...```)
3. Commit your changes (```git commit -m "..."```)
4. Push to the branch (```git push origin ...```)
5. Open a pull request

## 📫 Contact
I Wayan Yoga Adi Pranata

[Email](mailto:yogaadipranata10@gmail.com)
