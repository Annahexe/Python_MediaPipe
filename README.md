# Python MediaPipe – Real-Time Gesture & Pose Detection 🖐️

<p align="center">
  <img src="https://res.cloudinary.com/dwm2oprta/image/upload/v1769977970/ThumbThumbNail_nsozqi.png" height="300" alt="Thumbsup thumbnail" />
</p>

This repository contains a collection of **real-time computer vision demos** built with **Python, OpenCV, and MediaPipe**.  
Each script uses a webcam (or video input) to detect **hands, faces, and body poses**, and provides visual feedback directly on the video stream.

The project also includes a **launcher interface** that lets you run all demos from a single camera-based menu.

---

## ⭐ Featured: Like / Dislike Gesture System

### Like–Dislike Gesture System
- Detects **thumbs up / thumbs down** gestures
- Ensures the hand is **closed** to avoid false positives
- Displays:
  - `Like 🙂`, `Dislike 🙁`, or `Neutral`
  - A colored marker on the thumb
- Spawns **animated floating reaction icons** (👍 / 👎) with fade-out effects

**Script:** `likeDislikeGestureSystem.py`

---

## 📂 Included Scripts

### 🖐 Hand & Gesture Detection
- **Like / Dislike gesture system**  
  `likeDislikeGestureSystem.py`
- **Index finger up / down detection**  
  `indexHandGesture.py`

### 🙂 Face Detection
- **Face detection with counter**  
  `facesDetectionCounter.py`  
  Detects and counts faces in real time.

### 🕺 Pose & Body Detection
- **Right arm raised detection**  
  `armRaisedDetection.py`  
  Detects whether the right arm is raised above the shoulder.
- **Pose drawing (geometric shapes)**  
  `poseDrawing.py`  
  Draws simple shapes using detected body landmarks.
- **Pose video recorder**  
  `poseVideoRecorder.py`  
  Processes video/camera input and saves the pose-annotated result as an MP4 file.

### 🚀 Launcher
- **Camera-based launcher menu**  
  `launcher.py`  
  Displays a clickable menu on top of the webcam feed to launch each script without using the terminal.

---

## ▶ How to Run

### Requirements
- **Python 3.10**
- opencv-python mediapipe numpy
- A webcam

---

## 📄 License & Project Scope

This repository is a small educational study project created for learning and experimentation with MediaPipe and real-time computer vision.

The code is not intended for production use

Detection logic is simplified for clarity and learning purposes

Feel free to study, modify, and reuse the code for personal or educational projects

No formal license is applied.

© 2025

---

Thanks for checking it out! 
