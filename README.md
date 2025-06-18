📄 rith.py — Backend: Gesture Recognition Engine
markdown
Copy
Edit
# 🎯 rith.py — Core Gesture Recognition Engine

This is the backend module that powers all gesture detection and interaction logic for the touchless control system. It leverages MediaPipe's hand and face landmarks, Kalman filtering, and system control libraries (`pyautogui`, `win32api`) to offer real-time, touchless input handling.

### 🔍 Key Features

- Real-time webcam-based gesture tracking
- Finger state classification (pinch, scroll, copy, etc.)
- Whiteboard drawing engine
- Virtual monitor region (head-tracked draggable rectangle)
- Kalman filter for smoother pointer control
- Handles system actions (mouse, keyboard shortcuts)

### 🧠 How It Works

- **Finger Detection:** Identifies the state of each finger using MediaPipe landmarks
- **Gesture Classifier:** Determines active gestures (e.g., copy, paste, scroll)
- **Control Triggers:**
  - `pyautogui.click()` for mouse actions
  - `pyperclip.copy()` for clipboard functions
  - `win32api.keybd_event()` for keypress simulation
- **Overlay GUI:**
  - Shows feedback on gesture activation
  - Displays icons (copy.png, paste.png, etc.)

### 🛠️ Modules Used

- `cv2` — for camera input and drawing overlays
- `mediapipe` — hand/face detection
- `pyautogui`, `win32api`, `pyperclip` — for simulating input
- `numpy` — geometric calculations
- `KalmanFilter` — to smooth cursor movement

### 🧪 To Use

`rith.py` is not a standalone app. It’s imported and run by `single.py`.


Make sure your webcam is connected and clear lighting is available for detection.



---

## 🖼️ `single.py` — Streamlit Frontend Launcher

markdown
# 🌐 single.py — Streamlit Frontend Dashboard

This script launches an interactive Streamlit web dashboard that allows users to start the gesture controller, test their webcam, view tutorials, and meet the development team.

### 🧩 Components

- **Start Button:** Launches the gesture recognition backend (`rith.py`)
- **Lottie Animation:** Enhances visual appeal
- **Instructions Section:** 3-step setup with stylish tiles
- **Gesture Tutorial:** Shows demo images for each gesture
- **Camera Test Button:** Lets users quickly verify webcam functionality
- **Team Members:** Includes LinkedIn-linked images of contributors

### 💡 Highlights

- **Lottie Integration**: Plays animation from JSON (`Animation - 1742730378103.json`)
- **Gesture Preview**: Shows `index.png`, `copy.png`, `paste.png`, etc.
- **Streamlit Columns/Grid System**: Clean layout for steps and team
- **Live Camera Test**: `st.camera_input()` shows real-time feed

### ▶️ How to Run


streamlit run single.py
Ensure rith.py is in the same directory and dependencies are installed.

🛠️ Libraries Used
streamlit — web interface

streamlit_lottie — animation display

json — animation file loading

time — camera test duration

rith — the backend gesture logic
