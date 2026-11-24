# MoodSync AI – Real-Time Face Emotion Detection & Music Recommendation System

> **Your mood, your music.**  
> A beginner-friendly BCA Applied AI mini-project built with Python, OpenCV, Haar Cascade, TensorFlow/Keras CNN, Scikit-learn, Flask, and HTML/CSS/Vanilla JavaScript.

---

## 1. Project Folder Structure

```text
MoodSync-AI/
│
├── app.py                     # Flask backend server & API routes
├── requirements.txt           # Python library dependencies
├── README.md                  # Setup, training & execution guide
│
├── model/
│   ├── emotion_model.py       # Haar Cascade detector, 48x48 preprocessor, CNN architecture
│   ├── train_model.py         # FER-2013 training & Scikit-learn evaluation script
│   └── saved_model/           # Stores trained emotion_cnn.h5 & evaluation_report.json
│
├── dataset/
│   └── songs.csv              # Local Hindi & English mood-mapped song catalog
│
├── static/
│   ├── css/
│   │   └── style.css          # Dark glassmorphism dashboard styling
│   ├── js/
│   │   └── script.js          # Webcam capture, Canvas overlay, audio player & UI logic
│   └── audio/                 # Local royalty-free demo WAVs + user MP3 files
│
└── templates/
    └── index.html             # Single-page MoodSync AI dashboard
```

---

## 2. Applied AI Syllabus Alignment

- **Unit I – Introduction to Applied AI:** Real-world Computer Vision & Deep Learning application.
- **Unit II – Computer Vision Basics:** Grayscale conversion, Gaussian filtering, image resizing (`48x48`), pixel normalization (`0..1`), OpenCV, and Haar Cascade face detection.
- **Unit III – Deep Learning for Vision:** 3-block Convolutional Neural Network (`Conv2D`, `BatchNormalization`, `MaxPooling2D`, `Dropout`, `Flatten`, `Dense`, `Softmax`) built in TensorFlow/Keras.
- **Unit IV – NLP Fundamentals:** Structured text metadata filtering (song title, artist, language, mood category).
- **Unit VI – Integrated Applications:** End-to-end Flask + Frontend integration and model evaluation (`Accuracy`, `Precision`, `Recall`, `F1-Score`, and `Confusion Matrix`).

---

## 3. Installation Instructions (Windows / VS Code)

1. Open the `MoodSync-AI` folder in **VS Code**.
2. Create and activate a Python virtual environment (recommended Python 3.10 or 3.11):
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

---

## 4. How to Prepare FER-2013 & Train the CNN Model

MoodSync AI classifies **4 emotions**: `Angry`, `Happy`, `Neutral`, and `Sad`.  
It does **not** include fake pretrained weights. To enable live CNN emotion classification:

1. Download the public **FER-2013** dataset from Kaggle (`fer2013.csv` or the image folder version).
2. Place the dataset in **one** of these two formats:
   - **CSV format:** `MoodSync-AI/dataset/fer2013.csv`
   - **Image folder format:**
     ```text
     MoodSync-AI/dataset/fer2013/
       ├── train/{angry, happy, neutral, sad}/
       └── test/{angry, happy, neutral, sad}/
     ```
3. Run the training & evaluation script:
   ```bash
   python model/train_model.py
   ```
4. Once complete, the script saves:
   - `model/saved_model/emotion_cnn.h5` (trained CNN weights)
   - `model/saved_model/evaluation_report.json` (real Accuracy, Precision, Recall, F1-Score, and Confusion Matrix)

---

## 5. How to Run the Flask Application

```bash
python app.py
```

Open your browser and visit:
```text
http://127.0.0.1:5000
```

---

## 6. How to Test Webcam Detection & Manual Mood Mode

1. Click **Start Detection** on the Home Dashboard (or **Live Detection** in the left sidebar).
2. Click **Start Webcam** and grant camera permission when prompted by your browser.
3. Face the webcam in good lighting:
   - OpenCV Haar Cascade detects your face and draws a purple bounding box around it.
   - If `emotion_cnn.h5` is trained, the CNN predicts `Happy`, `Sad`, `Angry`, or `Neutral` with a confidence percentage and automatically updates your playlist.
   - If `emotion_cnn.h5` is not trained yet, Haar Cascade face detection still runs and the UI displays an honest reminder to run `train_model.py`.
4. You can also click any **Manual Mood** button (`Happy`, `Sad`, `Angry`, `Neutral`) on the Dashboard or Live Detection view to explore songs without using a webcam.

---

## 7. How to Add Local Songs (MP3 / WAV)

Four original synthesized demo `.wav` tracks are included in `static/audio/` (one per mood) so Play/Pause controls work immediately. Copyrighted commercial MP3 files are **not** bundled.

To add your own legally obtained local `.mp3` or `.wav` files:
- **Option A (via UI):** Open **My Playlists**, fill out **Add Local Song / Audio File**, attach your `.mp3`/`.wav`, and click **Add to Playlist**.
- **Option B (manual):** Copy your file into `static/audio/my_song.mp3` and add a row to `dataset/songs.csv`:
  ```csv
  My Song Title,Artist Name,Hindi,Happy,static/audio/my_song.mp3
  ```

---

## 8. How the Emotion-to-Song Recommendation Works

| Detected Emotion | Music Strategy |
| :--- | :--- |
| **Happy** | Energetic, upbeat and motivational songs |
| **Sad** | Soft, emotional and soothing songs |
| **Angry** | Calm, peaceful and relaxing songs |
| **Neutral** | Chill, acoustic and pleasant songs |

The backend reads `dataset/songs.csv`, filters tracks by the detected/selected mood and optional language (`All`, `Hindi`, `English`), and verifies on disk whether each track's local `audio_path` exists before showing Play/Pause controls.
