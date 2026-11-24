"""
app.py
------
Flask backend server for MoodSync AI:
Real-Time Face Emotion Detection & Mood-Based Music Recommendation System.

Connects:
- OpenCV Haar Cascade face detection
- TensorFlow/Keras CNN emotion classifier
- Local CSV song recommendation engine (dataset/songs.csv)
- Local audio playback & optional user MP3/WAV upload
- HTML / CSS / Vanilla JS frontend (templates/index.html & static/)
"""

import base64
import csv
import json
import os
import random

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from model.emotion_model import (
    DEFAULT_EVAL_PATH,
    DEFAULT_MODEL_PATH,
    EMOTION_LABELS,
    detect_faces_haar,
    load_face_cascade,
    load_trained_model,
    predict_emotion,
    preprocess_face,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SONGS_CSV_PATH = os.path.join(BASE_DIR, "dataset", "songs.csv")
AUDIO_DIR = os.path.join(BASE_DIR, "static", "audio")
ALLOWED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".ogg", ".m4a"}

MOOD_PROFILES = {
    "Happy": {
        "title": "Energetic & Upbeat",
        "description": "Energetic, upbeat and motivational songs to match your bright mood.",
    },
    "Sad": {
        "title": "Soft & Soothing",
        "description": "Soft, emotional and soothing songs for comfort and gentle reflection.",
    },
    "Angry": {
        "title": "Calm & Peaceful",
        "description": "Calm, peaceful and relaxing songs to help ease tension and restore balance.",
    },
    "Neutral": {
        "title": "Chill & Acoustic",
        "description": "Chill, acoustic and pleasant songs for focus and easy listening.",
    },
}

app = Flask(__name__, template_folder="templates", static_folder="static")

# Load OpenCV Haar Cascade and trained CNN model safely at startup
face_cascade, cascade_error = load_face_cascade()
emotion_model, model_error = load_trained_model(DEFAULT_MODEL_PATH)


def load_songs_from_csv():
    """
    Reads dataset/songs.csv and validates whether each song's local audio
    file actually exists on disk before exposing Play/Pause capability.
    """
    songs = []
    if not os.path.exists(SONGS_CSV_PATH):
        return songs

    with open(SONGS_CSV_PATH, mode="r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for index, row in enumerate(reader, start=1):
            song_name = (row.get("song_name") or "").strip()
            artist = (row.get("artist") or "Unknown Artist").strip()
            language = (row.get("language") or "English").strip()
            mood = (row.get("mood") or "Neutral").strip()
            raw_audio_path = (row.get("audio_path") or "").strip().replace("\\", "/")

            if not song_name:
                continue

            audio_available = False
            audio_url = None

            if raw_audio_path:
                # Resolve relative to project root safely
                candidate_path = os.path.abspath(os.path.join(BASE_DIR, raw_audio_path))
                if candidate_path.startswith(BASE_DIR) and os.path.isfile(candidate_path):
                    audio_available = True
                    # Convert to Flask static URL path
                    rel_path = os.path.relpath(candidate_path, BASE_DIR).replace("\\", "/")
                    audio_url = "/" + rel_path

            songs.append({
                "id": index,
                "song_name": song_name,
                "artist": artist,
                "language": language,
                "mood": mood,
                "audio_path": raw_audio_path,
                "audio_url": audio_url,
                "audio_available": audio_available,
            })

    return songs


def load_evaluation_report():
    """Loads real evaluation metrics if train_model.py has been run."""
    if not os.path.exists(DEFAULT_EVAL_PATH):
        return None
    try:
        with open(DEFAULT_EVAL_PATH, mode="r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


@app.route("/")
def index():
    """Serves the main MoodSync AI dashboard."""
    return render_template("index.html")


@app.route("/api/status", methods=["GET"])
def api_status():
    """Returns honest runtime status of Haar Cascade, CNN model, and dataset."""
    eval_report = load_evaluation_report()
    songs = load_songs_from_csv()
    playable_count = sum(1 for s in songs if s["audio_available"])

    return jsonify({
        "app_name": "MoodSync AI",
        "backend": "Python Flask + OpenCV + TensorFlow/Keras",
        "haar_cascade_loaded": face_cascade is not None,
        "haar_cascade_error": cascade_error,
        "model_loaded": emotion_model is not None,
        "model_error": model_error,
        "model_path": os.path.relpath(DEFAULT_MODEL_PATH, BASE_DIR).replace("\\", "/"),
        "supported_emotions": EMOTION_LABELS,
        "evaluation_available": eval_report is not None,
        "evaluation_report": eval_report,
        "total_songs": len(songs),
        "playable_songs": playable_count,
    })


@app.route("/api/songs", methods=["GET"])
def api_songs():
    """
    Returns mood-filtered song recommendations from dataset/songs.csv.
    Query params:
      - mood: Happy | Sad | Angry | Neutral | All
      - language: All | Hindi | English
      - shuffle: true | false
    """
    mood = (request.args.get("mood") or "Happy").strip()
    language = (request.args.get("language") or "All").strip()
    should_shuffle = (request.args.get("shuffle") or "false").lower() == "true"

    all_songs = load_songs_from_csv()
    filtered = all_songs

    if mood and mood.lower() != "all":
        filtered = [s for s in filtered if s["mood"].lower() == mood.lower()]

    if language and language.lower() != "all":
        filtered = [s for s in filtered if s["language"].lower() == language.lower()]

    if should_shuffle:
        # Keep playable tracks prioritized when refreshing so user can always test audio
        playable = [s for s in filtered if s["audio_available"]]
        catalog_only = [s for s in filtered if not s["audio_available"]]
        random.shuffle(playable)
        random.shuffle(catalog_only)
        filtered = playable + catalog_only

    profile = MOOD_PROFILES.get(mood, MOOD_PROFILES["Neutral"])
    return jsonify({
        "mood": mood,
        "language": language,
        "profile": profile,
        "count": len(filtered),
        "songs": filtered,
    })


@app.route("/api/detect", methods=["POST"])
def api_detect():
    """
    Receives a base64 webcam image frame from the browser, runs OpenCV Haar
    Cascade face detection, preprocesses the face (48x48 grayscale), and
    performs CNN emotion prediction when the trained model is present.
    """
    if face_cascade is None:
        return jsonify({
            "success": False,
            "error": cascade_error or "OpenCV Haar Cascade classifier unavailable.",
        }), 500

    payload = request.get_json(silent=True) or {}
    frame_data = payload.get("image")

    if not frame_data or not isinstance(frame_data, str):
        return jsonify({
            "success": False,
            "error": "No image frame provided.",
        }), 400

    try:
        # Strip data URI header if present (e.g., data:image/jpeg;base64,...)
        if "," in frame_data:
            frame_data = frame_data.split(",", 1)[1]

        image_bytes = base64.b64decode(frame_data)
        np_buffer = np.frombuffer(image_bytes, dtype=np.uint8)
        bgr_frame = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)

        if bgr_frame is None:
            return jsonify({
                "success": False,
                "error": "Unable to decode webcam image frame.",
            }), 400
    except Exception as exc:
        return jsonify({
            "success": False,
            "error": f"Invalid image payload: {str(exc)}",
        }), 400

    frame_height, frame_width = bgr_frame.shape[:2]
    gray_frame, faces = detect_faces_haar(bgr_frame, face_cascade)

    if not faces:
        return jsonify({
            "success": True,
            "face_detected": False,
            "frame_size": {"width": frame_width, "height": frame_height},
            "bbox": None,
            "model_loaded": emotion_model is not None,
            "emotion": None,
            "confidence": 0.0,
            "probabilities": None,
            "message": "No face detected. Ensure good lighting and face the webcam directly.",
        })

    # Use the largest detected face
    x, y, w, h = faces[0]
    bbox = {"x": x, "y": y, "w": w, "h": h}
    face_input = preprocess_face(gray_frame, (x, y, w, h))

    if face_input is None:
        return jsonify({
            "success": True,
            "face_detected": False,
            "frame_size": {"width": frame_width, "height": frame_height},
            "bbox": None,
            "model_loaded": emotion_model is not None,
            "emotion": None,
            "confidence": 0.0,
            "probabilities": None,
            "message": "Detected face region was too small to preprocess.",
        })

    if emotion_model is None:
        return jsonify({
            "success": True,
            "face_detected": True,
            "frame_size": {"width": frame_width, "height": frame_height},
            "bbox": bbox,
            "preprocessed_shape": list(face_input.shape),
            "model_loaded": False,
            "emotion": None,
            "confidence": 0.0,
            "probabilities": None,
            "message": (
                "Face detected via Haar Cascade and preprocessed to 48x48 grayscale. "
                "Train the CNN model (python model/train_model.py) to enable emotion predictions."
            ),
        })

    # Run real CNN inference
    result = predict_emotion(emotion_model, face_input)
    return jsonify({
        "success": True,
        "face_detected": True,
        "frame_size": {"width": frame_width, "height": frame_height},
        "bbox": bbox,
        "preprocessed_shape": list(face_input.shape),
        "model_loaded": True,
        "emotion": result["emotion"],
        "confidence": result["confidence"],
        "probabilities": result["probabilities"],
        "message": f"Detected {result['emotion']} ({result['confidence']}% confidence).",
    })


@app.route("/api/upload_song", methods=["POST"])
def api_upload_song():
    """
    Allows the user to upload a legally obtained local MP3/WAV/OGG file
    into static/audio/ and append its metadata to dataset/songs.csv.
    """
    song_name = (request.form.get("song_name") or "").strip()
    artist = (request.form.get("artist") or "").strip()
    language = (request.form.get("language") or "English").strip()
    mood = (request.form.get("mood") or "Happy").strip()
    audio_file = request.files.get("audio_file")

    if not song_name or not artist:
        return jsonify({"success": False, "error": "Song name and artist are required."}), 400

    if mood not in EMOTION_LABELS:
        return jsonify({"success": False, "error": "Invalid mood category."}), 400

    saved_rel_path = ""
    if audio_file and audio_file.filename:
        safe_name = secure_filename(audio_file.filename)
        _, ext = os.path.splitext(safe_name.lower())
        if ext not in ALLOWED_AUDIO_EXTENSIONS:
            return jsonify({
                "success": False,
                "error": "Only .mp3, .wav, .ogg, and .m4a audio files are supported.",
            }), 400

        os.makedirs(AUDIO_DIR, exist_ok=True)
        save_path = os.path.join(AUDIO_DIR, safe_name)
        audio_file.save(save_path)
        saved_rel_path = f"static/audio/{safe_name}"

    file_exists = os.path.exists(SONGS_CSV_PATH)
    with open(SONGS_CSV_PATH, mode="a", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        if not file_exists:
            writer.writerow(["song_name", "artist", "language", "mood", "audio_path"])
        writer.writerow([song_name, artist, language, mood, saved_rel_path])

    return jsonify({
        "success": True,
        "message": f"Added '{song_name}' to {mood} playlist.",
    })


if __name__ == "__main__":
    print("Starting MoodSync AI Flask Server on http://127.0.0.1:5000")
    if model_error:
        print(f"[NOTICE] {model_error}")
    app.run(host="0.0.0.0", port=5000, debug=True)
