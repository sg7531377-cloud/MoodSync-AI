import express from 'express';
import fs from 'node:fs';
import path from 'node:path';
import {
  escapeCsvCell,
  filterAndRecommendSongs,
  MOOD_PROFILES,
  parseSongsCsvContent,
  SUPPORTED_EMOTIONS,
} from './src/moodsync/catalog.ts';
import { extractBoundary, parseMultipartBody } from './src/moodsync/multipart.ts';

const PORT = Number(process.env.PORT) || 3000;
const ROOT_DIR = process.cwd();
const MOODSYNC_DIR = path.join(ROOT_DIR, 'MoodSync-AI');
const TEMPLATES_DIR = path.join(MOODSYNC_DIR, 'templates');
const STATIC_DIR = path.join(MOODSYNC_DIR, 'static');
const AUDIO_DIR = path.join(STATIC_DIR, 'audio');
const SONGS_CSV_PATH = path.join(MOODSYNC_DIR, 'dataset', 'songs.csv');
const MODEL_PATH = path.join(MOODSYNC_DIR, 'model', 'saved_model', 'emotion_cnn.h5');
const EVAL_PATH = path.join(MOODSYNC_DIR, 'model', 'saved_model', 'evaluation_report.json');

const ALLOWED_AUDIO_EXTENSIONS = new Set(['.mp3', '.wav', '.ogg', '.m4a']);

function readEvaluationReport(): Record<string, unknown> | null {
  if (!fs.existsSync(EVAL_PATH)) return null;
  try {
    return JSON.parse(fs.readFileSync(EVAL_PATH, 'utf8')) as Record<string, unknown>;
  } catch {
    return null;
  }
}

function readSongsCatalog() {
  if (!fs.existsSync(SONGS_CSV_PATH)) return [];
  const csvText = fs.readFileSync(SONGS_CSV_PATH, 'utf8');
  return parseSongsCsvContent(csvText, MOODSYNC_DIR);
}

function sanitizeFilename(rawName: string): string {
  const base = path.basename(rawName).replace(/[^a-zA-Z0-9._-]/g, '_');
  return base || `track_${Date.now()}.mp3`;
}

const app = express();
app.use(express.json({ limit: '8mb' }));
app.use('/static', express.static(STATIC_DIR));

app.get('/', (_req, res) => {
  res.sendFile(path.join(TEMPLATES_DIR, 'index.html'));
});

app.get('/api/status', (_req, res) => {
  const modelExists = fs.existsSync(MODEL_PATH);
  const evalReport = readEvaluationReport();
  const songs = readSongsCatalog();
  const playableCount = songs.filter((s) => s.audio_available).length;

  res.json({
    app_name: 'MoodSync AI',
    backend: 'Preview Server (Run python app.py locally for OpenCV + TensorFlow)',
    haar_cascade_loaded: true,
    haar_cascade_error: null,
    model_loaded: modelExists,
    model_error: modelExists
      ? null
      : "Trained model file not found at 'model/saved_model/emotion_cnn.h5'. Run 'python model/train_model.py' after preparing FER-2013, then launch 'python app.py'.",
    model_path: 'model/saved_model/emotion_cnn.h5',
    supported_emotions: [...SUPPORTED_EMOTIONS],
    evaluation_available: evalReport !== null,
    evaluation_report: evalReport,
    total_songs: songs.length,
    playable_songs: playableCount,
  });
});

app.get('/api/songs', (req, res) => {
  const mood = String(req.query.mood || 'Happy').trim();
  const language = String(req.query.language || 'All').trim();
  const shuffle = String(req.query.shuffle || 'false').toLowerCase() === 'true';

  const allSongs = readSongsCatalog();
  const songs = filterAndRecommendSongs(allSongs, mood, language, shuffle);
  const profile = MOOD_PROFILES[mood] || MOOD_PROFILES.Neutral;

  res.json({
    mood,
    language,
    profile,
    count: songs.length,
    songs,
  });
});

app.post('/api/detect', (req, res) => {
  const image = req.body && typeof req.body.image === 'string' ? req.body.image : '';
  if (!image) {
    res.status(400).json({ success: false, error: 'No image frame provided.' });
    return;
  }

  const modelExists = fs.existsSync(MODEL_PATH);
  res.json({
    success: true,
    face_detected: false,
    frame_size: { width: 640, height: 480 },
    bbox: null,
    model_loaded: modelExists,
    emotion: null,
    confidence: 0,
    probabilities: null,
    message:
      'OpenCV Haar Cascade & TensorFlow CNN run in Python Flask (python app.py). Use Manual Mood Selection here or run app.py locally with your trained FER-2013 model.',
  });
});

app.post(
  '/api/upload_song',
  express.raw({ type: 'multipart/form-data', limit: '25mb' }),
  (req, res) => {
    const contentType = req.headers['content-type'] || '';
    const boundary = extractBoundary(contentType);
    if (!boundary || !Buffer.isBuffer(req.body)) {
      res.status(400).json({ success: false, error: 'Invalid multipart upload request.' });
      return;
    }

    const parsed = parseMultipartBody(req.body, boundary);
    const songName = (parsed.fields.song_name || '').trim();
    const artist = (parsed.fields.artist || '').trim();
    const language = (parsed.fields.language || 'English').trim();
    const mood = (parsed.fields.mood || 'Happy').trim();

    if (!songName || !artist) {
      res.status(400).json({ success: false, error: 'Song name and artist are required.' });
      return;
    }

    if (!SUPPORTED_EMOTIONS.includes(mood as (typeof SUPPORTED_EMOTIONS)[number])) {
      res.status(400).json({ success: false, error: 'Invalid mood category.' });
      return;
    }

    let savedRelPath = '';
    if (parsed.file && parsed.file.filename) {
      const safeName = sanitizeFilename(parsed.file.filename);
      const ext = path.extname(safeName).toLowerCase();
      if (!ALLOWED_AUDIO_EXTENSIONS.has(ext)) {
        res.status(400).json({
          success: false,
          error: 'Only .mp3, .wav, .ogg, and .m4a audio files are supported.',
        });
        return;
      }

      fs.mkdirSync(AUDIO_DIR, { recursive: true });
      fs.writeFileSync(path.join(AUDIO_DIR, safeName), parsed.file.data);
      savedRelPath = `static/audio/${safeName}`;
    }

    const fileExists = fs.existsSync(SONGS_CSV_PATH);
    const rowLine =
      [
        escapeCsvCell(songName),
        escapeCsvCell(artist),
        escapeCsvCell(language),
        escapeCsvCell(mood),
        escapeCsvCell(savedRelPath),
      ].join(',') + '\n';

    if (!fileExists) {
      fs.mkdirSync(path.dirname(SONGS_CSV_PATH), { recursive: true });
      fs.writeFileSync(
        SONGS_CSV_PATH,
        'song_name,artist,language,mood,audio_path\n' + rowLine,
        'utf8'
      );
    } else {
      const current = fs.readFileSync(SONGS_CSV_PATH, 'utf8');
      const prefix = current.length > 0 && !current.endsWith('\n') ? '\n' : '';
      fs.appendFileSync(SONGS_CSV_PATH, prefix + rowLine, 'utf8');
    }

    res.json({
      success: true,
      message: `Added '${songName}' to ${mood} playlist.`,
    });
  }
);

app.listen(PORT, '0.0.0.0', () => {
  console.log(`MoodSync AI preview server running at http://localhost:${PORT}`);
});
