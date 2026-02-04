# Web Interface Documentation

A beautiful, modern web interface for the LAALM multi-modal transcription system.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [API Endpoints](#api-endpoints)
- [Frontend Components](#frontend-components)
- [Development](#development)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)

## Overview

The LAALM web interface provides an intuitive, user-friendly way to interact with the multi-modal transcription system. Built with React and FastAPI, it offers real-time processing feedback and beautiful visualizations of results.

**Tech Stack:**
- **Frontend**: React + Vite + TailwindCSS
- **Backend**: FastAPI + Uvicorn
- **Communication**: REST API with JSON

## Features

### User Interface
- ✨ **Elegant Design**: Modern glassmorphism UI with smooth animations
- 📤 **Drag & Drop**: Easy file upload for video and audio files
- 📊 **Real-time Progress**: Visual feedback during transcription
- 📈 **Confidence Metrics**: Word-level confidence visualization
- 💾 **Export Results**: Download transcriptions as JSON, SRT, or VTT

### Technical Features
- 🚀 **Fast Processing**: Parallel audio and video processing
- 🔄 **Auto-refresh**: Real-time status updates
- 📱 **Responsive**: Works on desktop, tablet, and mobile
- 🎨 **Customizable**: Easy theme and configuration changes

## Quick Start

### Prerequisites

- Python 3.11+ with LAALM dependencies installed
- Node.js 20.x or higher
- API keys configured in `.env` file

### Start the Application

**Option 1: Use startup script (recommended)**
```bash
./start_web.sh
```

**Option 2: Start manually**
```bash
# Terminal 1 - Backend
source .venv/bin/activate
python api.py

# Terminal 2 - Frontend
cd frontend
npm install  # First time only
npm run dev
```

**Access the application:**
- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## Architecture

### System Overview

```
┌─────────────┐         ┌──────────────┐         ┌─────────────┐
│   Browser   │ ◄─────► │ React Frontend│ ◄─────► │ FastAPI     │
│  (Client)   │  HTTP   │  (Port 5173)  │  REST   │ (Port 8000) │
└─────────────┘         └──────────────┘         └─────────────┘
                                                         │
                                                         ▼
                                                  ┌─────────────┐
                                                  │  Pipeline   │
                                                  │  - DeepGram │
                                                  │  - Auto-AVSR│
                                                  │  - Groq LLM │
                                                  └─────────────┘
```

### Project Structure

```
LAALM/
├── api.py                  # FastAPI backend server
├── start_web.sh           # Startup script
│
├── frontend/              # React frontend
│   ├── src/
│   │   ├── App.jsx       # Main component
│   │   ├── App.css       # Component styles
│   │   ├── index.css     # Global styles
│   │   └── main.jsx      # Entry point
│   ├── package.json      # Dependencies
│   ├── vite.config.js    # Vite config
│   └── tailwind.config.js # Tailwind config
│
├── uploads/               # Temporary file storage
└── logs/                  # Transcription logs
```

## API Endpoints

### Health Check

**GET /**

Check if the API is running.

**Response:**
```json
{
  "status": "online",
  "message": "LAALM API is running",
  "models_loaded": true
}
```

### Transcribe

**POST /transcribe**

Main transcription endpoint.

**Request:**
- Content-Type: `multipart/form-data`
- `video`: Video file (required)
- `audio`: Audio file (optional)

**Example (curl):**
```bash
curl -X POST http://localhost:8000/transcribe \
  -F "video=@samples/video/lwwz9s.mpg" \
  -F "audio=@samples/audio/lwwz9s.wav"
```

**Response:**
```json
{
  "audio_transcript": "lay white with zero again",
  "audio_confidence": 0.987,
  "video_transcript": "lay white at zero again",
  "video_confidence": 0.506,
  "final_transcript": "lay white with zero again",
  "final_confidence": 0.987,
  "agreement_rate": 0.333,
  "word_details": [
    {
      "word": "lay",
      "audio_conf": 0.99,
      "video_conf": 0.51,
      "source": "audio"
    }
  ],
  "corrections_applied": 5,
  "processing_time": 12.34,
  "timestamp": "2026-02-04T17:00:00"
}
```

### Get Logs

**GET /logs?limit=10**

Retrieve recent transcription logs.

**Parameters:**
- `limit` (optional): Number of recent logs to return (default: 10)

**Response:**
```json
{
  "logs": [
    {
      "timestamp": "2026-02-04T17:00:00",
      "video_file": "lwwz9s.mpg",
      "final_transcript": "lay white with zero again",
      "confidence": 0.987
    }
  ]
}
```

### Get Statistics

**GET /stats**

Get system statistics.

**Response:**
```json
{
  "total_uploads": 42,
  "total_logs": 156,
  "disk_usage_mb": 234.5,
  "uptime_seconds": 3600
}
```

### Delete Upload

**DELETE /uploads/{filename}**

Delete an uploaded file.

**Example:**
```bash
curl -X DELETE http://localhost:8000/uploads/video_20260204_170000.mpg
```

## Frontend Components

### Main App Component

**Location:** `frontend/src/App.jsx`

**Key Features:**
- File upload with drag-and-drop
- Progress tracking
- Results display
- Export functionality

**State Management:**
```javascript
const [videoFile, setVideoFile] = useState(null);
const [audioFile, setAudioFile] = useState(null);
const [isProcessing, setIsProcessing] = useState(false);
const [results, setResults] = useState(null);
const [error, setError] = useState(null);
```

### Styling

**Global Styles:** `frontend/src/index.css`
- TailwindCSS utilities
- Custom CSS variables
- Glassmorphism effects

**Component Styles:** `frontend/src/App.css`
- Component-specific styles
- Animations and transitions
- Responsive breakpoints

**Design System:**
- **Primary Color**: Indigo (#6366F1)
- **Secondary Color**: Purple (#8B5CF6)
- **Background**: Dark gradient (#0A0E27 → #1a1f3a)
- **Glass Effect**: White overlay with backdrop blur

## Development

### Frontend Development

```bash
cd frontend

# Install dependencies
npm install

# Start dev server with hot reload
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview

# Lint code
npm run lint
```

### Backend Development

```bash
# Activate environment
source .venv/bin/activate

# Run with auto-reload
python api.py

# Or use uvicorn directly
uvicorn api:app --reload --host 0.0.0.0 --port 8000

# Run in background
nohup python api.py > api.log 2>&1 &
```

### Adding New Features

#### 1. Add Backend Endpoint

Edit `api.py`:
```python
@app.post("/new-endpoint")
async def new_endpoint(data: dict):
    # Your logic here
    return {"result": "success"}
```

#### 2. Add Frontend Component

Edit `frontend/src/App.jsx`:
```javascript
const handleNewFeature = async () => {
  const response = await axios.post('/new-endpoint', data);
  setResults(response.data);
};
```

#### 3. Update Styling

Edit `frontend/tailwind.config.js` for theme changes:
```javascript
module.exports = {
  theme: {
    extend: {
      colors: {
        'custom-blue': '#1E40AF',
      }
    }
  }
}
```

## Deployment

### Production Build

**Frontend:**
```bash
cd frontend
npm run build
# Output in frontend/dist/
```

**Backend:**
```bash
# Use production ASGI server
pip install gunicorn
gunicorn api:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Docker Deployment (Optional)

Create `Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t laalm-api .
docker run -p 8000:8000 -v $(pwd)/.env:/app/.env laalm-api
```

### Nginx Configuration

```nginx
server {
    listen 80;
    server_name your-domain.com;

    # Frontend
    location / {
        root /path/to/frontend/dist;
        try_files $uri $uri/ /index.html;
    }

    # Backend API
    location /api/ {
        proxy_pass http://localhost:8000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Troubleshooting

### Backend Won't Start

**Problem:** Port 8000 already in use

**Solution:**
```bash
# Find process using port 8000
lsof -i :8000

# Kill the process
kill -9 <PID>

# Or use a different port
uvicorn api:app --port 8001
```

### Frontend Won't Start

**Problem:** Node modules missing or outdated

**Solution:**
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
```

### CORS Errors

**Problem:** Cross-origin request blocked

**Solution:** Check CORS configuration in `api.py`:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### File Upload Fails

**Problem:** File too large or wrong format

**Solution:**
- Check file size limits in `api.py`
- Verify file format is supported (.mpg, .mp4, .wav, etc.)
- Check disk space in `uploads/` directory

### Model Not Found

**Problem:** Backend can't find pre-trained models

**Solution:**
```bash
# Verify models exist
ls -lh auto_avsr/pretrained_models/

# Download models if missing
cd auto_avsr
python verify_setup.py
```

### Slow Processing

**Problem:** Transcription takes too long

**Solution:**
- Use GPU acceleration (check CUDA installation)
- Process smaller video files
- Compress videos before upload
- Check API rate limits

## Performance Tips

1. **Use separate audio file** for faster processing
2. **Compress videos** before upload (reduces upload time)
3. **Clean uploads/** directory regularly to save disk space
4. **Monitor logs/** directory size
5. **Use production build** for frontend (smaller, faster)
6. **Enable caching** for static assets

## Security Considerations

- Files are temporarily stored in `uploads/` directory
- CORS is configured for localhost only (update for production)
- API keys are required in `.env` file (never commit to git)
- Implement authentication for production deployments
- Set up file size limits to prevent abuse
- Clean uploaded files periodically

## Roadmap

Future enhancements planned:

- [ ] WebSocket support for real-time progress
- [ ] Batch processing for multiple files
- [ ] Results history with search/filter
- [ ] User authentication and API keys
- [ ] Audio extraction from video
- [ ] Video/audio preview before processing
- [ ] Export in multiple formats (SRT, VTT, TXT)
- [ ] Dark/Light theme toggle
- [ ] Multi-language support

---

**For more information, see:**
- [API Documentation](API.md)
- [Development Guide](DEVELOPMENT.md)
- [Main README](../README.md)
