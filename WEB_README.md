# LAALM Web Interface

A beautiful, minimal web interface for the LAALM (Lip-reading Augmented Audio Language Model) transcription system.

## ✨ Features

- **Elegant UI**: Clean, modern interface with glassmorphism design
- **Drag & Drop**: Easy file upload for video and audio files
- **Real-time Progress**: Visual feedback during transcription processing
- **Multi-Modal Results**: Display audio, video, and LLM-corrected transcripts
- **Confidence Scores**: View confidence metrics for each modality
- **Download Results**: Export transcription results as JSON
- **Responsive Design**: Works seamlessly on desktop and mobile devices

## 🚀 Quick Start

### Prerequisites

- Python 3.11+ with LAALM dependencies installed
- Node.js 20.x or higher
- LAALM models downloaded (see main README.md)
- API keys configured in `.env` file

### Start the Application

```bash
# Option 1: Use the startup script (recommended)
./start_web.sh

# Option 2: Start manually
# Terminal 1 - Backend
source .venv/bin/activate
python api.py

# Terminal 2 - Frontend
cd frontend
npm run dev
```

The application will be available at:
- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## 📁 Project Structure

```
LAALM/
├── api.py                 # FastAPI backend server
├── start_web.sh          # Startup script for both servers
├── frontend/             # Vite + React frontend
│   ├── src/
│   │   ├── App.jsx       # Main application component
│   │   ├── App.css       # Component styles
│   │   └── index.css     # Global styles with Tailwind
│   ├── package.json      # Frontend dependencies
│   ├── vite.config.js    # Vite configuration
│   └── tailwind.config.js # Tailwind CSS configuration
├── uploads/              # Temporary file storage
└── logs/                 # Transcription logs
```

## 🎨 Design Features

### Color Scheme
- **Primary**: Dark gradient background (`#0A0E27` → `#1a1f3a`)
- **Secondary**: Indigo (`#6366F1`) for primary actions
- **Accent**: Purple (`#8B5CF6`) for highlights
- **Glassmorphism**: White overlays with backdrop blur

### Component Styling
- **Rounded corners**: `rounded-xl` / `rounded-2xl` / `rounded-3xl`
- **Smooth transitions**: 300ms duration for all interactions
- **Hover effects**: Border color changes and background tints
- **Loading states**: Animated spinner and progress bar

### Icons
- `react-icons` library for consistent iconography
- Color-coded by modality:
  - 🎤 Orange for audio
  - 📹 Blue for video
  - 🤖 Purple for LLM corrections

## 🔌 API Endpoints

### `GET /`
Health check endpoint
```json
{
  "status": "online",
  "message": "LAALM API is running",
  "models_loaded": true
}
```

### `POST /transcribe`
Main transcription endpoint

**Request:**
- `video`: Video file (required, multipart/form-data)
- `audio`: Audio file (optional, multipart/form-data)

**Response:**
```json
{
  "audio_transcript": "string",
  "audio_confidence": 0.987,
  "video_transcript": "string",
  "video_confidence": 0.506,
  "final_transcript": "string",
  "final_confidence": 0.987,
  "agreement_rate": 0.333,
  "word_details": [...],
  "corrections_applied": 5,
  "processing_time": 12.34,
  "timestamp": "2025-12-29T18:00:00"
}
```

### `GET /logs?limit=10`
Retrieve recent transcription logs

### `GET /stats`
Get system statistics (file counts, disk usage)

### `DELETE /uploads/{filename}`
Delete uploaded file

## 🛠️ Development

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
```

### Backend Development

```bash
source .venv/bin/activate

# Run with auto-reload
python api.py

# Or use uvicorn directly
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

### Adding New Features

1. **Frontend**: Edit `frontend/src/App.jsx`
2. **Backend**: Edit `api.py` and add new endpoints
3. **Styling**: Modify `frontend/tailwind.config.js` for theme changes
4. **State Management**: Extend `useState` hooks in App.jsx

## 📦 Dependencies

### Backend
- `fastapi` - Modern web framework
- `uvicorn` - ASGI server
- `python-multipart` - File upload handling
- `pydantic` - Data validation

### Frontend
- `react` - UI library
- `vite` - Build tool and dev server
- `tailwindcss` - Utility-first CSS framework
- `axios` - HTTP client
- `react-icons` - Icon library

## 🔒 Security Notes

- Files are temporarily stored in `uploads/` directory
- CORS is configured for `localhost:5173` and `localhost:3000`
- Uploaded files should be cleaned periodically
- API keys are required in `.env` file

## 🐛 Troubleshooting

### Backend won't start
```bash
# Check Python environment
source .venv/bin/activate
python --version  # Should be 3.11+

# Install missing dependencies
pip install fastapi uvicorn python-multipart
```

### Frontend won't start
```bash
# Check Node version
node --version  # Should be 20.x+

# Reinstall dependencies
cd frontend
rm -rf node_modules package-lock.json
npm install
```

### CORS errors
- Ensure backend is running on port 8000
- Check `api.py` CORS middleware configuration
- Verify frontend is accessing `http://localhost:8000`

### Model not found errors
```bash
# Check model files exist
ls -lh auto_avsr/pretrained_models/*.pth
ls -lh auto_avsr/preparation/detectors/retinaface/weights/*.pth

# Download models if missing (see main README.md)
```

## 📊 Performance Tips

- Use separate audio file for faster processing
- Compress videos before upload for quicker uploads
- Clean `uploads/` directory regularly to save disk space
- Monitor `logs/` directory size

## 🎯 Roadmap

- [ ] WebSocket support for real-time progress updates
- [ ] Batch processing for multiple files
- [ ] Results history with search/filter
- [ ] User authentication and API keys
- [ ] Audio extraction from video
- [ ] Preview video/audio before processing
- [ ] Export results in multiple formats (SRT, VTT, TXT)
- [ ] Dark/Light theme toggle

## 📄 License

Same as main LAALM project.

## 🤝 Contributing

Contributions welcome! Please ensure:
1. Code follows existing style conventions
2. Test both frontend and backend changes
3. Update documentation for new features
4. Run linters before committing

## 💬 Support

For issues or questions:
1. Check the troubleshooting section
2. Review API documentation at http://localhost:8000/docs
3. Check main LAALM README.md
4. Open an issue on GitHub

---

**Enjoy using LAALM Web Interface! 🎉**
