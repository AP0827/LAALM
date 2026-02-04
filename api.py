"""
LAALM FastAPI Backend Server
Provides REST API endpoints for audio/video transcription with multi-modal fusion
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import tempfile
import shutil
from pathlib import Path
import json
from datetime import datetime

# Import LAALM pipeline
from pipeline import run_mvp
from logger import get_logger

# Initialize FastAPI app
app = FastAPI(
    title="LAALM API",
    description="Audio-Visual Speech Recognition with LLM Correction",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.staticfiles import StaticFiles

# Create uploads and captions directories
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
CAPTIONS_DIR = Path("captions")
CAPTIONS_DIR.mkdir(exist_ok=True)

# Mount captions directory
app.mount("/captions", StaticFiles(directory="captions"), name="captions")

# Response models
class TranscriptionResult(BaseModel):
    audio_transcript: str
    audio_confidence: float
    video_transcript: str
    video_confidence: float
    final_transcript: str
    final_confidence: float
    agreement_rate: float
    word_details: List[Dict[str, Any]]
    corrections_applied: int
    processing_time: float
    timestamp: str
    captions: Optional[Dict[str, str]] = None

class HealthResponse(BaseModel):
    status: str
    message: str
    models_loaded: bool

class ErrorResponse(BaseModel):
    error: str
    detail: str


@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "message": "LAALM API is running",
        "models_loaded": True
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check"""
    try:
        # Check if models exist
        vsr_model_path = Path("auto_avsr/pretrained_models/vsr_trlrs2lrs3vox2avsp_base.pth")
        retinaface_model_path = Path("auto_avsr/preparation/detectors/retinaface/weights/Resnet50_Final.pth")
        
        models_exist = vsr_model_path.exists() and retinaface_model_path.exists()
        
        return {
            "status": "healthy" if models_exist else "degraded",
            "message": "All systems operational" if models_exist else "Some models missing",
            "models_loaded": models_exist
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "models_loaded": False
        }


@app.post("/transcribe", response_model=TranscriptionResult)
async def transcribe(
    video: UploadFile = File(...),
    audio: Optional[UploadFile] = File(None),
    use_advanced_preprocessing: bool = True,

    video_denoise_strength: int = 0,
    audio_denoise_strength: int = 0,
    use_temporal_smoothing: bool = False
):
    """
    Transcribe audio/video files using LAALM pipeline
    
    Args:
        video: Video file (required)
        audio: Audio file (optional)
        use_advanced_preprocessing: Enable advanced video enhancement (default: True)
        video_denoise_strength: Video denoising 0-10
        audio_denoise_strength: Audio denoising 0-10
        use_temporal_smoothing: Enable temporal smoothing (default: False - can blur lips)
    
    Returns:
        TranscriptionResult with all modality outputs
    """
    video_path = None
    audio_path = None
    
    try:
        start_time = datetime.now()
        
        # Save uploaded video file
        video_path = UPLOAD_DIR / f"video_{start_time.strftime('%Y%m%d_%H%M%S')}_{video.filename}"
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
        
        # Save audio file if provided
        cleanup_audio = False
        if audio:
            audio_path = UPLOAD_DIR / f"audio_{start_time.strftime('%Y%m%d_%H%M%S')}_{audio.filename}"
            with open(audio_path, "wb") as buffer:
                shutil.copyfileobj(audio.file, buffer)
        else:
            cleanup_audio = True
            # Extract audio from video using ffmpeg
            import subprocess
            
            # Create a path based on the video filename but with .wav extension
            audio_filename = f"extracted_{start_time.strftime('%Y%m%d_%H%M%S')}_{Path(video.filename).stem}.wav"
            audio_path = UPLOAD_DIR / audio_filename
            
            print(f"🔊 Extracting audio from {video_path} to {audio_path}...")
            
            # Build audio filters
            audio_filters = []
            if use_advanced_preprocessing:
                # 1. Bandpass Filter (Safe Speech Range: 200Hz - 3000Hz)
                # Removes low rumble (wind/machinery) and high hiss
                audio_filters.append("highpass=f=200,lowpass=f=3000")
                
                # 2. Noise Reduction (if strength > 2)
                if audio_denoise_strength > 2:
                    # Map 3-10 to 5-30dB noise reduction
                    # nr: noise reduction level in dB
                    nr_level = min(30, (audio_denoise_strength - 1) * 3)
                    audio_filters.append(f"afftdn=nr={nr_level}")
                    print(f"   ✨ Applying Audio Denoising (Level {nr_level}dB)")

            try:
                # ffmpeg -i input_video.mp4 -vn -acodec pcm_s16le -ar 16000 -ac 1 output_audio.wav
                command = [
                    "ffmpeg",
                    "-y", # Overwrite output if exists
                    "-i", str(video_path),
                    "-vn", # No video
                    "-acodec", "pcm_s16le", # PWM 16-bit
                    "-ar", "16000", # 16kHz
                    "-ac", "1", # Mono
                ]
                
                # Apply filters if we have any
                if audio_filters:
                    command.extend(["-af", ",".join(audio_filters)])
                
                command.append(str(audio_path))
                
                # Run ffmpeg, suppressing output unless error
                subprocess.run(
                    command, 
                    check=True, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE
                )
                print("   ✓ Audio extraction successful")
                
            except subprocess.CalledProcessError as e:
                print(f"   ❌ Audio extraction failed: {e}")
                # Fallback: Let pipeline handle missing audio (will use mock or fail gracefully)
                if audio_path.exists():
                    audio_path.unlink()
                audio_path = None
        
        # Run LAALM pipeline
        result = run_mvp(
            video_file=str(video_path),
            audio_file=str(audio_path) if audio_path else None,
            use_advanced_preprocessing=use_advanced_preprocessing,
            video_denoise_strength=video_denoise_strength,
            use_temporal_smoothing=use_temporal_smoothing
        )

        # Cleanup temporary audio file
        if cleanup_audio and audio_path and audio_path.exists():
            print(f"🧹 Cleaning up temporary audio: {audio_path}")
            try:
                audio_path.unlink()
            except Exception as e:
                print(f"   ⚠ Failed to delete temp audio: {e}")
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Verify result structure and sanitize types
        import numpy as np
        
        def sanitize_for_json(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: sanitize_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [sanitize_for_json(i) for i in obj]
            return obj

        # Sanitize the entire result dictionary
        result = sanitize_for_json(result)
        
        # Extract results from pipeline output
        deepgram_result = result['deepgram']
        avsr_result = result['avsr']
        groq_result = result['groq']
        word_confidences = result['combined_words']
        
        # Calculate agreement metrics
        total_words = len(word_confidences)
        agreed_words = sum(1 for w in word_confidences if w.get('agreed', False))
        agreement_rate = agreed_words / total_words if total_words > 0 else 0.0
        
        # Count corrections
        corrections_applied = len(groq_result.get('corrections', []))
        
        # Process captions URLs
        captions_urls = {}
        if 'captions' in result:
            for fmt, path in result['captions'].items():
                # Convert absolute path to relative URL
                filename = Path(path).name
                captions_urls[fmt] = f"/captions/{filename}"
        
        # Build response
        response = TranscriptionResult(
            audio_transcript=deepgram_result['transcript'],
            audio_confidence=deepgram_result['overall_confidence'],
            video_transcript=avsr_result['transcript'],
            video_confidence=avsr_result['overall_confidence'],
            final_transcript=result['final_transcript'],
            final_confidence=groq_result['confidence'],
            agreement_rate=agreement_rate,
            word_details=word_confidences,
            corrections_applied=corrections_applied,
            processing_time=processing_time,
            timestamp=start_time.isoformat(),
            captions=captions_urls
        )
        
        return response
        
    except Exception as e:
        # Clean up files on error
        if video_path and video_path.exists():
            video_path.unlink()
        if audio_path and audio_path.exists():
            audio_path.unlink()
        
        raise HTTPException(
            status_code=500,
            detail=f"Transcription failed: {str(e)}"
        )


@app.post("/transcribe-separate")
async def transcribe_separate(
    video: UploadFile = File(...),
    audio: UploadFile = File(...),
    use_advanced_preprocessing: bool = True,
    denoise_strength: int = 3,
    use_temporal_smoothing: bool = False
):
    """
    Transcribe with separate audio and video files
    Alias for /transcribe with required audio parameter
    """
    return await transcribe(
        video=video, 
        audio=audio,
        use_advanced_preprocessing=use_advanced_preprocessing,
        denoise_strength=denoise_strength,
        use_temporal_smoothing=use_temporal_smoothing
    )


@app.get("/logs")
async def get_logs(limit: int = 10):
    """
    Get recent processing logs
    
    Args:
        limit: Maximum number of log entries to return
    
    Returns:
        List of recent transcription results from logs
    """
    try:
        logs_dir = Path("logs")
        if not logs_dir.exists():
            return {"logs": [], "count": 0}
        
        # Get most recent JSON result files
        json_files = sorted(
            logs_dir.glob("results_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )[:limit]
        
        logs = []
        for json_file in json_files:
            with open(json_file, 'r') as f:
                logs.append(json.load(f))
        
        return {
            "logs": logs,
            "count": len(logs)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve logs: {str(e)}"
        )


@app.delete("/uploads/{filename}")
async def delete_upload(filename: str):
    """
    Delete an uploaded file
    
    Args:
        filename: Name of file to delete
    
    Returns:
        Success message
    """
    try:
        file_path = UPLOAD_DIR / filename
        if file_path.exists():
            file_path.unlink()
            return {"message": f"Deleted {filename}"}
        else:
            raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete file: {str(e)}"
        )


@app.get("/stats")
async def get_stats():
    """
    Get system statistics
    
    Returns:
        Statistics about usage and performance
    """
    try:
        logs_dir = Path("logs")
        uploads_dir = Path("uploads")
        
        # Count files
        total_logs = len(list(logs_dir.glob("*.log"))) if logs_dir.exists() else 0
        total_results = len(list(logs_dir.glob("results_*.json"))) if logs_dir.exists() else 0
        total_uploads = len(list(uploads_dir.glob("*"))) if uploads_dir.exists() else 0
        
        # Calculate disk usage
        logs_size = sum(f.stat().st_size for f in logs_dir.glob("*") if f.is_file()) if logs_dir.exists() else 0
        uploads_size = sum(f.stat().st_size for f in uploads_dir.glob("*") if f.is_file()) if uploads_dir.exists() else 0
        
        return {
            "total_logs": total_logs,
            "total_results": total_results,
            "total_uploads": total_uploads,
            "logs_size_mb": round(logs_size / 1024 / 1024, 2),
            "uploads_size_mb": round(uploads_size / 1024 / 1024, 2),
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get stats: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting LAALM API Server...")
    print("📍 API will be available at: http://localhost:8000")
    print("📚 API docs available at: http://localhost:8000/docs")
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
