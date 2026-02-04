# API Documentation

Complete REST API reference for the LAALM transcription system.

## Table of Contents

- [Overview](#overview)
- [Base URL](#base-url)
- [Authentication](#authentication)
- [Endpoints](#endpoints)
- [Data Models](#data-models)
- [Error Handling](#error-handling)
- [Rate Limiting](#rate-limiting)
- [Examples](#examples)

## Overview

The LAALM API provides programmatic access to multi-modal speech transcription capabilities. Built with FastAPI, it offers:

- RESTful endpoints
- JSON request/response format
- Automatic OpenAPI documentation
- File upload support
- Real-time processing

## Base URL

**Development:**
```
http://localhost:8000
```

**Production:**
```
https://your-domain.com/api
```

## Authentication

Currently, the API uses API keys configured in the `.env` file for external services (Groq, DeepGram). No authentication is required for the LAALM API itself in development mode.

**For production deployments**, implement authentication:
- API key authentication
- JWT tokens
- OAuth 2.0

## Endpoints

### Health Check

Check if the API is running and models are loaded.

**Endpoint:** `GET /`

**Response:**
```json
{
  "status": "online",
  "message": "LAALM API is running",
  "models_loaded": true,
  "version": "1.0.0"
}
```

**Status Codes:**
- `200 OK` - API is running
- `503 Service Unavailable` - Models not loaded

---

### Transcribe Video

Process video and optional audio files to generate transcriptions.

**Endpoint:** `POST /transcribe`

**Content-Type:** `multipart/form-data`

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| video | File | Yes | Video file (.mpg, .mp4, .avi) |
| audio | File | No | Audio file (.wav, .mp3) |

**Request Example (curl):**
```bash
curl -X POST http://localhost:8000/transcribe \
  -F "video=@samples/video/lwwz9s.mpg" \
  -F "audio=@samples/audio/lwwz9s.wav"
```

**Request Example (Python):**
```python
import requests

files = {
    'video': open('samples/video/lwwz9s.mpg', 'rb'),
    'audio': open('samples/audio/lwwz9s.wav', 'rb')
}

response = requests.post('http://localhost:8000/transcribe', files=files)
result = response.json()
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
      "source": "audio",
      "start_time": 0.0,
      "end_time": 0.3
    },
    {
      "word": "white",
      "audio_conf": 0.98,
      "video_conf": 0.52,
      "source": "audio",
      "start_time": 0.3,
      "end_time": 0.6
    }
  ],
  "corrections_applied": 5,
  "processing_time": 12.34,
  "timestamp": "2026-02-04T17:00:00.000Z"
}
```

**Status Codes:**
- `200 OK` - Transcription successful
- `400 Bad Request` - Invalid file format or missing video
- `500 Internal Server Error` - Processing error

---

### Get Logs

Retrieve recent transcription logs.

**Endpoint:** `GET /logs`

**Query Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| limit | integer | No | 10 | Number of logs to return |

**Request Example:**
```bash
curl http://localhost:8000/logs?limit=5
```

**Response:**
```json
{
  "logs": [
    {
      "timestamp": "2026-02-04T17:00:00",
      "video_file": "lwwz9s.mpg",
      "audio_file": "lwwz9s.wav",
      "final_transcript": "lay white with zero again",
      "confidence": 0.987,
      "processing_time": 12.34
    }
  ],
  "total": 156
}
```

**Status Codes:**
- `200 OK` - Logs retrieved successfully
- `404 Not Found` - No logs available

---

### Get Statistics

Get system statistics and usage metrics.

**Endpoint:** `GET /stats`

**Response:**
```json
{
  "total_uploads": 42,
  "total_logs": 156,
  "total_transcriptions": 138,
  "disk_usage_mb": 234.5,
  "uptime_seconds": 3600,
  "average_processing_time": 11.2,
  "success_rate": 0.95
}
```

**Status Codes:**
- `200 OK` - Statistics retrieved successfully

---

### Delete Upload

Delete an uploaded file from the server.

**Endpoint:** `DELETE /uploads/{filename}`

**Path Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| filename | string | Yes | Name of file to delete |

**Request Example:**
```bash
curl -X DELETE http://localhost:8000/uploads/video_20260204_170000.mpg
```

**Response:**
```json
{
  "message": "File deleted successfully",
  "filename": "video_20260204_170000.mpg"
}
```

**Status Codes:**
- `200 OK` - File deleted successfully
- `404 Not Found` - File not found
- `403 Forbidden` - Permission denied

---

### Export Captions

Export transcription results in various formats.

**Endpoint:** `GET /export/{format}`

**Path Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| format | string | Yes | Export format: `srt`, `vtt`, `txt`, `json` |

**Query Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| transcript_id | string | Yes | ID of transcription to export |

**Request Example:**
```bash
curl http://localhost:8000/export/srt?transcript_id=20260204_170000
```

**Response (SRT):**
```
1
00:00:00,000 --> 00:00:03,000
lay white with zero again
```

**Status Codes:**
- `200 OK` - Export successful
- `404 Not Found` - Transcript not found
- `400 Bad Request` - Invalid format

## Data Models

### TranscriptionResult

```python
{
  "audio_transcript": str,        # Audio-only transcription
  "audio_confidence": float,      # Audio confidence (0-1)
  "video_transcript": str,        # Video-only transcription
  "video_confidence": float,      # Video confidence (0-1)
  "final_transcript": str,        # LLM-corrected final result
  "final_confidence": float,      # Final confidence (0-1)
  "agreement_rate": float,        # Audio-video agreement (0-1)
  "word_details": [WordDetail],   # Per-word details
  "corrections_applied": int,     # Number of LLM corrections
  "processing_time": float,       # Processing time in seconds
  "timestamp": str               # ISO 8601 timestamp
}
```

### WordDetail

```python
{
  "word": str,                   # The word
  "audio_conf": float,           # Audio confidence (0-1)
  "video_conf": float,           # Video confidence (0-1)
  "source": str,                 # "audio", "video", or "llm"
  "start_time": float,           # Start time in seconds
  "end_time": float             # End time in seconds
}
```

### LogEntry

```python
{
  "timestamp": str,              # ISO 8601 timestamp
  "video_file": str,             # Video filename
  "audio_file": str | null,      # Audio filename or null
  "final_transcript": str,       # Final transcription
  "confidence": float,           # Overall confidence
  "processing_time": float      # Processing time in seconds
}
```

## Error Handling

### Error Response Format

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable error message",
    "details": {
      "field": "Additional context"
    }
  }
}
```

### Common Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `INVALID_FILE_FORMAT` | 400 | Unsupported file format |
| `FILE_TOO_LARGE` | 400 | File exceeds size limit |
| `MISSING_VIDEO` | 400 | Video file is required |
| `PROCESSING_ERROR` | 500 | Error during transcription |
| `MODEL_NOT_LOADED` | 503 | Models not initialized |
| `API_KEY_INVALID` | 401 | Invalid API key |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests |

### Example Error Response

```json
{
  "error": {
    "code": "INVALID_FILE_FORMAT",
    "message": "Video file must be in .mpg, .mp4, or .avi format",
    "details": {
      "received_format": ".txt",
      "supported_formats": [".mpg", ".mp4", ".avi"]
    }
  }
}
```

## Rate Limiting

**Current Limits:**
- No rate limiting in development mode
- External API limits apply (Groq, DeepGram)

**Recommended Production Limits:**
- 100 requests per hour per IP
- 10 concurrent uploads per user
- Max file size: 100 MB

**Rate Limit Headers:**
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1643723400
```

## Examples

### Python Client

```python
import requests
from pathlib import Path

class LAALMClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def transcribe(self, video_path, audio_path=None):
        """Transcribe video with optional audio."""
        files = {'video': open(video_path, 'rb')}
        if audio_path:
            files['audio'] = open(audio_path, 'rb')
        
        response = requests.post(
            f"{self.base_url}/transcribe",
            files=files
        )
        response.raise_for_status()
        return response.json()
    
    def get_logs(self, limit=10):
        """Get recent transcription logs."""
        response = requests.get(
            f"{self.base_url}/logs",
            params={'limit': limit}
        )
        response.raise_for_status()
        return response.json()
    
    def get_stats(self):
        """Get system statistics."""
        response = requests.get(f"{self.base_url}/stats")
        response.raise_for_status()
        return response.json()

# Usage
client = LAALMClient()
result = client.transcribe(
    "samples/video/lwwz9s.mpg",
    "samples/audio/lwwz9s.wav"
)
print(f"Transcript: {result['final_transcript']}")
print(f"Confidence: {result['final_confidence']:.2%}")
```

### JavaScript Client

```javascript
class LAALMClient {
  constructor(baseURL = 'http://localhost:8000') {
    this.baseURL = baseURL;
  }

  async transcribe(videoFile, audioFile = null) {
    const formData = new FormData();
    formData.append('video', videoFile);
    if (audioFile) {
      formData.append('audio', audioFile);
    }

    const response = await fetch(`${this.baseURL}/transcribe`, {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  }

  async getLogs(limit = 10) {
    const response = await fetch(
      `${this.baseURL}/logs?limit=${limit}`
    );
    return await response.json();
  }

  async getStats() {
    const response = await fetch(`${this.baseURL}/stats`);
    return await response.json();
  }
}

// Usage
const client = new LAALMClient();
const videoFile = document.getElementById('video-input').files[0];
const audioFile = document.getElementById('audio-input').files[0];

const result = await client.transcribe(videoFile, audioFile);
console.log('Transcript:', result.final_transcript);
console.log('Confidence:', result.final_confidence);
```

### Batch Processing

```python
import asyncio
import aiohttp
from pathlib import Path

async def transcribe_batch(video_files, audio_files=None):
    """Process multiple files in parallel."""
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i, video in enumerate(video_files):
            audio = audio_files[i] if audio_files else None
            task = transcribe_file(session, video, audio)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        return results

async def transcribe_file(session, video_path, audio_path=None):
    """Transcribe a single file."""
    data = aiohttp.FormData()
    data.add_field('video', open(video_path, 'rb'))
    if audio_path:
        data.add_field('audio', open(audio_path, 'rb'))
    
    async with session.post(
        'http://localhost:8000/transcribe',
        data=data
    ) as response:
        return await response.json()

# Usage
video_files = list(Path('samples/video').glob('*.mpg'))
audio_files = list(Path('samples/audio').glob('*.wav'))

results = asyncio.run(transcribe_batch(video_files, audio_files))
for result in results:
    print(f"Transcript: {result['final_transcript']}")
```

## Interactive API Documentation

FastAPI provides automatic interactive API documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

These interfaces allow you to:
- Explore all endpoints
- Test API calls directly in the browser
- View request/response schemas
- Download OpenAPI specification

---

**For more information, see:**
- [Web Interface Documentation](WEB_INTERFACE.md)
- [Development Guide](DEVELOPMENT.md)
- [Main README](../README.md)
