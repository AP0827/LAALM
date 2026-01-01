import { useState } from 'react';
import { FiPlay, FiDownload, FiCheckCircle, FiAlertCircle, FiLoader } from 'react-icons/fi';
import { BiMicrophone, BiVideo, BiBot } from 'react-icons/bi';
import axios from 'axios';
import './App.css';

const API_URL = 'http://localhost:8000';

function App() {
  const [videoFile, setVideoFile] = useState(null);
  const [audioFile, setAudioFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [progress, setProgress] = useState(0);

  const handleVideoChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setVideoFile(file);
      setError(null);
    }
  };

  const handleAudioChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setAudioFile(file);
      setError(null);
    }
  };

  const handleSubmit = async () => {
    if (!videoFile) {
      setError('Please upload a video file');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);
    setProgress(0);

    const formData = new FormData();
    formData.append('video', videoFile);
    if (audioFile) {
      formData.append('audio', audioFile);
    }

    try {
      const progressInterval = setInterval(() => {
        setProgress((prev) => {
          if (prev >= 90) {
            clearInterval(progressInterval);
            return 90;
          }
          return prev + 10;
        });
      }, 500);

      const response = await axios.post(`${API_URL}/transcribe`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      clearInterval(progressInterval);
      setProgress(100);
      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to process files');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setVideoFile(null);
    setAudioFile(null);
    setResult(null);
    setError(null);
    setProgress(0);
  };

  return (
    <div className="min-h-screen py-8 px-4 pt-32">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-6xl md:text-7xl font-bold text-white mb-4 tracking-tight">
            LAALM
          </h1>
          <p className="text-xl md:text-2xl text-gray-400 mb-2">
            Lip-reading Augmented Audio Language Model
          </p>
          <p className="text-gray-500 max-w-2xl mx-auto">
            Advanced multi-modal speech recognition combining audio analysis, visual lip-reading, 
            and AI-powered semantic correction
          </p>
        </div>

        {/* Main Transcription Card */}
        <div className="bg-white/5 backdrop-blur-xl rounded-3xl shadow-2xl border border-white/10 overflow-hidden mb-8">
          {!result ? (
            <div className="p-8 space-y-6">
              {/* Video Upload */}
              <div className="group">
                <label className="block text-sm font-medium text-gray-300 mb-3">
                  Video File *
                </label>
                <div className="relative">
                  <input
                    type="file"
                    accept="video/*"
                    onChange={handleVideoChange}
                    className="hidden"
                    id="video-upload"
                    disabled={loading}
                  />
                  <label
                    htmlFor="video-upload"
                    className="flex items-center justify-center w-full h-32 border-2 border-dashed border-gray-600 rounded-2xl cursor-pointer hover:border-secondary transition-all duration-300 hover:bg-white/5"
                  >
                    <div className="text-center">
                      <BiVideo className="mx-auto text-4xl text-gray-400 mb-2" />
                      <p className="text-sm text-gray-400">
                        {videoFile ? videoFile.name : 'Click to upload video'}
                      </p>
                    </div>
                  </label>
                </div>
              </div>

              {/* Audio Upload */}
              <div className="group">
                <label className="block text-sm font-medium text-gray-300 mb-3">
                  Audio File (Optional)
                </label>
                <div className="relative">
                  <input
                    type="file"
                    accept="audio/*"
                    onChange={handleAudioChange}
                    className="hidden"
                    id="audio-upload"
                    disabled={loading}
                  />
                  <label
                    htmlFor="audio-upload"
                    className="flex items-center justify-center w-full h-32 border-2 border-dashed border-gray-600 rounded-2xl cursor-pointer hover:border-accent transition-all duration-300 hover:bg-white/5"
                  >
                    <div className="text-center">
                      <BiMicrophone className="mx-auto text-4xl text-gray-400 mb-2" />
                      <p className="text-sm text-gray-400">
                        {audioFile ? audioFile.name : 'Click to upload audio'}
                      </p>
                    </div>
                  </label>
                </div>
              </div>

              {/* Error Message */}
              {error && (
                <div className="flex items-center gap-3 p-4 bg-red-500/10 border border-red-500/20 rounded-xl">
                  <FiAlertCircle className="text-red-400 text-xl flex-shrink-0" />
                  <p className="text-red-400 text-sm">{error}</p>
                </div>
              )}

              {/* Progress Bar */}
              {loading && (
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-sm text-gray-400">
                    <span>Processing...</span>
                    <span>{progress}%</span>
                  </div>
                  <div className="w-full h-2 bg-gray-700 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-secondary to-accent transition-all duration-300"
                      style={{ width: `${progress}%` }}
                    />
                  </div>
                </div>
              )}

              {/* Submit Button */}
              <button
                onClick={handleSubmit}
                disabled={loading || !videoFile}
                className="w-full py-4 bg-gradient-to-r from-secondary to-accent text-white rounded-xl font-medium text-lg hover:shadow-lg hover:shadow-secondary/50 transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-3"
              >
                {loading ? (
                  <>
                    <FiLoader className="animate-spin text-2xl" />
                    Processing...
                  </>
                ) : (
                  <>
                    <FiPlay className="text-2xl" />
                    Start Transcription
                  </>
                )}
              </button>
            </div>
          ) : (
            <div className="p-8 space-y-6">
              {/* Success Header */}
              <div className="flex items-center gap-3 p-4 bg-green-500/10 border border-green-500/20 rounded-xl">
                <FiCheckCircle className="text-green-400 text-2xl flex-shrink-0" />
                <div>
                  <h3 className="text-white font-medium">Transcription Complete</h3>
                  <p className="text-sm text-gray-400">
                    Processed in {result.processing_time.toFixed(2)}s
                  </p>
                </div>
              </div>

              {/* Audio Transcript */}
              <div className="space-y-3">
                <div className="flex items-center gap-2 text-orange-400">
                  <BiMicrophone className="text-xl" />
                  <h3 className="font-medium">Audio Transcript</h3>
                </div>
                <div className="bg-white/5 rounded-xl p-4 border border-white/10">
                  <p className="text-white text-lg mb-2">{result.audio_transcript}</p>
                  <div className="flex items-center gap-4 text-sm text-gray-400">
                    <span>Confidence: {(result.audio_confidence * 100).toFixed(1)}%</span>
                  </div>
                </div>
              </div>

              {/* Video Transcript */}
              <div className="space-y-3">
                <div className="flex items-center gap-2 text-blue-400">
                  <BiVideo className="text-xl" />
                  <h3 className="font-medium">Video Transcript</h3>
                </div>
                <div className="bg-white/5 rounded-xl p-4 border border-white/10">
                  <p className="text-white text-lg mb-2">{result.video_transcript}</p>
                  <div className="flex items-center gap-4 text-sm text-gray-400">
                    <span>Confidence: {(result.video_confidence * 100).toFixed(1)}%</span>
                  </div>
                </div>
              </div>

              {/* Final Transcript */}
              <div className="space-y-3">
                <div className="flex items-center gap-2 text-purple-400">
                  <BiBot className="text-xl" />
                  <h3 className="font-medium">Final Corrected Transcript</h3>
                </div>
                <div className="bg-gradient-to-r from-secondary/10 to-accent/10 rounded-xl p-4 border border-purple-500/20">
                  <p className="text-white text-xl font-medium mb-2">
                    {result.final_transcript}
                  </p>
                  <div className="flex flex-wrap items-center gap-4 text-sm text-gray-300">
                    <span>Confidence: {(result.final_confidence * 100).toFixed(1)}%</span>
                    <span>•</span>
                    <span>Agreement: {(result.agreement_rate * 100).toFixed(1)}%</span>
                    <span>•</span>
                    <span>{result.corrections_applied} corrections applied</span>
                  </div>
                </div>
              </div>

              {/* Action Buttons */}
              <div className="flex gap-3 pt-4">
                <button
                  onClick={handleReset}
                  className="flex-1 py-3 bg-white/5 border border-white/10 text-white rounded-xl font-medium hover:bg-white/10 transition-all duration-300"
                >
                  New Transcription
                </button>
                <button
                  onClick={() => {
                    const blob = new Blob([JSON.stringify(result, null, 2)], {
                      type: 'application/json',
                    });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `laalm-result-${Date.now()}.json`;
                    a.click();
                  }}
                  className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-secondary to-accent text-white rounded-xl font-medium hover:shadow-lg hover:shadow-secondary/50 transition-all duration-300"
                >
                  <FiDownload className="text-xl" />
                  Download Results
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Stats */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
          <div className="bg-white/5 backdrop-blur-xl rounded-2xl p-6 border border-white/10 text-center">
            <div className="text-4xl font-bold text-secondary mb-2">98.7%</div>
            <div className="text-gray-400 text-sm">Audio Accuracy</div>
          </div>
          <div className="bg-white/5 backdrop-blur-xl rounded-2xl p-6 border border-white/10 text-center">
            <div className="text-4xl font-bold text-accent mb-2">80%</div>
            <div className="text-gray-400 text-sm">Visual Accuracy</div>
          </div>
          <div className="bg-white/5 backdrop-blur-xl rounded-2xl p-6 border border-white/10 text-center">
            <div className="text-4xl font-bold bg-gradient-to-r from-secondary to-accent bg-clip-text text-transparent mb-2">
              99%
            </div>
            <div className="text-gray-400 text-sm">Combined Accuracy</div>
          </div>
        </div>

        {/* How It Works */}
        <div className="bg-white/5 backdrop-blur-xl rounded-2xl p-8 border border-white/10 mb-12">
          <h2 className="text-3xl font-bold text-white text-center mb-8">
            How It Works
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            {[
              { icon: <BiVideo />, title: 'Upload Video', desc: 'Upload your video and optional audio', color: 'text-blue-400' },
              { icon: <BiMicrophone />, title: 'Audio Analysis', desc: 'DeepGram processes with 98.7% accuracy', color: 'text-orange-400' },
              { icon: <BiVideo />, title: 'Lip Reading', desc: 'auto_avsr analyzes visual speech', color: 'text-blue-400' },
              { icon: <BiBot />, title: 'AI Correction', desc: 'Groq LLM validates semantically', color: 'text-purple-400' },
            ].map((step, idx) => (
              <div key={idx} className="text-center">
                <div className={`text-5xl mb-4 ${step.color} flex justify-center`}>{step.icon}</div>
                <h3 className="text-lg font-bold text-white mb-2">{step.title}</h3>
                <p className="text-gray-400 text-sm">{step.desc}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Footer */}
        <footer className="text-center text-gray-500 text-sm py-8">
          <p className="mb-2">Powered by DeepGram, auto_avsr, and Groq LLM</p>
          <p>© 2025 LAALM. Built with React, FastAPI, and cutting-edge AI</p>
        </footer>
      </div>
    </div>
  );
}

export default App;
