import React, { useState } from 'react';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';
import { FiLoader, FiCheck, FiX } from 'react-icons/fi';

import Hero from './components/Hero';
import UploadZone from './components/UploadZone';
import SettingsPanel from './components/SettingsPanel';
import ResultsDashboard from './components/ResultsDashboard';

const App = () => {
  const [status, setStatus] = useState('idle'); // idle, uploading, processing, success, error
  const [errorMsg, setErrorMsg] = useState('');

  const [selectedVideo, setSelectedVideo] = useState(null);
  const [selectedAudio, setSelectedAudio] = useState(null);

  const [settings, setSettings] = useState({
    use_advanced_preprocessing: true,
    denoise_strength: 3,
    use_temporal_smoothing: false
  });

  const [result, setResult] = useState(null);

  const handleFileSelect = (type, file) => {
    if (type === 'video') setSelectedVideo(file);
    if (type === 'audio') setSelectedAudio(file);
    setErrorMsg('');
  };

  const handleTranscribe = async () => {
    if (!selectedVideo) {
      setErrorMsg("Please upload a video file to continue.");
      return;
    }

    setStatus('uploading');

    const formData = new FormData();
    formData.append('video', selectedVideo);
    if (selectedAudio) formData.append('audio', selectedAudio);

    // Append settings
    formData.append('use_advanced_preprocessing', settings.use_advanced_preprocessing);
    formData.append('denoise_strength', settings.denoise_strength);
    formData.append('use_temporal_smoothing', settings.use_temporal_smoothing);

    try {
      setStatus('processing');
      // Adding a minimum delay to show the nice animation :)
      const startTime = Date.now();

      const response = await axios.post('http://localhost:8000/transcribe', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      const elapsed = Date.now() - startTime;
      if (elapsed < 2000) await new Promise(r => setTimeout(r, 2000 - elapsed));

      setResult(response.data);
      setStatus('success');
    } catch (err) {
      console.error(err);
      setStatus('error');
      setErrorMsg(err.response?.data?.detail || "Transcription failed. Please ensure the backend is running.");
    }
  };

  const reset = () => {
    setStatus('idle');
    setResult(null);
    setSelectedVideo(null);
    setSelectedAudio(null);
    setErrorMsg('');
  };

  return (
    <div className="min-h-screen pb-20 overflow-x-hidden">

      {/* Background Ambience */}
      <div className="fixed inset-0 pointer-events-none z-[-1]">
        <div className="absolute top-0 right-0 w-1/3 h-1/3 bg-indigo-900/10 blur-[120px]" />
        <div className="absolute bottom-0 left-0 w-1/3 h-1/3 bg-purple-900/10 blur-[120px]" />
      </div>

      <Hero />

      <motion.main
        className="px-4 z-10 relative"
        layout
      >
        <AnimatePresence mode="wait">

          {/* IDLE STATE: Upload & Settings */}
          {status === 'idle' && (
            <motion.div
              key="idle"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95 }}
              transition={{ duration: 0.5 }}
            >
              <UploadZone
                onFileSelect={handleFileSelect}
                selectedVideo={selectedVideo}
                selectedAudio={selectedAudio}
              />

              <SettingsPanel
                settings={settings}
                setSettings={setSettings}
              />

              <div className="text-center mt-12">
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={handleTranscribe}
                  disabled={!selectedVideo}
                  className={`
                    px-12 py-4 rounded-full font-bold text-lg shadow-2xl transition-all duration-300
                    ${selectedVideo
                      ? 'bg-gradient-to-r from-indigo-500 to-purple-600 text-white shadow-indigo-500/30'
                      : 'bg-gray-800 text-gray-500 cursor-not-allowed'}
                  `}
                >
                  Start Transcription
                </motion.button>
              </div>
            </motion.div>
          )}

          {/* LOADING STATE */}
          {(status === 'uploading' || status === 'processing') && (
            <motion.div
              key="loading"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              className="flex flex-col items-center justify-center py-20"
            >
              <div className="relative w-32 h-32 mb-8">
                <div className="absolute inset-0 border-4 border-indigo-500/30 rounded-full" />
                <div className="absolute inset-0 border-4 border-transparent border-t-indigo-500 border-r-purple-500 rounded-full animate-spin" />
                <div className="absolute inset-0 flex items-center justify-center">
                  <FiLoader className="text-white/50 animate-pulse" size={32} />
                </div>
              </div>
              <h2 className="text-2xl font-bold text-white mb-2">
                {status === 'uploading' ? 'Uploading Media...' : 'Analyzing Lip Movements...'}
              </h2>
              <p className="text-gray-400">
                {status === 'processing' && "Running VSR, aligning modalities, and applying LLM corrections."}
              </p>
            </motion.div>
          )}

          {/* RESULTS STATE */}
          {status === 'success' && (
            <motion.div
              key="results"
              initial={{ opacity: 0, y: 50 }}
              animate={{ opacity: 1, y: 0 }}
            >
              <div className="text-center mb-8">
                <button
                  onClick={reset}
                  className="text-gray-400 hover:text-white underline text-sm"
                >
                  ← Upload New File
                </button>
              </div>
              <ResultsDashboard result={result} />
            </motion.div>
          )}

          {/* ERROR STATE */}
          {status === 'error' && (
            <motion.div
              key="error"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="max-w-md mx-auto glass-card border-red-500/50 p-6 rounded-xl text-center"
            >
              <div className="w-16 h-16 bg-red-500/20 rounded-full flex items-center justify-center mx-auto mb-4 text-red-500">
                <FiX size={32} />
              </div>
              <h3 className="text-xl font-bold text-white mb-2">Processing Failed</h3>
              <p className="text-gray-300 mb-6">{errorMsg}</p>
              <button
                onClick={reset}
                className="px-6 py-2 bg-white/10 hover:bg-white/20 rounded-lg text-white"
              >
                Try Again
              </button>
            </motion.div>
          )}

        </AnimatePresence>
      </motion.main>

      {/* Toast Notification */}
      <AnimatePresence>
        {errorMsg && status === 'idle' && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
            className="fixed bottom-8 left-1/2 -translate-x-1/2 bg-red-500/90 text-white px-6 py-3 rounded-full shadow-2xl flex items-center gap-2 backdrop-blur-md z-50"
          >
            <FiX /> {errorMsg}
          </motion.div>
        )}
      </AnimatePresence>

    </div>
  );
};

export default App;
