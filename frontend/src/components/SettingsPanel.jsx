import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FiSliders, FiActivity, FiZap, FiInfo } from 'react-icons/fi';

const SettingsPanel = ({ settings, setSettings }) => {
    const toggleAdvanced = () => {
        setSettings(prev => ({
            ...prev,
            use_advanced_preprocessing: !prev.use_advanced_preprocessing
        }));
    };

    const updateDenoise = (e) => {
        setSettings(prev => ({
            ...prev,
            denoise_strength: parseInt(e.target.value)
        }));
    };

    const toggleSmoothing = () => {
        setSettings(prev => ({
            ...prev,
            use_temporal_smoothing: !prev.use_temporal_smoothing
        }));
    };

    return (
        <div className="w-full max-w-4xl mx-auto mt-8">
            <motion.div
                className="glass-card rounded-2xl p-6 border-l-4 border-l-purple-500"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
            >
                <div className="flex items-center mb-6 space-x-3">
                    <FiSliders className="text-purple-400" size={24} />
                    <h3 className="text-xl font-semibold text-white">Preprocessing Controls</h3>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-8">

                    {/* Main Toggle */}
                    <div className="space-y-4">
                        <div className="flex items-center justify-between">
                            <div className="space-y-1">
                                <div className="flex items-center space-x-2">
                                    <span className="font-medium text-white">Advanced Enhancement</span>
                                    <FiZap className="text-yellow-400" />
                                </div>
                                <p className="text-xs text-gray-400">Activates VSR specific filters</p>
                            </div>

                            <button
                                onClick={toggleAdvanced}
                                className={`w-14 h-8 rounded-full p-1 transition-colors duration-300 ${settings.use_advanced_preprocessing ? 'bg-purple-600' : 'bg-gray-700'
                                    }`}
                            >
                                <motion.div
                                    className="w-6 h-6 bg-white rounded-full shadow-md"
                                    layout
                                    transition={{ type: "spring", stiffness: 700, damping: 30 }}
                                    animate={{
                                        x: settings.use_advanced_preprocessing ? 24 : 0
                                    }}
                                />
                            </button>
                        </div>

                        <div className="flex items-center justify-between opacity-80">
                            <div className="space-y-1">
                                <span className="font-medium text-gray-300">Temporal Smoothing</span>
                                <p className="text-xs text-gray-500">Frame averaging (can blur lips)</p>
                            </div>
                            <button
                                onClick={toggleSmoothing}
                                disabled={!settings.use_advanced_preprocessing}
                                className={`w-12 h-6 rounded-full p-1 transition-colors duration-300 ${!settings.use_advanced_preprocessing ? 'opacity-50 cursor-not-allowed bg-gray-800' :
                                    settings.use_temporal_smoothing ? 'bg-blue-600' : 'bg-gray-700'
                                    }`}
                            >
                                <motion.div
                                    className="w-4 h-4 bg-white rounded-full shadow-md"
                                    layout
                                    animate={{ x: settings.use_temporal_smoothing ? 24 : 0 }}
                                />
                            </button>
                        </div>
                    </div>

                    {/* Video Denoise Slider */}
                    <div className={`space-y-4 transition-opacity duration-300 ${!settings.use_advanced_preprocessing ? 'opacity-50 pointer-events-none' : ''}`}>
                        <div className="flex justify-between items-center">
                            <span className="font-medium text-gray-300 flex items-center gap-2">
                                <FiActivity className="text-blue-400" /> Video Denoise
                            </span>
                            <span className="bg-blue-500/20 text-blue-300 px-2 py-1 rounded text-xs font-mono">
                                {settings.video_denoise_strength} / 10
                            </span>
                        </div>

                        <input
                            type="range"
                            min="0"
                            max="10"
                            value={settings.video_denoise_strength}
                            onChange={(e) => setSettings(prev => ({ ...prev, video_denoise_strength: parseInt(e.target.value) }))}
                            className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
                        />
                        <div className="flex justify-between text-xs text-gray-500 font-mono">
                            <span>Off</span>
                            <span>Strong</span>
                        </div>
                    </div>

                    {/* Audio Denoise Slider */}
                    <div className="space-y-4">
                        <div className="flex justify-between items-center">
                            <span className="font-medium text-gray-300 flex items-center gap-2">
                                <FiActivity className="text-green-400" /> Audio Denoise
                            </span>
                            <span className="bg-green-500/20 text-green-300 px-2 py-1 rounded text-xs font-mono">
                                {settings.audio_denoise_strength} / 10
                            </span>
                        </div>

                        <input
                            type="range"
                            min="0"
                            max="10"
                            value={settings.audio_denoise_strength}
                            onChange={(e) => setSettings(prev => ({ ...prev, audio_denoise_strength: parseInt(e.target.value) }))}
                            className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-green-500"
                        />

                        <div className="flex justify-between text-xs text-gray-500 font-mono">
                            <span>Raw</span>
                            <span> aggressive (&gt;5)</span>
                        </div>

                        {settings.audio_denoise_strength > 2 && (
                            <motion.div
                                initial={{ opacity: 0, height: 0 }}
                                animate={{ opacity: 1, height: 'auto' }}
                                className="flex items-start gap-2 text-xs text-green-500 bg-green-500/10 p-2 rounded"
                            >
                                <FiInfo className="mt-0.5 flex-shrink-0" />
                                <span>Active: Bandpass Filter + FFT Noise Reduction (Level {settings.audio_denoise_strength}).</span>
                            </motion.div>
                        )}
                    </div>

                </div>
            </motion.div>
        </div>
    );
};

export default SettingsPanel;
