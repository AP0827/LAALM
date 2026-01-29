import React from 'react';
import { motion } from 'framer-motion';
import { FiDownload, FiCheckCircle, FiAlertTriangle, FiCpu, FiClock } from 'react-icons/fi';

const ConfidenceBadge = ({ score }) => {
    let color = "bg-red-500/20 text-red-300 border-red-500/20";
    if (score >= 0.9) color = "bg-green-500/20 text-green-300 border-green-500/20";
    else if (score >= 0.7) color = "bg-yellow-500/20 text-yellow-300 border-yellow-500/20";

    return (
        <span className={`px-2 py-1 rounded text-xs font-mono border ${color}`}>
            {(score * 100).toFixed(1)}%
        </span>
    );
};

const TranscriptCard = ({ title, text, confidence, icon, colorClass, delay }) => (
    <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay }}
        className="glass-card rounded-xl p-6 h-full flex flex-col"
    >
        <div className="flex items-center justify-between mb-4">
            <div className="flex items-center space-x-2">
                <div className={`p-2 rounded-lg bg-white/5 ${colorClass}`}>
                    {icon}
                </div>
                <h3 className="font-semibold text-gray-300">{title}</h3>
            </div>
            <ConfidenceBadge score={confidence} />
        </div>
        <div className="flex-grow bg-black/20 rounded-lg p-4 font-mono text-sm leading-relaxed text-gray-300 overflow-y-auto max-h-[200px]">
            {text}
        </div>
    </motion.div>
);

const ResultsDashboard = ({ result, downloadBaseUrl = "http://localhost:8000" }) => {
    if (!result) return null;

    return (
        <div className="w-full max-w-6xl mx-auto space-y-8 mt-12 pb-20">

            {/* Header Metrics */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <motion.div
                    initial={{ scale: 0.9, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    className="glass-card p-4 rounded-xl flex items-center justify-between"
                >
                    <div>
                        <p className="text-gray-400 text-xs uppercase tracking-wider">Final Confidence</p>
                        <p className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-green-400 to-emerald-500">
                            {(result.final_confidence * 100).toFixed(1)}%
                        </p>
                    </div>
                    <FiCheckCircle className="text-emerald-500/50" size={32} />
                </motion.div>

                <motion.div
                    initial={{ scale: 0.9, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    transition={{ delay: 0.1 }}
                    className="glass-card p-4 rounded-xl flex items-center justify-between"
                >
                    <div>
                        <p className="text-gray-400 text-xs uppercase tracking-wider">Processing Time</p>
                        <p className="text-2xl font-bold text-white">
                            {result.processing_time.toFixed(2)}s
                        </p>
                    </div>
                    <FiClock className="text-blue-500/50" size={32} />
                </motion.div>

                <motion.div
                    initial={{ scale: 0.9, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    transition={{ delay: 0.2 }}
                    className="glass-card p-4 rounded-xl flex items-center justify-between"
                >
                    <div>
                        <p className="text-gray-400 text-xs uppercase tracking-wider">Corrections</p>
                        <p className="text-2xl font-bold text-yellow-500">
                            {result.corrections_applied}
                        </p>
                    </div>
                    <FiAlertTriangle className="text-yellow-500/50" size={32} />
                </motion.div>

                <motion.div
                    initial={{ scale: 0.9, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    transition={{ delay: 0.3 }}
                    className="glass-card p-4 rounded-xl flex items-center justify-between"
                >
                    <div>
                        <p className="text-gray-400 text-xs uppercase tracking-wider">Word Matches</p>
                        <p className="text-2xl font-bold text-purple-400">
                            {(result.agreement_rate * 100).toFixed(1)}%
                        </p>
                    </div>
                    <FiCpu className="text-purple-500/50" size={32} />
                </motion.div>
            </div>

            {/* Main Transcripts Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <TranscriptCard
                    title="Audio (DeepGram)"
                    text={result.audio_transcript}
                    confidence={result.audio_confidence}
                    icon={<span className="text-lg">🎤</span>}
                    colorClass="text-orange-400"
                    delay={0.4}
                />
                <TranscriptCard
                    title="Video (Auto-AVSR)"
                    text={result.video_transcript}
                    confidence={result.video_confidence}
                    icon={<span className="text-lg">📹</span>}
                    colorClass="text-blue-400"
                    delay={0.5}
                />
                <TranscriptCard
                    title="Final Fused Output"
                    text={result.final_transcript}
                    confidence={result.final_confidence}
                    icon={<span className="text-lg">🤖</span>}
                    colorClass="text-purple-400 glow"
                    delay={0.6}
                />
            </div>

            {/* Caption Downloads */}
            {result.captions && (
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.7 }}
                    className="glass-card p-6 rounded-xl border border-indigo-500/30 bg-indigo-500/5"
                >
                    <div className="flex flex-col md:flex-row items-center justify-between gap-4">
                        <div>
                            <h3 className="text-lg font-semibold text-white">Download Captions</h3>
                            <p className="text-sm text-gray-400">Production-ready files with timestamps and formatting</p>
                        </div>
                        <div className="flex gap-4">
                            <a
                                href={`${downloadBaseUrl}${result.captions.srt}`}
                                target="_blank"
                                rel="noreferrer"
                                download
                                className="glass-button px-6 py-2 rounded-lg flex items-center gap-2 text-indigo-300 font-medium"
                            >
                                <FiDownload /> .SRT
                            </a>
                            <a
                                href={`${downloadBaseUrl}${result.captions.vtt}`}
                                target="_blank"
                                rel="noreferrer"
                                download
                                className="glass-button px-6 py-2 rounded-lg flex items-center gap-2 text-purple-300 font-medium"
                            >
                                <FiDownload /> .VTT
                            </a>
                        </div>
                    </div>
                </motion.div>
            )}

            {/* Word Level Details (Collapsed by default logic handled by CSS max-height or separate component if needed) */}
            <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.8 }}
                className="glass-card p-6 rounded-xl"
            >
                <h3 className="text-sm font-medium text-gray-400 mb-4 uppercase tracking-wider">Word-Level Alignment Details</h3>
                <div className="flex flex-wrap gap-2">
                    {result.word_details.map((word, idx) => (
                        <div
                            key={idx}
                            className={`tooltip-trigger px-3 py-1 rounded text-sm relative group cursor-help
                        ${word.agreed ? 'bg-green-500/10 text-green-300 border border-green-500/20' : 'bg-red-500/10 text-red-300 border border-red-500/20'}
                    `}
                        >
                            {word.word}
                            {/* Tooltip */}
                            <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-48 p-2 bg-gray-900 rounded-lg text-xs text-left shadow-xl opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10 border border-white/10">
                                <div className="grid grid-cols-2 gap-x-2">
                                    <span className="text-gray-500">Audio:</span>
                                    <span className={word.deepgram.confidence > 0.8 ? 'text-green-400' : 'text-yellow-400'}>
                                        {word.deepgram.word} ({(word.deepgram.confidence * 100).toFixed(0)}%)
                                    </span>
                                    <span className="text-gray-500">Video:</span>
                                    <span className={word.avsr.confidence > 0.8 ? 'text-green-400' : 'text-yellow-400'}>
                                        {word.avsr.word} ({(word.avsr.confidence * 100).toFixed(0)}%)
                                    </span>
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
            </motion.div>
        </div>
    );
};

export default ResultsDashboard;
