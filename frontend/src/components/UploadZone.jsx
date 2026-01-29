import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { motion, AnimatePresence } from 'framer-motion';
import { FiUploadCloud, FiVideo, FiMusic, FiCheck } from 'react-icons/fi';

const UploadZone = ({ onFileSelect, selectedVideo, selectedAudio }) => {
    const onDrop = useCallback((acceptedFiles) => {
        acceptedFiles.forEach((file) => {
            if (file.type.startsWith('video/')) {
                onFileSelect('video', file);
            } else if (file.type.startsWith('audio/')) {
                onFileSelect('audio', file);
            }
        });
    }, [onFileSelect]);

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: {
            'video/*': [],
            'audio/*': []
        }
    });

    return (
        <div className="w-full max-w-4xl mx-auto space-y-6">
            <motion.div
                {...getRootProps()}
                className={`relative overflow-hidden glass-card rounded-3xl p-12 text-center cursor-pointer transition-all duration-300 group
          ${isDragActive ? 'border-indigo-500 bg-indigo-500/10' : 'border-white/10 hover:border-indigo-400/50'}
        `}
                whileHover={{ scale: 1.01 }}
                whileTap={{ scale: 0.99 }}
            >
                <input {...getInputProps()} />

                {/* Animated Glow Effect */}
                <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/5 to-transparent -translate-x-full group-hover:animate-[shimmer_2s_infinite]" />

                <div className="flex flex-col items-center justify-center space-y-6">
                    <motion.div
                        animate={{ y: isDragActive ? -10 : 0 }}
                        className="p-6 rounded-full bg-indigo-500/20 text-indigo-400"
                    >
                        <FiUploadCloud size={48} />
                    </motion.div>

                    <div className="space-y-2">
                        <h3 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-indigo-200 to-white">
                            {isDragActive ? "Drop files here..." : "Drag & Drop Video or Audio"}
                        </h3>
                        <p className="text-gray-400">
                            Supports MP4, AVI, MOV, WAV, MP3
                        </p>
                    </div>
                </div>
            </motion.div>

            {/* File Indicators */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Video Indicator */}
                <AnimatePresence>
                    <motion.div
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        className={`glass-card p-4 rounded-xl flex items-center justify-between
              ${selectedVideo ? 'border-green-500/50 bg-green-500/5' : 'border-white/5 opacity-50'}
            `}
                    >
                        <div className="flex items-center space-x-3">
                            <div className={`p-2 rounded-lg ${selectedVideo ? 'bg-green-500/20 text-green-400' : 'bg-white/5'}`}>
                                <FiVideo size={24} />
                            </div>
                            <div className="text-left">
                                <p className="text-sm font-medium text-gray-300">Video Source</p>
                                <p className="text-xs text-gray-500 truncate max-w-[200px]">
                                    {selectedVideo ? selectedVideo.name : "Required"}
                                </p>
                            </div>
                        </div>
                        {selectedVideo && (
                            <motion.div
                                initial={{ scale: 0 }}
                                animate={{ scale: 1 }}
                                className="text-green-400"
                            >
                                <FiCheck size={20} />
                            </motion.div>
                        )}
                    </motion.div>
                </AnimatePresence>

                {/* Audio Indicator */}
                <AnimatePresence>
                    <motion.div
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        className={`glass-card p-4 rounded-xl flex items-center justify-between
              ${selectedAudio ? 'border-orange-500/50 bg-orange-500/5' : 'border-white/5 opacity-50'}
            `}
                    >
                        <div className="flex items-center space-x-3">
                            <div className={`p-2 rounded-lg ${selectedAudio ? 'bg-orange-500/20 text-orange-400' : 'bg-white/5'}`}>
                                <FiMusic size={24} />
                            </div>
                            <div className="text-left">
                                <p className="text-sm font-medium text-gray-300">Audio Source</p>
                                <p className="text-xs text-gray-500 truncate max-w-[200px]">
                                    {selectedAudio ? selectedAudio.name : "Optional"}
                                </p>
                            </div>
                        </div>
                        {selectedAudio && (
                            <motion.div
                                initial={{ scale: 0 }}
                                animate={{ scale: 1 }}
                                className="text-orange-400"
                            >
                                <FiCheck size={20} />
                            </motion.div>
                        )}
                    </motion.div>
                </AnimatePresence>
            </div>
        </div>
    );
};

export default UploadZone;
