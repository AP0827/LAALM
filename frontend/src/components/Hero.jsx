import React from 'react';
import { motion } from 'framer-motion';

const Hero = () => {
    return (
        <div className="relative w-full flex flex-col items-center justify-center pt-24 pb-12 text-center z-10">

            {/* Background Ambience */}
            <div className="absolute top-0 left-1/2 -translate-x-1/2 w-full h-[500px] overflow-hidden pointer-events-none -z-10">
                <div className="absolute top-[-20%] left-[20%] w-[500px] h-[500px] bg-indigo-500/20 rounded-full blur-[100px] animate-float" />
                <div className="absolute top-[10%] right-[20%] w-[400px] h-[400px] bg-purple-500/20 rounded-full blur-[80px] animate-float" style={{ animationDelay: '2s' }} />
            </div>

            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8 }}
            >
                <span className="px-4 py-1.5 rounded-full border border-indigo-500/30 bg-indigo-500/10 text-indigo-300 text-sm font-medium tracking-wide mb-6 inline-block backdrop-blur-md">
                    v1.0 • Multi-Modal Intelligence
                </span>

                <h1 className="text-6xl md:text-8xl font-black tracking-tight mb-6">
                    <span className="text-white drop-shadow-2xl">L</span>
                    <span className="text-indigo-400 drop-shadow-2xl">AA</span>
                    <span className="text-white drop-shadow-2xl">LM</span>
                </h1>

                <p className="text-xl md:text-2xl text-gray-400 max-w-2xl mx-auto font-light leading-relaxed">
                    Lip-Reading Enhanced Speech Recognition
                    <br />
                    <span className="aurora-text font-semibold">
                        Audio + Vision + LLM Correction
                    </span>
                </p>

                <motion.div
                    className="mt-12 flex space-x-2 justify-center"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 1 }}
                >
                    <div className="w-2 h-2 rounded-full bg-white/20 animate-pulse" />
                    <div className="w-2 h-2 rounded-full bg-white/20 animate-pulse delay-100" />
                    <div className="w-2 h-2 rounded-full bg-white/20 animate-pulse delay-200" />
                </motion.div>
            </motion.div>
        </div>
    );
};

export default Hero;
