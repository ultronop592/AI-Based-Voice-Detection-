"use client";

import { useState, useRef } from "react";
import Link from "next/link";

export default function AudioPage() {
    const [file, setFile] = useState<File | null>(null);
    const [result, setResult] = useState<any>(null);
    const [loading, setLoading] = useState(false);
    const [recording, setRecording] = useState(false);
    const [mediaRecorder, setMediaRecorder] = useState<MediaRecorder | null>(null);
    const [dragActive, setDragActive] = useState(false);
    const inputRef = useRef<HTMLInputElement>(null);
    const chunksRef = useRef<Blob[]>([]);

    const handleFile = (f: File) => {
        setFile(f);
        setResult(null);
    };

    const handleDrag = (e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === "dragenter" || e.type === "dragover") setDragActive(true);
        else if (e.type === "dragleave") setDragActive(false);
    };

    const handleDrop = (e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);
        if (e.dataTransfer.files?.[0]) handleFile(e.dataTransfer.files[0]);
    };

    const startRecording = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            const recorder = new MediaRecorder(stream);
            chunksRef.current = [];

            recorder.ondataavailable = (e) => {
                if (e.data.size > 0) chunksRef.current.push(e.data);
            };

            recorder.onstop = () => {
                const blob = new Blob(chunksRef.current, { type: "audio/webm" });
                const f = new File([blob], "recording.webm", { type: "audio/webm" });
                setFile(f);
                stream.getTracks().forEach((t) => t.stop());
            };

            recorder.start();
            setMediaRecorder(recorder);
            setRecording(true);
        } catch (error) {
            console.error("Failed to start recording:", error);
        }
    };

    const stopRecording = () => {
        if (mediaRecorder) {
            mediaRecorder.stop();
            setRecording(false);
        }
    };

    const handleSubmit = async () => {
        if (!file) return;

        setLoading(true);
        setResult(null);

        try {
            const formData = new FormData();
            formData.append("file", file);

            const response = await fetch("http://localhost:8000/predict/audio", {
                method: "POST",
                body: formData,
            });

            const data = await response.json();
            setResult(data);
        } catch (error) {
            setResult({ error: "Failed to connect to API" });
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen" style={{ background: '#0F1419' }}>
            {/* Header */}
            <header className="glass sticky top-0 z-50">
                <div className="max-w-4xl mx-auto px-6 py-4 flex items-center justify-between">
                    <Link href="/" className="flex items-center gap-3 hover:opacity-80 transition-opacity">
                        <div className="w-10 h-10 rounded-xl bg-blue-500 flex items-center justify-center">
                            <span className="text-xl">🛡️</span>
                        </div>
                        <span className="text-white font-semibold">AI Detection Hub</span>
                    </Link>
                    <div className="text-gray-400 text-sm">Audio Detection</div>
                </div>
            </header>

            <main className="max-w-4xl mx-auto px-6 py-12">
                {/* Title */}
                <div className="text-center mb-10 animate-fade-in">
                    <div className="w-20 h-20 rounded-2xl bg-blue-500/20 border border-blue-500/30 flex items-center justify-center mx-auto mb-6">
                        <span className="text-4xl">🎤</span>
                    </div>
                    <h1 className="text-3xl font-bold text-white mb-3">Audio Fraud Detection</h1>
                    <p className="text-gray-400">Record or upload audio to detect fraudulent calls</p>
                </div>

                {/* Recording Section */}
                <div className="gradient-border p-6 mb-6 animate-slide-up">
                    <div className="text-center">
                        <button
                            onClick={recording ? stopRecording : startRecording}
                            className={`w-24 h-24 rounded-full flex items-center justify-center text-4xl transition-all ${recording
                                    ? "bg-red-500 animate-pulse shadow-lg shadow-red-500/50"
                                    : "bg-blue-500 hover:bg-blue-400 shadow-lg shadow-blue-500/30"
                                }`}
                        >
                            {recording ? "⏹️" : "🎙️"}
                        </button>
                        <p className="text-gray-400 text-sm mt-4">
                            {recording ? "Recording... Click to stop" : "Click to start recording"}
                        </p>
                    </div>
                </div>

                {/* Upload Area */}
                <div
                    className={`gradient-border p-6 mb-6 cursor-pointer transition-all ${dragActive ? "border-blue-500 scale-[1.02]" : ""
                        }`}
                    onDragEnter={handleDrag}
                    onDragLeave={handleDrag}
                    onDragOver={handleDrag}
                    onDrop={handleDrop}
                    onClick={() => inputRef.current?.click()}
                >
                    <input
                        ref={inputRef}
                        type="file"
                        accept="audio/*"
                        className="hidden"
                        onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
                    />

                    {file ? (
                        <div className="text-center">
                            <div className="text-4xl mb-3">🎵</div>
                            <p className="text-white font-medium">{file.name}</p>
                            <p className="text-gray-500 text-sm">{(file.size / 1024).toFixed(1)} KB</p>
                            <button
                                onClick={(e) => {
                                    e.stopPropagation();
                                    setFile(null);
                                    setResult(null);
                                }}
                                className="text-blue-400 text-sm mt-2 hover:underline"
                            >
                                Remove file
                            </button>
                        </div>
                    ) : (
                        <div className="text-center py-4">
                            <div className="text-4xl mb-3">📁</div>
                            <p className="text-white font-medium">Or drop an audio file here</p>
                            <p className="text-gray-500 text-sm">Supports WAV, MP3, WebM, M4A</p>
                        </div>
                    )}
                </div>

                {/* Analyze Button */}
                {file && (
                    <div className="text-center mb-6">
                        <button
                            onClick={handleSubmit}
                            disabled={loading}
                            className="btn-primary px-8 py-3 rounded-xl text-white font-medium disabled:opacity-50"
                        >
                            {loading ? (
                                <span className="flex items-center gap-2">
                                    <svg className="animate-spin w-5 h-5" viewBox="0 0 24 24">
                                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                                    </svg>
                                    Analyzing Audio...
                                </span>
                            ) : (
                                "Analyze Audio"
                            )}
                        </button>
                    </div>
                )}

                {/* Result Card */}
                {result && (
                    <div className="gradient-border p-6 animate-fade-in">
                        <h3 className="text-lg font-semibold text-white mb-4">Analysis Result</h3>

                        {result.error ? (
                            <div className="text-red-400">{result.error}</div>
                        ) : (
                            <div className="space-y-6">
                                {/* Prediction Badge */}
                                <div className="flex items-center gap-4">
                                    <span className={`text-5xl ${result.prediction === "Fraud Call" ? "animate-pulse" : ""}`}>
                                        {result.prediction === "Fraud Call" ? "🚨" : "✅"}
                                    </span>
                                    <div>
                                        <div className={`text-2xl font-bold ${result.prediction === "Fraud Call" ? "text-red-400" : "text-green-400"
                                            }`}>
                                            {result.prediction}
                                        </div>
                                        <div className="text-gray-400">
                                            Confidence: {((result.confidence || 0) * 100).toFixed(1)}%
                                        </div>
                                    </div>
                                </div>

                                {/* Transcription */}
                                {result.transcription && (
                                    <div>
                                        <h4 className="text-sm font-medium text-gray-400 mb-2">Transcription</h4>
                                        <div className="bg-gray-800/50 rounded-xl p-4 border border-gray-700">
                                            <p className="text-gray-300 text-sm italic">"{result.transcription}"</p>
                                        </div>
                                    </div>
                                )}

                                {/* Confidence Bar */}
                                <div>
                                    <div className="flex justify-between text-sm mb-2">
                                        <span className="text-gray-400">Detection Confidence</span>
                                        <span className="text-white">{((result.confidence || 0) * 100).toFixed(1)}%</span>
                                    </div>
                                    <div className="h-3 bg-gray-700 rounded-full overflow-hidden">
                                        <div
                                            className={`h-full rounded-full transition-all duration-500 ${result.prediction === "Fraud Call" ? "bg-red-500" : "bg-green-500"
                                                }`}
                                            style={{ width: `${(result.confidence || 0) * 100}%` }}
                                        />
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>
                )}
            </main>
        </div>
    );
}
