"use client";

import { useState, useRef } from "react";
import Link from "next/link";
import { ArrowLeft, Mic, Upload, X, Square, CheckCircle, AlertTriangle } from "lucide-react";

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
        <div className="min-h-screen" style={{ background: 'linear-gradient(180deg, #0a0a0f 0%, #0f1419 50%, #0a0a0f 100%)' }}>
            {/* Header */}
            <header className="glass sticky top-0 z-50">
                <div className="max-w-4xl mx-auto px-6 py-4 flex items-center justify-between">
                    <Link href="/" className="flex items-center gap-3 text-gray-400 hover:text-white transition-colors">
                        <ArrowLeft className="w-5 h-5" />
                        <span>Back to Home</span>
                    </Link>
                    <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-green-500 to-emerald-500 flex items-center justify-center">
                            <Mic className="w-5 h-5 text-white" />
                        </div>
                        <span className="text-white font-semibold">Audio Detection</span>
                    </div>
                </div>
            </header>

            <main className="max-w-4xl mx-auto px-6 py-16">
                {/* Title */}
                <div className="text-center mb-12 animate-fade-in">
                    <div className="w-24 h-24 rounded-3xl bg-gradient-to-br from-green-500 to-emerald-500 flex items-center justify-center mx-auto mb-8 shadow-2xl shadow-green-500/30">
                        <Mic className="w-12 h-12 text-white" />
                    </div>
                    <h1 className="text-4xl font-bold text-white mb-4">Audio Fraud Detection</h1>
                    <p className="text-gray-400 text-lg max-w-2xl mx-auto">
                        Record or upload audio to detect fraudulent calls using speech recognition and AI analysis
                    </p>
                </div>

                {/* Recording Section */}
                <div className="rounded-3xl bg-gradient-to-br from-gray-900 to-gray-800 border border-gray-700/50 p-10 mb-8 animate-slide-up">
                    <h3 className="text-lg font-semibold text-white mb-6 text-center">Record Audio</h3>
                    <div className="flex flex-col items-center">
                        <button
                            onClick={recording ? stopRecording : startRecording}
                            className={`w-28 h-28 rounded-full flex items-center justify-center text-white transition-all shadow-2xl ${recording
                                    ? "bg-gradient-to-br from-red-500 to-red-600 shadow-red-500/50 animate-pulse"
                                    : "bg-gradient-to-br from-green-500 to-emerald-500 shadow-green-500/50 hover:scale-105"
                                }`}
                        >
                            {recording ? (
                                <Square className="w-10 h-10" />
                            ) : (
                                <Mic className="w-12 h-12" />
                            )}
                        </button>
                        <p className="text-gray-400 mt-6 text-lg">
                            {recording ? "Recording... Click to stop" : "Click to start recording"}
                        </p>
                        {recording && (
                            <div className="flex items-center gap-2 mt-4">
                                <span className="w-3 h-3 bg-red-500 rounded-full animate-pulse"></span>
                                <span className="text-red-400 font-medium">Recording in progress</span>
                            </div>
                        )}
                    </div>
                </div>

                {/* Upload Area */}
                <div
                    className={`rounded-3xl bg-gradient-to-br from-gray-900 to-gray-800 border-2 border-dashed p-10 mb-8 cursor-pointer transition-all ${dragActive ? "border-green-500 bg-green-500/5 scale-[1.02]" : "border-gray-600 hover:border-gray-500"
                        }`}
                    onDragEnter={handleDrag}
                    onDragLeave={handleDrag}
                    onDragOver={handleDrag}
                    onDrop={handleDrop}
                    onClick={() => !file && inputRef.current?.click()}
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
                            <div className="relative inline-block">
                                <div className="w-24 h-24 rounded-2xl bg-green-500/10 flex items-center justify-center">
                                    <Mic className="w-12 h-12 text-green-400" />
                                </div>
                                <button
                                    onClick={(e) => {
                                        e.stopPropagation();
                                        setFile(null);
                                        setResult(null);
                                    }}
                                    className="absolute -top-3 -right-3 w-10 h-10 bg-red-500 rounded-full flex items-center justify-center text-white hover:bg-red-600 transition-colors shadow-lg"
                                >
                                    <X className="w-5 h-5" />
                                </button>
                            </div>
                            <p className="text-white font-medium mt-6">{file.name}</p>
                            <p className="text-gray-500">{(file.size / 1024).toFixed(1)} KB</p>
                        </div>
                    ) : (
                        <div className="text-center py-4">
                            <div className="w-16 h-16 rounded-2xl bg-green-500/10 flex items-center justify-center mx-auto mb-4">
                                <Upload className="w-8 h-8 text-green-400" />
                            </div>
                            <p className="text-white text-lg font-medium mb-1">Or upload an audio file</p>
                            <p className="text-gray-500">Supports WAV, MP3, WebM, M4A</p>
                        </div>
                    )}
                </div>

                {/* Analyze Button */}
                {file && (
                    <div className="text-center mb-8">
                        <button
                            onClick={handleSubmit}
                            disabled={loading}
                            className="flex items-center gap-3 mx-auto bg-gradient-to-r from-green-500 to-emerald-500 hover:from-green-600 hover:to-emerald-600 px-10 py-4 rounded-2xl text-white font-semibold disabled:opacity-50 transition-all shadow-lg shadow-green-500/30 hover:shadow-green-500/50"
                        >
                            {loading ? (
                                <>
                                    <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                                    Analyzing Audio...
                                </>
                            ) : (
                                "Analyze Audio"
                            )}
                        </button>
                    </div>
                )}

                {/* Result Card */}
                {result && (
                    <div className={`rounded-3xl p-8 animate-fade-in ${result.prediction === "Fraud Call"
                            ? "bg-gradient-to-br from-red-900/30 to-red-800/20 border border-red-500/30"
                            : "bg-gradient-to-br from-green-900/30 to-green-800/20 border border-green-500/30"
                        }`}>
                        <h3 className="text-xl font-semibold text-white mb-6">Analysis Result</h3>

                        {result.error ? (
                            <div className="text-red-400 text-lg">{result.error}</div>
                        ) : (
                            <div className="space-y-6">
                                {/* Prediction Badge */}
                                <div className="flex items-center gap-6">
                                    <div className={`w-20 h-20 rounded-2xl flex items-center justify-center ${result.prediction === "Fraud Call" ? "bg-red-500/20" : "bg-green-500/20"
                                        }`}>
                                        {result.prediction === "Fraud Call" ? (
                                            <AlertTriangle className="w-10 h-10 text-red-400" />
                                        ) : (
                                            <CheckCircle className="w-10 h-10 text-green-400" />
                                        )}
                                    </div>
                                    <div>
                                        <div className={`text-4xl font-bold ${result.prediction === "Fraud Call" ? "text-red-400" : "text-green-400"
                                            }`}>
                                            {result.prediction}
                                        </div>
                                        <div className="text-gray-400 text-lg mt-1">
                                            Confidence: {((result.confidence || 0) * 100).toFixed(1)}%
                                        </div>
                                    </div>
                                </div>

                                {/* Transcription */}
                                {result.transcription && (
                                    <div className="bg-gray-800/30 rounded-2xl p-6">
                                        <h4 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4">Transcription</h4>
                                        <p className="text-gray-200 text-lg leading-relaxed italic">
                                            "{result.transcription}"
                                        </p>
                                    </div>
                                )}

                                {/* Confidence Bar */}
                                <div className="bg-gray-800/30 rounded-2xl p-6">
                                    <div className="flex justify-between text-sm mb-3">
                                        <span className="text-gray-400 font-medium">Detection Confidence</span>
                                        <span className="text-white font-bold text-lg">{((result.confidence || 0) * 100).toFixed(1)}%</span>
                                    </div>
                                    <div className="h-4 bg-gray-700 rounded-full overflow-hidden">
                                        <div
                                            className={`h-full rounded-full transition-all duration-700 ${result.prediction === "Fraud Call"
                                                    ? "bg-gradient-to-r from-red-500 to-orange-500"
                                                    : "bg-gradient-to-r from-green-500 to-emerald-500"
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
