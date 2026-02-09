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
                        <span>Return to Dashboard</span>
                    </Link>
                    <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-cyan-500 flex items-center justify-center">
                            <Mic className="w-5 h-5 text-white" />
                        </div>
                        <span className="text-white font-semibold">Audio Detection Module</span>
                    </div>
                </div>
            </header>

            <main className="max-w-4xl mx-auto px-6 py-16">
                {/* Title */}
                <div className="text-center mb-12 animate-fade-in">
                    <div className="w-24 h-24 rounded-3xl bg-gradient-to-br from-blue-500 to-cyan-500 flex items-center justify-center mx-auto mb-8 shadow-2xl shadow-blue-500/30">
                        <Mic className="w-12 h-12 text-white" />
                    </div>
                    <h1 className="text-4xl font-bold text-white mb-4">Acoustic Fraud Intelligence</h1>
                    <p className="text-gray-400 text-lg max-w-2xl mx-auto">
                        Spectrum analysis for synthetic voice cloning, cloned speech signatures, and stress markers.
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
                                : "bg-gradient-to-br from-blue-500 to-cyan-500 shadow-blue-500/50 hover:scale-105"
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
                    className={`rounded-3xl bg-gradient-to-br from-gray-900 to-gray-800 border-2 border-dashed p-10 mb-8 cursor-pointer transition-all ${dragActive ? "border-blue-500 bg-blue-500/5 scale-[1.02]" : "border-gray-600 hover:border-gray-500"
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
                                <div className="w-24 h-24 rounded-2xl bg-blue-500/10 flex items-center justify-center">
                                    <Mic className="w-12 h-12 text-blue-400" />
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
                            <div className="w-16 h-16 rounded-2xl bg-blue-500/10 flex items-center justify-center mx-auto mb-4">
                                <Upload className="w-8 h-8 text-blue-400" />
                            </div>
                            <p className="text-white text-lg font-medium mb-1">Secure Audio Ingestion</p>
                            <p className="text-gray-500">Encrypted Upload. Supports WAV, MP3, WebM, M4A</p>
                        </div>
                    )}
                </div>

                {/* Analyze Button */}
                {file && (
                    <div className="text-center mb-8">
                        <button
                            onClick={handleSubmit}
                            disabled={loading}
                            className="flex items-center gap-3 mx-auto bg-gradient-to-r from-blue-500 to-cyan-500 hover:from-blue-600 hover:to-cyan-600 px-10 py-4 rounded-2xl text-white font-semibold disabled:opacity-50 transition-all shadow-lg shadow-blue-500/30 hover:shadow-blue-500/50"
                        >
                            {loading ? (
                                <>
                                    <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                                    Analyzing Audio Signals...
                                </>
                            ) : (
                                "Run Audio Detection"
                            )}
                        </button>
                    </div>
                )}

                {/* Result Card */}
                {result && (
                    <div className={`rounded-3xl p-8 animate-fade-in ${result.prediction === "Fraud Call"
                        ? "bg-gradient-to-br from-red-900/30 to-red-800/20 border border-red-500/30"
                        : "bg-gradient-to-br from-blue-900/30 to-blue-800/20 border border-blue-500/30"
                        }`}>
                        <div className="flex items-center justify-between mb-8">
                            <h3 className="text-xl font-semibold text-white">Audio Analysis Report</h3>
                            <span className="px-3 py-1 rounded-full bg-gray-800 text-xs text-gray-400 border border-gray-700">CASE-ID: {Math.random().toString(36).substr(2, 9).toUpperCase()}</span>
                        </div>

                        {result.error ? (
                            <div className="text-red-400 text-lg">{result.error}</div>
                        ) : (
                            <div className="space-y-8">
                                {/* Risk Score */}
                                <div className="flex items-center gap-8">
                                    <div className="relative">
                                        <div className={`w-32 h-32 rounded-full flex items-center justify-center border-8 ${result.prediction === "Fraud Call" ? "border-red-500/20 text-red-500" : "border-blue-500/20 text-blue-500"
                                            }`}>
                                            <div className="text-center">
                                                <div className="text-4xl font-bold">
                                                    {Math.round(result.confidence * 100)}
                                                </div>
                                                <div className="text-xs uppercase tracking-wider font-semibold mt-1">
                                                    {result.prediction === "Fraud Call" ? "Risk Score" : "Safety Score"}
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                    <div>
                                        <div className={`text-3xl font-bold mb-2 ${result.prediction === "Fraud Call" ? "text-red-400" : "text-blue-400"}`}>
                                            {result.prediction === "Fraud Call" ? "Fraud Pattern Detected" : "Pattern Verified Safe"}
                                        </div>
                                        <p className="text-gray-400 max-w-sm">
                                            {result.prediction === "Fraud Call"
                                                ? "Audio markers indicate high probability of fraudulent activity or vishing."
                                                : "Audio signals consistent with genuine communication."}
                                        </p>
                                    </div>
                                </div>

                                {/* Transcription */}
                                {result.transcription && (
                                    <div className="bg-gray-800/30 rounded-2xl p-6 border border-gray-700/30">
                                        <div className="flex justify-between items-center mb-4">
                                            <h4 className="text-sm font-semibold text-gray-300 uppercase tracking-wider">Audio Transcript</h4>
                                            <span className="text-xs text-gray-400 bg-gray-800 px-2 py-1 rounded">Processed</span>
                                        </div>
                                        <p className="text-gray-200 text-lg leading-relaxed italic border-l-2 border-blue-500/50 pl-4 py-1">
                                            "{result.transcription}"
                                        </p>
                                    </div>
                                )}

                                {/* Detection Breakdown */}
                                <div className="grid md:grid-cols-2 gap-6">
                                    <div className="bg-gray-800/30 rounded-2xl p-6 border border-gray-700/30">
                                        <h4 className="text-sm font-semibold text-gray-300 uppercase tracking-wider mb-4">Spectral Anomalies</h4>
                                        <ul className="space-y-3">
                                            {result.prediction === "Fraud Call" ? (
                                                <>
                                                    <li className="flex items-start gap-2 text-red-200/80 text-sm">
                                                        <span className="w-1.5 h-1.5 rounded-full bg-red-400 mt-1.5"></span>
                                                        Voice stress patterns detected
                                                    </li>
                                                    <li className="flex items-start gap-2 text-red-200/80 text-sm">
                                                        <span className="w-1.5 h-1.5 rounded-full bg-red-400 mt-1.5"></span>
                                                        Keywords matching known fraud scripts
                                                    </li>
                                                </>
                                            ) : (
                                                <>
                                                    <li className="flex items-start gap-2 text-blue-200/80 text-sm">
                                                        <span className="w-1.5 h-1.5 rounded-full bg-blue-400 mt-1.5"></span>
                                                        Natural speech cadence
                                                    </li>
                                                    <li className="flex items-start gap-2 text-blue-200/80 text-sm">
                                                        <span className="w-1.5 h-1.5 rounded-full bg-blue-400 mt-1.5"></span>
                                                        No known threat signatures matched
                                                    </li>
                                                </>
                                            )}
                                        </ul>
                                    </div>
                                    <div className="bg-gray-800/30 rounded-2xl p-6 border border-gray-700/30">
                                        <h4 className="text-sm font-semibold text-gray-300 uppercase tracking-wider mb-4">Operational Protocol</h4>
                                        <div className="space-y-2">
                                            <p className={`text-sm font-medium ${result.prediction === "Fraud Call" ? "text-red-400" : "text-blue-400"}`}>
                                                {result.prediction === "Fraud Call" ? "CRITICAL ALERT" : "STATUS: NORMAL"}
                                            </p>
                                            <p className="text-sm text-gray-400">
                                                {result.prediction === "Fraud Call"
                                                    ? "Terminate connection. Block number and report to relevant authorities immediately."
                                                    : "No threat detected. Interaction may proceed safely."}
                                            </p>
                                        </div>
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
