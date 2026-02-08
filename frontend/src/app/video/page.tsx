"use client";

import { useState, useRef } from "react";
import Link from "next/link";

export default function VideoPage() {
    const [file, setFile] = useState<File | null>(null);
    const [result, setResult] = useState<any>(null);
    const [loading, setLoading] = useState(false);
    const [dragActive, setDragActive] = useState(false);
    const inputRef = useRef<HTMLInputElement>(null);

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

    const handleSubmit = async () => {
        if (!file) return;

        setLoading(true);
        setResult(null);

        try {
            const formData = new FormData();
            formData.append("file", file);

            const response = await fetch("http://localhost:8000/predict/video", {
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

    const getScoreColor = (score: number) => {
        if (score >= 60) return "text-red-400";
        if (score >= 30) return "text-yellow-400";
        return "text-green-400";
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
                    <div className="text-gray-400 text-sm">Video Analysis</div>
                </div>
            </header>

            <main className="max-w-4xl mx-auto px-6 py-12">
                {/* Title */}
                <div className="text-center mb-10 animate-fade-in">
                    <div className="w-20 h-20 rounded-2xl bg-blue-500/20 border border-blue-500/30 flex items-center justify-center mx-auto mb-6">
                        <span className="text-4xl">🎬</span>
                    </div>
                    <h1 className="text-3xl font-bold text-white mb-3">Video Deepfake Detection</h1>
                    <p className="text-gray-400">Analyze videos for manipulation artifacts and inconsistencies</p>
                </div>

                {/* Upload Area */}
                <div
                    className={`gradient-border p-8 mb-6 animate-slide-up cursor-pointer transition-all ${dragActive ? "border-blue-500 scale-[1.02]" : ""
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
                        accept="video/*"
                        className="hidden"
                        onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
                    />

                    {file ? (
                        <div className="text-center">
                            <div className="text-5xl mb-4">🎥</div>
                            <p className="text-white font-medium">{file.name}</p>
                            <p className="text-gray-500 text-sm">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
                            <button
                                onClick={(e) => {
                                    e.stopPropagation();
                                    setFile(null);
                                    setResult(null);
                                }}
                                className="text-blue-400 text-sm mt-2 hover:underline"
                            >
                                Choose different video
                            </button>
                        </div>
                    ) : (
                        <div className="text-center py-8">
                            <div className="text-5xl mb-4 animate-float">📹</div>
                            <p className="text-white font-medium mb-2">
                                Drop a video here or click to upload
                            </p>
                            <p className="text-gray-500 text-sm">Supports MP4, AVI, MOV, WebM</p>
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
                                    Analyzing Video...
                                </span>
                            ) : (
                                "Analyze Video"
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
                                    <span className={`text-5xl ${result.prediction === "DEEPFAKE" ? "animate-pulse" : ""}`}>
                                        {result.prediction === "DEEPFAKE" ? "⚠️" : result.prediction === "SUSPICIOUS" ? "🔍" : "✅"}
                                    </span>
                                    <div>
                                        <div className={`text-2xl font-bold ${result.prediction === "DEEPFAKE" ? "text-red-400"
                                                : result.prediction === "SUSPICIOUS" ? "text-yellow-400"
                                                    : "text-green-400"
                                            }`}>
                                            {result.prediction}
                                        </div>
                                        <div className={`text-lg ${getScoreColor(result.deepfake_risk_score)}`}>
                                            Risk Score: {result.deepfake_risk_score}%
                                        </div>
                                    </div>
                                </div>

                                {/* Video Info */}
                                {result.video_info && (
                                    <div className="grid grid-cols-3 gap-4">
                                        <div className="bg-gray-800/50 rounded-xl p-4 text-center border border-gray-700">
                                            <div className="text-xl font-bold text-blue-400">{result.video_info.fps?.toFixed(0) || "N/A"}</div>
                                            <div className="text-xs text-gray-400">FPS</div>
                                        </div>
                                        <div className="bg-gray-800/50 rounded-xl p-4 text-center border border-gray-700">
                                            <div className="text-xl font-bold text-blue-400">{result.video_info.duration || "N/A"}s</div>
                                            <div className="text-xs text-gray-400">Duration</div>
                                        </div>
                                        <div className="bg-gray-800/50 rounded-xl p-4 text-center border border-gray-700">
                                            <div className="text-xl font-bold text-blue-400">{result.video_info.frames_analyzed || "N/A"}</div>
                                            <div className="text-xs text-gray-400">Frames</div>
                                        </div>
                                    </div>
                                )}

                                {/* Component Scores */}
                                {result.component_scores && (
                                    <div className="space-y-3">
                                        <h4 className="text-sm font-medium text-gray-400">Component Analysis</h4>
                                        {Object.entries(result.component_scores).map(([key, value]: [string, any]) => (
                                            <div key={key}>
                                                <div className="flex justify-between text-sm mb-1">
                                                    <span className="text-gray-300 capitalize">{key}</span>
                                                    <span className={getScoreColor(value)}>{value}%</span>
                                                </div>
                                                <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
                                                    <div
                                                        className={`h-full rounded-full transition-all duration-500 ${value >= 60 ? "bg-red-500" : value >= 30 ? "bg-yellow-500" : "bg-green-500"
                                                            }`}
                                                        style={{ width: `${Math.min(value, 100)}%` }}
                                                    />
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                )}

                                {/* Disclaimer */}
                                <p className="text-xs text-gray-500 italic">
                                    {result.disclaimer || "Heuristic analysis - not a trained ML classifier"}
                                </p>
                            </div>
                        )}
                    </div>
                )}
            </main>
        </div>
    );
}
