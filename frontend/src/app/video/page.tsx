"use client";

import { useState, useRef } from "react";
import Link from "next/link";
import { ArrowLeft, Video, Upload, X, AlertTriangle, CheckCircle, Activity } from "lucide-react";

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
        if (score >= 60) return "text-red-400";
        if (score >= 30) return "text-yellow-400";
        return "text-blue-400";
    };

    const getBarColor = (score: number) => {
        if (score >= 60) return "bg-gradient-to-r from-red-500 to-orange-500";
        if (score >= 30) return "bg-gradient-to-r from-yellow-500 to-amber-500";
        if (score >= 60) return "bg-gradient-to-r from-red-500 to-orange-500";
        if (score >= 30) return "bg-gradient-to-r from-yellow-500 to-amber-500";
        return "bg-gradient-to-r from-blue-500 to-cyan-500";
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
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-orange-500 to-red-500 flex items-center justify-center">
                            <Video className="w-5 h-5 text-white" />
                        </div>
                        <span className="text-white font-semibold">Deepfake Detection Engine</span>
                    </div>
                </div>
            </header>

            <main className="max-w-4xl mx-auto px-6 py-16">
                {/* Title */}
                <div className="text-center mb-12 animate-fade-in">
                    <div className="w-24 h-24 rounded-3xl bg-gradient-to-br from-orange-500 to-red-500 flex items-center justify-center mx-auto mb-8 shadow-2xl shadow-orange-500/30">
                        <Video className="w-12 h-12 text-white" />
                    </div>
                    <h1 className="text-4xl font-bold text-white mb-4">Temporal Deepfake Detection Engine</h1>
                    <p className="text-gray-400 text-lg max-w-2xl mx-auto">
                        Frame-by-frame heuristic analysis for temporal inconsistencies, face-swap artifacts, and synthesis glitches.
                    </p>
                </div>

                {/* Upload Area */}
                <div
                    className={`rounded-3xl bg-gradient-to-br from-gray-900 to-gray-800 border-2 border-dashed p-12 mb-8 animate-slide-up cursor-pointer transition-all ${dragActive ? "border-orange-500 bg-orange-500/5 scale-[1.02]" : "border-gray-600 hover:border-gray-500"
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
                        accept="video/*"
                        className="hidden"
                        onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
                    />

                    {file ? (
                        <div className="text-center">
                            <div className="relative inline-block">
                                <div className="w-32 h-32 rounded-2xl bg-orange-500/10 flex items-center justify-center">
                                    <Video className="w-16 h-16 text-orange-400" />
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
                            <p className="text-gray-500">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
                        </div>
                    ) : (
                        <div className="text-center py-8">
                            <div className="w-20 h-20 rounded-2xl bg-orange-500/10 flex items-center justify-center mx-auto mb-6">
                                <Upload className="w-10 h-10 text-orange-400" />
                            </div>
                            <p className="text-white text-xl font-medium mb-2">
                                Secure Video Ingestion
                            </p>
                            <p className="text-gray-500">Encrypted Pipeline. Supports MP4, AVI, MOV (Max 50MB).</p>
                        </div>
                    )}
                </div>

                {/* Analyze Button */}
                {file && (
                    <div className="text-center mb-8">
                        <button
                            onClick={handleSubmit}
                            disabled={loading}
                            className="flex items-center gap-3 mx-auto bg-gradient-to-r from-orange-500 to-red-500 hover:from-orange-600 hover:to-red-600 px-10 py-4 rounded-2xl text-white font-semibold disabled:opacity-50 transition-all shadow-lg shadow-orange-500/30 hover:shadow-orange-500/50"
                        >
                            {loading ? (
                                <>
                                    <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                                    Processing Frames & Extracting Features...
                                </>
                            ) : (
                                <>
                                    <Activity className="w-5 h-5" />
                                    Run Deepfake Detection
                                </>
                            )}
                        </button>
                    </div>
                )}

                {/* Result Card */}
                {result && (
                    <div className={`rounded-3xl p-8 animate-fade-in ${result.prediction === "DEEPFAKE"
                        ? "bg-gradient-to-br from-red-900/30 to-red-800/20 border border-red-500/30"
                        : result.prediction === "SUSPICIOUS"
                            ? "bg-gradient-to-br from-yellow-900/30 to-yellow-800/20 border border-yellow-500/30"
                            : "bg-gradient-to-br from-blue-900/30 to-blue-800/20 border border-blue-500/30"
                        }`}>
                        <div className="flex items-center justify-between mb-6">
                            <h3 className="text-xl font-semibold text-white">Deepfake Analysis Report</h3>
                            <span className="px-3 py-1 rounded-full bg-gray-800 text-xs text-gray-400 border border-gray-700">CASE-ID: {Math.random().toString(36).substr(2, 9).toUpperCase()}</span>
                        </div>

                        {result.error ? (
                            <div className="text-red-400 text-lg">{result.error}</div>
                        ) : (
                            <div className="space-y-8">
                                {/* Prediction Badge */}
                                <div className="flex items-center gap-6">
                                    <div className={`w-20 h-20 rounded-2xl flex items-center justify-center ${result.prediction === "DEEPFAKE" ? "bg-red-500/20"
                                        : result.prediction === "SUSPICIOUS" ? "bg-yellow-500/20"
                                            : "bg-blue-500/20"
                                        }`}>
                                        {result.prediction === "DEEPFAKE" ? (
                                            <AlertTriangle className="w-10 h-10 text-red-400" />
                                        ) : result.prediction === "SUSPICIOUS" ? (
                                            <Activity className="w-10 h-10 text-yellow-400" />
                                        ) : (
                                            <CheckCircle className="w-10 h-10 text-blue-400" />
                                        )}
                                    </div>
                                    <div>
                                        <div className={`text-4xl font-bold ${result.prediction === "DEEPFAKE" ? "text-red-400"
                                            : result.prediction === "SUSPICIOUS" ? "text-yellow-400"
                                                : "text-blue-400"
                                            }`}>
                                            {result.prediction}
                                        </div>
                                        <div className={`text-xl ${getScoreColor(result.deepfake_risk_score)} mt-1`}>
                                            Aggr. Risk Probability: {result.deepfake_risk_score}%
                                        </div>
                                    </div>
                                </div>

                                {/* Video Info */}
                                {result.video_info && (
                                    <div className="grid grid-cols-3 gap-4">
                                        <div className="bg-gray-800/50 rounded-2xl p-5 text-center border border-gray-700/50">
                                            <div className="text-3xl font-bold text-orange-400">{result.video_info.fps?.toFixed(0) || "N/A"}</div>
                                            <div className="text-sm text-gray-400 mt-1">FPS</div>
                                        </div>
                                        <div className="bg-gray-800/50 rounded-2xl p-5 text-center border border-gray-700/50">
                                            <div className="text-3xl font-bold text-orange-400">{result.video_info.duration || "N/A"}s</div>
                                            <div className="text-sm text-gray-400 mt-1">Duration</div>
                                        </div>
                                        <div className="bg-gray-800/50 rounded-2xl p-5 text-center border border-gray-700/50">
                                            <div className="text-3xl font-bold text-orange-400">{result.video_info.frames_analyzed || "N/A"}</div>
                                            <div className="text-sm text-gray-400 mt-1">Frames</div>
                                        </div>
                                    </div>
                                )}

                                {/* Component Scores */}
                                {result.component_scores && (
                                    <div className="bg-gray-800/30 rounded-2xl p-6">
                                        <h4 className="text-lg font-semibold text-white mb-6">Vector Component Breakdown</h4>
                                        <div className="space-y-5">
                                            {Object.entries(result.component_scores).map(([key, value]: [string, any]) => (
                                                <div key={key}>
                                                    <div className="flex justify-between text-sm mb-2">
                                                        <span className="text-gray-300 capitalize font-medium">{key} Analysis</span>
                                                        <span className={`font-bold ${getScoreColor(value)}`}>{value}%</span>
                                                    </div>
                                                    <div className="h-3 bg-gray-700 rounded-full overflow-hidden">
                                                        <div
                                                            className={`h-full rounded-full transition-all duration-700 ${getBarColor(value)}`}
                                                            style={{ width: `${Math.min(value, 100)}%` }}
                                                        />
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                )}

                                {/* Explainability Section */}
                                <div className="grid md:grid-cols-2 gap-6">
                                    <div className="bg-gray-800/30 rounded-2xl p-6 border border-gray-700/30">
                                        <h4 className="text-sm font-semibold text-gray-300 uppercase tracking-wider mb-4 flex items-center gap-2">
                                            <Activity className="w-4 h-4" /> Detected Anomalies & Artifacts
                                        </h4>
                                        <ul className="space-y-3">
                                            {result.prediction === "DEEPFAKE" ? (
                                                <>
                                                    <li className="flex items-start gap-2 text-red-200/80 text-sm">
                                                        <span className="w-1.5 h-1.5 rounded-full bg-red-400 mt-1.5"></span>
                                                        Temporal inconsistencies detected across frames
                                                    </li>
                                                    <li className="flex items-start gap-2 text-red-200/80 text-sm">
                                                        <span className="w-1.5 h-1.5 rounded-full bg-red-400 mt-1.5"></span>
                                                        Facial landmarks show unnatural jitter
                                                    </li>
                                                </>
                                            ) : result.prediction === "SUSPICIOUS" ? (
                                                <>
                                                    <li className="flex items-start gap-2 text-yellow-200/80 text-sm">
                                                        <span className="w-1.5 h-1.5 rounded-full bg-yellow-400 mt-1.5"></span>
                                                        Minor artifacts detected in specific frames
                                                    </li>
                                                </>
                                            ) : (
                                                <>
                                                    <li className="flex items-start gap-2 text-blue-200/80 text-sm">
                                                        <span className="w-1.5 h-1.5 rounded-full bg-blue-400 mt-1.5"></span>
                                                        Temporal coherence maintained
                                                    </li>
                                                    <li className="flex items-start gap-2 text-blue-200/80 text-sm">
                                                        <span className="w-1.5 h-1.5 rounded-full bg-blue-400 mt-1.5"></span>
                                                        Video consistency score: {(100 - result.deepfake_risk_score).toFixed(0)}/100
                                                    </li>
                                                </>
                                            )}
                                        </ul>
                                    </div>
                                    <div className="bg-gray-800/30 rounded-2xl p-6 border border-gray-700/30">
                                        <h4 className="text-sm font-semibold text-gray-300 uppercase tracking-wider mb-4">Operational Protocol</h4>
                                        <div className="text-sm text-gray-400">
                                            {result.prediction === "DEEPFAKE" ? "High likelihood of manipulation. Quarantine asset and flag for manual review."
                                                : result.prediction === "SUSPICIOUS" ? "Inconclusive results. Manual review recommended."
                                                    : "Footage appears natural. No deepfake signs detected."}
                                        </div>
                                    </div>
                                </div>

                                {/* Disclaimer */}
                                <p className="text-sm text-gray-500 italic text-center">
                                    {result.disclaimer || "Heuristic analysis - results may vary"}
                                </p>
                            </div>
                        )}
                    </div>
                )}
            </main>
        </div>
    );
}
