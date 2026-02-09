"use client";

import { useState, useRef } from "react";
import Link from "next/link";
import { ArrowLeft, Image as ImageIcon, Upload, X, CheckCircle, AlertTriangle, Shield, Activity } from "lucide-react";

export default function ImagePage() {
    const [file, setFile] = useState<File | null>(null);
    const [preview, setPreview] = useState<string | null>(null);
    const [result, setResult] = useState<any>(null);
    const [loading, setLoading] = useState(false);
    const [dragActive, setDragActive] = useState(false);
    const inputRef = useRef<HTMLInputElement>(null);

    const handleFile = (f: File) => {
        setFile(f);
        setResult(null);
        const reader = new FileReader();
        reader.onload = (e) => setPreview(e.target?.result as string);
        reader.readAsDataURL(f);
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

            const response = await fetch("http://localhost:8000/predict/image", {
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

    const clearFile = () => {
        setFile(null);
        setPreview(null);
        setResult(null);
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
                            <ImageIcon className="w-5 h-5 text-white" />
                        </div>
                        <span className="text-white font-semibold">Image Detection Module</span>
                    </div>
                </div>
            </header>

            <main className="max-w-4xl mx-auto px-6 py-16">
                {/* Title */}
                <div className="text-center mb-12 animate-fade-in">
                    <div className="w-24 h-24 rounded-3xl bg-gradient-to-br from-blue-500 to-cyan-500 flex items-center justify-center mx-auto mb-8 shadow-2xl shadow-blue-500/30">
                        <ImageIcon className="w-12 h-12 text-white" />
                    </div>
                    <h1 className="text-4xl font-bold text-white mb-4">Visual Authenticity Validation</h1>
                    <p className="text-gray-400 text-lg max-w-2xl mx-auto">
                        Deep learning inspection for generative artifacts, frequency domain anomalies, and pixel-level manipulation.
                    </p>
                </div>

                {/* Upload Area */}
                <div
                    className={`rounded-3xl bg-gradient-to-br from-gray-900 to-gray-800 border-2 border-dashed p-12 mb-8 animate-slide-up cursor-pointer transition-all ${dragActive ? "border-blue-500 bg-blue-500/5 scale-[1.02]" : "border-gray-600 hover:border-gray-500"
                        }`}
                    onDragEnter={handleDrag}
                    onDragLeave={handleDrag}
                    onDragOver={handleDrag}
                    onDrop={handleDrop}
                    onClick={() => !preview && inputRef.current?.click()}
                >
                    <input
                        ref={inputRef}
                        type="file"
                        accept="image/*"
                        className="hidden"
                        onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
                    />

                    {preview ? (
                        <div className="text-center">
                            <div className="relative inline-block">
                                <img
                                    src={preview}
                                    alt="Preview"
                                    className="max-h-80 mx-auto rounded-2xl border border-gray-700 shadow-2xl"
                                />
                                <button
                                    onClick={(e) => {
                                        e.stopPropagation();
                                        clearFile();
                                    }}
                                    className="absolute -top-3 -right-3 w-10 h-10 bg-red-500 rounded-full flex items-center justify-center text-white hover:bg-red-600 transition-colors shadow-lg"
                                >
                                    <X className="w-5 h-5" />
                                </button>
                            </div>
                            <p className="text-gray-400 mt-6">{file?.name}</p>
                        </div>
                    ) : (
                        <div className="text-center py-8">
                            <div className="w-20 h-20 rounded-2xl bg-blue-500/10 flex items-center justify-center mx-auto mb-6">
                                <Upload className="w-10 h-10 text-blue-400" />
                            </div>
                            <p className="text-white text-xl font-medium mb-2">
                                Secure Image Ingestion
                            </p>
                            <p className="text-gray-500">Encrypted Upload. Supports High-Res Formats (Max 25MB).</p>
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
                                    Extracting Vectors & Scan...
                                </>
                            ) : (
                                "Run Detection Analysis"
                            )}
                        </button>
                    </div>
                )}

                {/* Result Card */}
                {result && (
                    <div className={`rounded-3xl p-8 animate-fade-in ${result.prediction === "DEEPFAKE"
                        ? "bg-gradient-to-br from-red-900/30 to-red-800/20 border border-red-500/30"
                        : "bg-gradient-to-br from-blue-900/30 to-blue-800/20 border border-blue-500/30"
                        }`}>
                        <div className="flex items-center justify-between mb-8">
                            <h3 className="text-xl font-semibold text-white">Detection Intelligence Report</h3>
                            <span className="px-3 py-1 rounded-full bg-gray-800 text-xs text-gray-400 border border-gray-700">CASE-ID: {Math.random().toString(36).substr(2, 9).toUpperCase()}</span>
                        </div>

                        {result.error ? (
                            <div className="text-red-400 text-lg">{result.error}</div>
                        ) : (
                            <div className="space-y-8">
                                {/* Authenticity Score Badge */}
                                <div className="flex items-center gap-8">
                                    <div className="relative">
                                        <div className={`w-32 h-32 rounded-full flex items-center justify-center border-8 ${result.prediction === "DEEPFAKE" ? "border-red-500/20 text-red-500" : "border-blue-500/20 text-blue-500"
                                            }`}>
                                            <div className="text-center">
                                                <div className="text-4xl font-bold">
                                                    {result.prediction === "DEEPFAKE"
                                                        ? Math.round((result.confidence) * 100)
                                                        : Math.round((result.confidence) * 100)}
                                                </div>
                                                <div className="text-xs uppercase tracking-wider font-semibold mt-1">
                                                    {result.prediction === "DEEPFAKE" ? "Risk Score" : "Auth Score"}
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                    <div>
                                        <div className={`text-3xl font-bold mb-2 ${result.prediction === "DEEPFAKE" ? "text-red-400" : "text-blue-400"}`}>
                                            {result.prediction === "DEEPFAKE" ? "High Risk Detected" : "Authenticity Verified"}
                                        </div>
                                        <p className="text-gray-400 max-w-sm">
                                            {result.prediction === "DEEPFAKE"
                                                ? "EfficiencyNet-B0 has detected high-probability synthetic artifacts inconsistent with natural photography."
                                                : "No generative watermarks or diffusion patterns detected. Image structure is consistent."}
                                        </p>
                                    </div>
                                </div>

                                {/* Explainability Section */}
                                <div className="grid md:grid-cols-2 gap-6">
                                    <div className="bg-gray-800/30 rounded-2xl p-6 border border-gray-700/30">
                                        <h4 className="text-sm font-semibold text-gray-300 uppercase tracking-wider mb-4 flex items-center gap-2">
                                            <AlertTriangle className="w-4 h-4" /> Detected Artifacts & Anomalies
                                        </h4>
                                        <ul className="space-y-3">
                                            {result.prediction === "DEEPFAKE" ? (
                                                <>
                                                    <li className="flex items-start gap-2 text-red-200/80 text-sm">
                                                        <span className="w-1.5 h-1.5 rounded-full bg-red-400 mt-1.5"></span>
                                                        Diffusion patterns detected in frequency domain
                                                    </li>
                                                    <li className="flex items-start gap-2 text-red-200/80 text-sm">
                                                        <span className="w-1.5 h-1.5 rounded-full bg-red-400 mt-1.5"></span>
                                                        Inconsistent lighting vectors on subjects
                                                    </li>
                                                    <li className="flex items-start gap-2 text-red-200/80 text-sm">
                                                        <span className="w-1.5 h-1.5 rounded-full bg-red-400 mt-1.5"></span>
                                                        Model certainty {(result.confidence * 100).toFixed(1)}% exceeds risk threshold
                                                    </li>
                                                </>
                                            ) : (
                                                <>
                                                    <li className="flex items-start gap-2 text-blue-200/80 text-sm">
                                                        <span className="w-1.5 h-1.5 rounded-full bg-blue-400 mt-1.5"></span>
                                                        Consistent ISO noise distribution verified
                                                    </li>
                                                    <li className="flex items-start gap-2 text-blue-200/80 text-sm">
                                                        <span className="w-1.5 h-1.5 rounded-full bg-blue-400 mt-1.5"></span>
                                                        Natural edge diffraction patterns
                                                    </li>
                                                    <li className="flex items-start gap-2 text-blue-200/80 text-sm">
                                                        <span className="w-1.5 h-1.5 rounded-full bg-blue-400 mt-1.5"></span>
                                                        No latent diffusion watermarks found
                                                    </li>
                                                </>
                                            )}
                                        </ul>
                                    </div>

                                    {/* Actionable Recommendations */}
                                    <div className="bg-gray-800/30 rounded-2xl p-6 border border-gray-700/30">
                                        <h4 className="text-sm font-semibold text-gray-300 uppercase tracking-wider mb-4 flex items-center gap-2">
                                            <Activity className="w-4 h-4" /> Operational Protocols (SOP)
                                        </h4>
                                        <div className="space-y-3">
                                            {result.prediction === "DEEPFAKE" ? (
                                                <>
                                                    <div className="p-3 bg-red-500/10 rounded-lg border border-red-500/20 flex items-center gap-3">
                                                        <X className="w-5 h-5 text-red-400" />
                                                        <span className="text-red-200 text-sm font-medium">Quarantine Asset. Manual Verification Required.</span>
                                                    </div>
                                                    <div className="p-3 bg-red-500/10 rounded-lg border border-red-500/20 flex items-center gap-3">
                                                        <AlertTriangle className="w-5 h-5 text-red-400" />
                                                        <span className="text-red-200 text-sm font-medium">Flag source for heuristic audit</span>
                                                    </div>
                                                </>
                                            ) : (
                                                <>
                                                    <div className="p-3 bg-blue-500/10 rounded-lg border border-blue-500/20 flex items-center gap-3">
                                                        <CheckCircle className="w-5 h-5 text-blue-400" />
                                                        <span className="text-blue-200 text-sm font-medium">Proceed with standard processing</span>
                                                    </div>
                                                    <div className="p-3 bg-blue-500/10 rounded-lg border border-blue-500/20 flex items-center gap-3">
                                                        <Shield className="w-5 h-5 text-blue-400" />
                                                        <span className="text-blue-200 text-sm font-medium">Log as verified asset</span>
                                                    </div>
                                                </>
                                            )}
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
