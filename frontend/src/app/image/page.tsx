"use client";

import { useState, useRef } from "react";
import Link from "next/link";
import { ArrowLeft, Image as ImageIcon, Upload, X, CheckCircle, AlertTriangle } from "lucide-react";

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
                        <span>Back to Home</span>
                    </Link>
                    <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-purple-500 to-violet-500 flex items-center justify-center">
                            <ImageIcon className="w-5 h-5 text-white" />
                        </div>
                        <span className="text-white font-semibold">Image Detection</span>
                    </div>
                </div>
            </header>

            <main className="max-w-4xl mx-auto px-6 py-16">
                {/* Title */}
                <div className="text-center mb-12 animate-fade-in">
                    <div className="w-24 h-24 rounded-3xl bg-gradient-to-br from-purple-500 to-violet-500 flex items-center justify-center mx-auto mb-8 shadow-2xl shadow-purple-500/30">
                        <ImageIcon className="w-12 h-12 text-white" />
                    </div>
                    <h1 className="text-4xl font-bold text-white mb-4">Image Deepfake Detection</h1>
                    <p className="text-gray-400 text-lg max-w-2xl mx-auto">
                        Upload an image to detect AI-generated deepfakes using our trained neural network
                    </p>
                </div>

                {/* Upload Area */}
                <div
                    className={`rounded-3xl bg-gradient-to-br from-gray-900 to-gray-800 border-2 border-dashed p-12 mb-8 animate-slide-up cursor-pointer transition-all ${dragActive ? "border-purple-500 bg-purple-500/5 scale-[1.02]" : "border-gray-600 hover:border-gray-500"
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
                            <div className="w-20 h-20 rounded-2xl bg-purple-500/10 flex items-center justify-center mx-auto mb-6">
                                <Upload className="w-10 h-10 text-purple-400" />
                            </div>
                            <p className="text-white text-xl font-medium mb-2">
                                Drop an image here or click to upload
                            </p>
                            <p className="text-gray-500">Supports JPG, PNG, WebP up to 10MB</p>
                        </div>
                    )}
                </div>

                {/* Analyze Button */}
                {file && (
                    <div className="text-center mb-8">
                        <button
                            onClick={handleSubmit}
                            disabled={loading}
                            className="flex items-center gap-3 mx-auto bg-gradient-to-r from-purple-500 to-violet-500 hover:from-purple-600 hover:to-violet-600 px-10 py-4 rounded-2xl text-white font-semibold disabled:opacity-50 transition-all shadow-lg shadow-purple-500/30 hover:shadow-purple-500/50"
                        >
                            {loading ? (
                                <>
                                    <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                                    Analyzing...
                                </>
                            ) : (
                                "Detect Deepfake"
                            )}
                        </button>
                    </div>
                )}

                {/* Result Card */}
                {result && (
                    <div className={`rounded-3xl p-8 animate-fade-in ${result.prediction === "DEEPFAKE"
                            ? "bg-gradient-to-br from-red-900/30 to-red-800/20 border border-red-500/30"
                            : "bg-gradient-to-br from-green-900/30 to-green-800/20 border border-green-500/30"
                        }`}>
                        <h3 className="text-xl font-semibold text-white mb-6">Detection Result</h3>

                        {result.error ? (
                            <div className="text-red-400 text-lg">{result.error}</div>
                        ) : (
                            <div className="space-y-6">
                                {/* Prediction Badge */}
                                <div className="flex items-center gap-6">
                                    <div className={`w-20 h-20 rounded-2xl flex items-center justify-center ${result.prediction === "DEEPFAKE" ? "bg-red-500/20" : "bg-green-500/20"
                                        }`}>
                                        {result.prediction === "DEEPFAKE" ? (
                                            <AlertTriangle className="w-10 h-10 text-red-400" />
                                        ) : (
                                            <CheckCircle className="w-10 h-10 text-green-400" />
                                        )}
                                    </div>
                                    <div>
                                        <div className={`text-4xl font-bold ${result.prediction === "DEEPFAKE" ? "text-red-400" : "text-green-400"
                                            }`}>
                                            {result.prediction}
                                        </div>
                                        <div className="text-gray-400 text-lg mt-1">
                                            Confidence: {(result.confidence * 100).toFixed(1)}%
                                        </div>
                                    </div>
                                </div>

                                {/* Probability Bars */}
                                {result.probabilities && (
                                    <div className="grid md:grid-cols-2 gap-6 bg-gray-800/30 rounded-2xl p-6">
                                        <div>
                                            <div className="flex justify-between text-sm mb-2">
                                                <span className="text-green-400 font-medium flex items-center gap-2">
                                                    <CheckCircle className="w-4 h-4" /> Real
                                                </span>
                                                <span className="text-white font-bold">{(result.probabilities.real * 100).toFixed(1)}%</span>
                                            </div>
                                            <div className="h-4 bg-gray-700 rounded-full overflow-hidden">
                                                <div
                                                    className="h-full bg-gradient-to-r from-green-500 to-emerald-500 rounded-full transition-all duration-700"
                                                    style={{ width: `${result.probabilities.real * 100}%` }}
                                                />
                                            </div>
                                        </div>
                                        <div>
                                            <div className="flex justify-between text-sm mb-2">
                                                <span className="text-red-400 font-medium flex items-center gap-2">
                                                    <AlertTriangle className="w-4 h-4" /> Deepfake
                                                </span>
                                                <span className="text-white font-bold">{(result.probabilities.fake * 100).toFixed(1)}%</span>
                                            </div>
                                            <div className="h-4 bg-gray-700 rounded-full overflow-hidden">
                                                <div
                                                    className="h-full bg-gradient-to-r from-red-500 to-orange-500 rounded-full transition-all duration-700"
                                                    style={{ width: `${result.probabilities.fake * 100}%` }}
                                                />
                                            </div>
                                        </div>
                                    </div>
                                )}
                            </div>
                        )}
                    </div>
                )}
            </main>
        </div>
    );
}
