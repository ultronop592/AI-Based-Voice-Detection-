"use client";

import { useState, useRef } from "react";
import Link from "next/link";

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
                    <div className="text-gray-400 text-sm">Image Detection</div>
                </div>
            </header>

            <main className="max-w-4xl mx-auto px-6 py-12">
                {/* Title */}
                <div className="text-center mb-10 animate-fade-in">
                    <div className="w-20 h-20 rounded-2xl bg-blue-500/20 border border-blue-500/30 flex items-center justify-center mx-auto mb-6">
                        <span className="text-4xl">🖼️</span>
                    </div>
                    <h1 className="text-3xl font-bold text-white mb-3">Image Deepfake Detection</h1>
                    <p className="text-gray-400">Upload an image to detect AI-generated deepfakes</p>
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
                        accept="image/*"
                        className="hidden"
                        onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
                    />

                    {preview ? (
                        <div className="text-center">
                            <img
                                src={preview}
                                alt="Preview"
                                className="max-h-64 mx-auto rounded-xl border border-gray-700 mb-4"
                            />
                            <p className="text-gray-400 text-sm">{file?.name}</p>
                            <button
                                onClick={(e) => {
                                    e.stopPropagation();
                                    setFile(null);
                                    setPreview(null);
                                    setResult(null);
                                }}
                                className="text-blue-400 text-sm mt-2 hover:underline"
                            >
                                Choose different image
                            </button>
                        </div>
                    ) : (
                        <div className="text-center py-8">
                            <div className="text-5xl mb-4 animate-float">📷</div>
                            <p className="text-white font-medium mb-2">
                                Drop an image here or click to upload
                            </p>
                            <p className="text-gray-500 text-sm">Supports JPG, PNG, WebP</p>
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
                                    Analyzing...
                                </span>
                            ) : (
                                "Detect Deepfake"
                            )}
                        </button>
                    </div>
                )}

                {/* Result Card */}
                {result && (
                    <div className="gradient-border p-6 animate-fade-in">
                        <h3 className="text-lg font-semibold text-white mb-4">Detection Result</h3>

                        {result.error ? (
                            <div className="text-red-400">{result.error}</div>
                        ) : (
                            <div className="space-y-6">
                                {/* Prediction Badge */}
                                <div className="flex items-center gap-4">
                                    <span className={`text-5xl ${result.prediction === "DEEPFAKE" ? "animate-pulse" : ""}`}>
                                        {result.prediction === "DEEPFAKE" ? "⚠️" : "✅"}
                                    </span>
                                    <div>
                                        <div className={`text-2xl font-bold ${result.prediction === "DEEPFAKE" ? "text-red-400" : "text-green-400"
                                            }`}>
                                            {result.prediction}
                                        </div>
                                        <div className="text-gray-400">
                                            Confidence: {(result.confidence * 100).toFixed(1)}%
                                        </div>
                                    </div>
                                </div>

                                {/* Probability Bars */}
                                {result.probabilities && (
                                    <div className="space-y-3">
                                        <div>
                                            <div className="flex justify-between text-sm mb-1">
                                                <span className="text-green-400">Real</span>
                                                <span className="text-white">{(result.probabilities.real * 100).toFixed(1)}%</span>
                                            </div>
                                            <div className="h-3 bg-gray-700 rounded-full overflow-hidden">
                                                <div
                                                    className="h-full bg-green-500 rounded-full transition-all duration-500"
                                                    style={{ width: `${result.probabilities.real * 100}%` }}
                                                />
                                            </div>
                                        </div>
                                        <div>
                                            <div className="flex justify-between text-sm mb-1">
                                                <span className="text-red-400">Deepfake</span>
                                                <span className="text-white">{(result.probabilities.fake * 100).toFixed(1)}%</span>
                                            </div>
                                            <div className="h-3 bg-gray-700 rounded-full overflow-hidden">
                                                <div
                                                    className="h-full bg-red-500 rounded-full transition-all duration-500"
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
