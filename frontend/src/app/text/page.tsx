"use client";

import { useState } from "react";
import Link from "next/link";

export default function TextPage() {
    const [text, setText] = useState("");
    const [result, setResult] = useState<any>(null);
    const [loading, setLoading] = useState(false);

    const handleSubmit = async () => {
        if (!text.trim()) return;

        setLoading(true);
        setResult(null);

        try {
            const response = await fetch("http://localhost:8000/predict/text", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text }),
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
                    <div className="text-gray-400 text-sm">Text Analysis</div>
                </div>
            </header>

            <main className="max-w-4xl mx-auto px-6 py-12">
                {/* Title */}
                <div className="text-center mb-10 animate-fade-in">
                    <div className="w-20 h-20 rounded-2xl bg-blue-500/20 border border-blue-500/30 flex items-center justify-center mx-auto mb-6">
                        <span className="text-4xl">📝</span>
                    </div>
                    <h1 className="text-3xl font-bold text-white mb-3">Text Fraud Detection</h1>
                    <p className="text-gray-400">Analyze call transcripts to detect fraud patterns</p>
                </div>

                {/* Input Card */}
                <div className="gradient-border p-6 mb-6 animate-slide-up">
                    <label className="block text-sm font-medium text-gray-300 mb-3">
                        Enter call transcript or suspicious text
                    </label>
                    <textarea
                        className="w-full h-40 px-4 py-3 bg-gray-800/50 border border-gray-700 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:border-blue-500 transition-colors resize-none"
                        placeholder="Paste the call transcript here..."
                        value={text}
                        onChange={(e) => setText(e.target.value)}
                    />
                    <div className="flex justify-between items-center mt-4">
                        <span className="text-sm text-gray-500">{text.length} characters</span>
                        <button
                            onClick={handleSubmit}
                            disabled={loading || !text.trim()}
                            className="btn-primary px-6 py-3 rounded-xl text-white font-medium disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                            {loading ? (
                                <span className="flex items-center gap-2">
                                    <svg className="animate-spin w-4 h-4" viewBox="0 0 24 24">
                                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                                    </svg>
                                    Analyzing...
                                </span>
                            ) : (
                                "Analyze Text"
                            )}
                        </button>
                    </div>
                </div>

                {/* Result Card */}
                {result && (
                    <div className={`gradient-border p-6 animate-fade-in ${result.prediction === "Fraud Call"
                            ? "border-red-500/50"
                            : result.prediction === "Genuine Call"
                                ? "border-green-500/50"
                                : "border-gray-700"
                        }`}>
                        <h3 className="text-lg font-semibold text-white mb-4">Analysis Result</h3>

                        {result.error ? (
                            <div className="text-red-400">{result.error}</div>
                        ) : (
                            <div className="space-y-4">
                                {/* Prediction Badge */}
                                <div className="flex items-center gap-4">
                                    <span className={`text-5xl ${result.prediction === "Fraud Call" ? "animate-pulse" : ""
                                        }`}>
                                        {result.prediction === "Fraud Call" ? "🚨" : "✅"}
                                    </span>
                                    <div>
                                        <div className={`text-2xl font-bold ${result.prediction === "Fraud Call" ? "text-red-400" : "text-green-400"
                                            }`}>
                                            {result.prediction}
                                        </div>
                                        <div className="text-gray-400">
                                            Confidence: {(result.confidence * 100).toFixed(1)}%
                                        </div>
                                    </div>
                                </div>

                                {/* Confidence Bar */}
                                <div>
                                    <div className="flex justify-between text-sm mb-2">
                                        <span className="text-gray-400">Detection Confidence</span>
                                        <span className="text-white">{(result.confidence * 100).toFixed(1)}%</span>
                                    </div>
                                    <div className="h-3 bg-gray-700 rounded-full overflow-hidden">
                                        <div
                                            className={`h-full rounded-full transition-all duration-500 ${result.prediction === "Fraud Call" ? "bg-red-500" : "bg-green-500"
                                                }`}
                                            style={{ width: `${result.confidence * 100}%` }}
                                        />
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>
                )}

                {/* Example Texts */}
                <div className="mt-8">
                    <h4 className="text-sm font-medium text-gray-400 mb-3">Try these examples:</h4>
                    <div className="flex flex-wrap gap-2">
                        {[
                            "Your bank account is blocked. Share OTP immediately to restore access.",
                            "Hello, this is a reminder about your appointment tomorrow at 3 PM.",
                            "You have won a lottery! Send $100 to claim your prize of $1 million."
                        ].map((example, i) => (
                            <button
                                key={i}
                                onClick={() => setText(example)}
                                className="text-sm px-3 py-2 bg-gray-800 text-gray-300 rounded-lg hover:bg-gray-700 transition-colors truncate max-w-xs"
                            >
                                {example.substring(0, 40)}...
                            </button>
                        ))}
                    </div>
                </div>
            </main>
        </div>
    );
}
