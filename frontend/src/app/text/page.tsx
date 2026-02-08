"use client";

import { useState } from "react";
import Link from "next/link";
import { ArrowLeft, FileText, Send, Sparkles } from "lucide-react";

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

    const examples = [
        "Your bank account is blocked. Share OTP immediately to restore access.",
        "Hello, this is a reminder about your appointment tomorrow at 3 PM.",
        "Congratulations! You won $1 million. Send $100 to claim your prize now!"
    ];

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
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-cyan-500 flex items-center justify-center">
                            <FileText className="w-5 h-5 text-white" />
                        </div>
                        <span className="text-white font-semibold">Text Analysis</span>
                    </div>
                </div>
            </header>

            <main className="max-w-4xl mx-auto px-6 py-16">
                {/* Title */}
                <div className="text-center mb-12 animate-fade-in">
                    <div className="w-24 h-24 rounded-3xl bg-gradient-to-br from-blue-500 to-cyan-500 flex items-center justify-center mx-auto mb-8 shadow-2xl shadow-blue-500/30">
                        <FileText className="w-12 h-12 text-white" />
                    </div>
                    <h1 className="text-4xl font-bold text-white mb-4">Text Fraud Detection</h1>
                    <p className="text-gray-400 text-lg max-w-2xl mx-auto">
                        Analyze call transcripts or messages to detect fraud patterns using advanced NLP
                    </p>
                </div>

                {/* Input Card */}
                <div className="rounded-3xl bg-gradient-to-br from-gray-900 to-gray-800 border border-gray-700/50 p-8 mb-8 animate-slide-up">
                    <label className="block text-lg font-semibold text-white mb-4">
                        Enter text to analyze
                    </label>
                    <textarea
                        className="w-full h-48 px-6 py-4 bg-gray-800/50 border border-gray-600 rounded-2xl text-white placeholder-gray-500 focus:outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 transition-all resize-none text-lg"
                        placeholder="Paste the call transcript or suspicious message here..."
                        value={text}
                        onChange={(e) => setText(e.target.value)}
                    />
                    <div className="flex justify-between items-center mt-6">
                        <span className="text-gray-500">{text.length} characters</span>
                        <button
                            onClick={handleSubmit}
                            disabled={loading || !text.trim()}
                            className="flex items-center gap-3 bg-gradient-to-r from-blue-500 to-cyan-500 hover:from-blue-600 hover:to-cyan-600 px-8 py-4 rounded-2xl text-white font-semibold disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-lg shadow-blue-500/30 hover:shadow-blue-500/50"
                        >
                            {loading ? (
                                <>
                                    <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                                    Analyzing...
                                </>
                            ) : (
                                <>
                                    <Send className="w-5 h-5" />
                                    Analyze Text
                                </>
                            )}
                        </button>
                    </div>
                </div>

                {/* Example Texts */}
                <div className="mb-8">
                    <h4 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4 flex items-center gap-2">
                        <Sparkles className="w-4 h-4" />
                        Try these examples
                    </h4>
                    <div className="grid gap-3">
                        {examples.map((example, i) => (
                            <button
                                key={i}
                                onClick={() => setText(example)}
                                className="text-left px-5 py-4 bg-gray-800/50 border border-gray-700/50 text-gray-300 rounded-xl hover:bg-gray-700/50 hover:border-gray-600 transition-all"
                            >
                                "{example}"
                            </button>
                        ))}
                    </div>
                </div>

                {/* Result Card */}
                {result && (
                    <div className={`rounded-3xl p-8 animate-fade-in ${result.prediction === "Fraud Call"
                            ? "bg-gradient-to-br from-red-900/30 to-red-800/20 border border-red-500/30"
                            : result.prediction === "Genuine Call"
                                ? "bg-gradient-to-br from-green-900/30 to-green-800/20 border border-green-500/30"
                                : "bg-gradient-to-br from-gray-900 to-gray-800 border border-gray-700/50"
                        }`}>
                        <h3 className="text-xl font-semibold text-white mb-6">Analysis Result</h3>

                        {result.error ? (
                            <div className="text-red-400 text-lg">{result.error}</div>
                        ) : (
                            <div className="space-y-6">
                                {/* Prediction Badge */}
                                <div className="flex items-center gap-6">
                                    <span className={`text-7xl ${result.prediction === "Fraud Call" ? "animate-pulse" : ""
                                        }`}>
                                        {result.prediction === "Fraud Call" ? "🚨" : "✅"}
                                    </span>
                                    <div>
                                        <div className={`text-4xl font-bold ${result.prediction === "Fraud Call" ? "text-red-400" : "text-green-400"
                                            }`}>
                                            {result.prediction}
                                        </div>
                                        <div className="text-gray-400 text-lg mt-1">
                                            Confidence: {(result.confidence * 100).toFixed(1)}%
                                        </div>
                                    </div>
                                </div>

                                {/* Confidence Bar */}
                                <div className="bg-gray-800/50 rounded-2xl p-6">
                                    <div className="flex justify-between text-sm mb-3">
                                        <span className="text-gray-400 font-medium">Detection Confidence</span>
                                        <span className="text-white font-bold text-lg">{(result.confidence * 100).toFixed(1)}%</span>
                                    </div>
                                    <div className="h-4 bg-gray-700 rounded-full overflow-hidden">
                                        <div
                                            className={`h-full rounded-full transition-all duration-700 ${result.prediction === "Fraud Call" ? "bg-gradient-to-r from-red-500 to-orange-500" : "bg-gradient-to-r from-green-500 to-emerald-500"
                                                }`}
                                            style={{ width: `${result.confidence * 100}%` }}
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
