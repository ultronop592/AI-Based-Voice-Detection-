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
                        <span>Return to Dashboard</span>
                    </Link>
                    <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-cyan-500 flex items-center justify-center">
                            <FileText className="w-5 h-5 text-white" />
                        </div>
                        <span className="text-white font-semibold">Text Detection Module</span>
                    </div>
                </div>
            </header>

            <main className="max-w-4xl mx-auto px-6 py-16">
                {/* Title */}
                <div className="text-center mb-12 animate-fade-in">
                    <div className="w-24 h-24 rounded-3xl bg-gradient-to-br from-blue-500 to-cyan-500 flex items-center justify-center mx-auto mb-8 shadow-2xl shadow-blue-500/30">
                        <FileText className="w-12 h-12 text-white" />
                    </div>
                    <h1 className="text-4xl font-bold text-white mb-4">Linguistic Risk Intelligence</h1>
                    <p className="text-gray-400 text-lg max-w-2xl mx-auto">
                        NLP vector analysis for social engineering patterns and suspicious linguistic constructs.
                    </p>
                </div>

                {/* Input Card */}
                <div className="rounded-3xl bg-gradient-to-br from-gray-900 to-gray-800 border border-gray-700/50 p-8 mb-8 animate-slide-up">
                    <label className="block text-lg font-semibold text-white mb-4">
                        Secure Text Ingestion
                    </label>
                    <textarea
                        className="w-full h-48 px-6 py-4 bg-gray-800/50 border border-gray-600 rounded-2xl text-white placeholder-gray-500 focus:outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 transition-all resize-none text-lg"
                        placeholder="Input transcript or message corpus for linguistic auditing..."
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
                                    Processing Vectors...
                                </>
                            ) : (
                                <>
                                    <Send className="w-5 h-5" />
                                    Run Linguistic Scan
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
                        : "bg-gradient-to-br from-blue-900/30 to-blue-800/20 border border-blue-500/30"
                        }`}>
                        <div className="flex items-center justify-between mb-8">
                            <h3 className="text-xl font-semibold text-white">Linguistic Analysis Report</h3>
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
                                            {result.prediction === "Fraud Call" ? "Fraud Pattern Detected" : "Safe Communication"}
                                        </div>
                                        <p className="text-gray-400 max-w-sm">
                                            {result.prediction === "Fraud Call"
                                                ? "Linguistic analysis detected urgency, financial pressure, or suspicious keywords."
                                                : "Standard communication patterns detected. No malicious intent found."}
                                        </p>
                                    </div>
                                </div>

                                {/* Explainability */}
                                <div className="grid md:grid-cols-2 gap-6">
                                    <div className="bg-gray-800/30 rounded-2xl p-6 border border-gray-700/30">
                                        <h4 className="text-sm font-semibold text-gray-300 uppercase tracking-wider mb-4">Detected Patterns & Keywords</h4>
                                        <ul className="space-y-3">
                                            {result.prediction === "Fraud Call" ? (
                                                <>
                                                    <li className="flex items-start gap-2 text-red-200/80 text-sm">
                                                        <span className="w-1.5 h-1.5 rounded-full bg-red-400 mt-1.5"></span>
                                                        High urgency language detected
                                                    </li>
                                                    <li className="flex items-start gap-2 text-red-200/80 text-sm">
                                                        <span className="w-1.5 h-1.5 rounded-full bg-red-400 mt-1.5"></span>
                                                        Financial pressure keywords identified
                                                    </li>
                                                </>
                                            ) : (
                                                <>
                                                    <li className="flex items-start gap-2 text-blue-200/80 text-sm">
                                                        <span className="w-1.5 h-1.5 rounded-full bg-blue-400 mt-1.5"></span>
                                                        Neutral / Positive sentiment
                                                    </li>
                                                    <li className="flex items-start gap-2 text-blue-200/80 text-sm">
                                                        <span className="w-1.5 h-1.5 rounded-full bg-blue-400 mt-1.5"></span>
                                                        Absence of coercion patterns
                                                    </li>
                                                </>
                                            )}
                                        </ul>
                                    </div>
                                    <div className="bg-gray-800/30 rounded-2xl p-6 border border-gray-700/30">
                                        <h4 className="text-sm font-semibold text-gray-300 uppercase tracking-wider mb-4">Operational Protocol</h4>
                                        <p className="text-sm text-gray-400 leading-relaxed">
                                            {result.prediction === "Fraud Call"
                                                ? "Immediate Caution: Do not share OTPs, PINs, or transfer money. Verify the caller identity through official channels."
                                                : "No immediate action required. Proceed with normal communication protocol."}
                                        </p>
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
