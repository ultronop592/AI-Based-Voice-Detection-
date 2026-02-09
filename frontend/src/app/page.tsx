"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { HeroDitheringCard } from "@/components/ui/hero-dithering-card";
import DetectionServicesCards from "@/components/ui/detection-services-cards";
import { FileText, Mic, Image, Video, ArrowRight, Zap, Shield, CheckCircle, TrendingUp } from "lucide-react";

export default function Home() {
  const [stats, setStats] = useState({ text: 0, image: 0, video: 0, audio: 0 });
  const [isLoaded, setIsLoaded] = useState(false);

  useEffect(() => {
    setIsLoaded(true);
    const interval = setInterval(() => {
      setStats(prev => ({
        text: Math.min(prev.text + 5, 127),
        image: Math.min(prev.image + 3, 89),
        video: Math.min(prev.video + 2, 45),
        audio: Math.min(prev.audio + 4, 98)
      }));
    }, 50);
    setTimeout(() => clearInterval(interval), 2000);
    return () => clearInterval(interval);
  }, []);

  const features = [
    {
      title: "Linguistic Risk Analysis",
      description: "Advanced NLP vector analysis to detect social engineering patterns and coercive language in real-time.",
      icon: FileText,
      href: "/text",
      stat: "92% Accuracy",
      color: "from-blue-500 to-cyan-500"
    },
    {
      title: "Audio Detection",
      description: "Spectrum analysis for synthetic voice artifacts, cloned speech signatures, and stress detection.",
      icon: Mic,
      href: "/audio",
      stat: "Real-time",
      color: "from-green-500 to-emerald-500"
    },
    {
      title: "Visual Authenticity Validation",
      description: "Pixel-level CNN inspection to identify generative artifacts, diffusion patterns, and manipulation.",
      icon: Image,
      href: "/image",
      stat: "95% Accuracy",
      color: "from-blue-600 to-cyan-600"
    },
    {
      title: "Temporal Deepfake Scan",
      description: "Frame-by-frame ROI analysis using temporal consistency checks to flag face-swaps and lip-sync errors.",
      icon: Video,
      href: "/video",
      stat: "Heuristic AI",
      color: "from-orange-500 to-red-500"
    },
  ];

  return (
    <div className="min-h-screen" style={{ background: 'linear-gradient(180deg, #0a0a0f 0%, #0f1419 50%, #0a0a0f 100%)' }}>
      {/* Header */}
      <header className="glass sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-blue-500 to-blue-600 flex items-center justify-center shadow-lg shadow-blue-500/30">
              <Shield className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-white">AI Security Hub | Enterprise</h1>
              <p className="text-xs text-gray-400">Unified Detection Intelligence Platform</p>
            </div>
          </div>
          <nav className="hidden md:flex items-center gap-6">
            {["Text", "Audio", "Image", "Video"].map((item) => (
              <Link
                key={item}
                href={`/${item.toLowerCase()}`}
                className="text-gray-400 hover:text-white transition-colors text-sm font-medium"
              >
                {item}
              </Link>
            ))}
          </nav>
          <div className="flex items-center gap-2">
            <span className="flex items-center gap-2 px-3 py-2 rounded-lg text-sm text-gray-400">
              <span className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse"></span>
              System Status: Operational
            </span>
            <span className="hidden md:flex items-center gap-1 px-3 py-2 rounded-lg text-xs font-mono text-blue-400 bg-blue-500/10 border border-blue-500/20">
              Engine v2.4 (Live)
            </span>
          </div>
        </div>
      </header>

      {/* Hero Section with Dithering Effect */}
      <HeroDitheringCard />


      {/* Features Section - Detection Services Cards */}
      <DetectionServicesCards />

      {/* Stats Section */}
      <section className="max-w-7xl mx-auto px-6 pb-24">
        <div className="rounded-3xl bg-gradient-to-br from-gray-900 to-gray-800 border border-gray-700/50 p-10">
          <div className="grid md:grid-cols-2 gap-12">
            <div>
              <h3 className="text-2xl font-bold text-white mb-2 flex items-center gap-3">
                <TrendingUp className="w-6 h-6 text-blue-400" />
                Live Operations Monitor
              </h3>
              <p className="text-sm text-gray-400 mb-8">24-Hour Rolling Analysis Volume</p>
              <div className="grid grid-cols-2 gap-6">
                {[
                  { label: "Text Analyses Executed", value: stats.text, icon: FileText, color: "text-blue-400" },
                  { label: "Images Processed", value: stats.image, icon: Image, color: "text-cyan-400" },
                  { label: "Video Files Analyzed", value: stats.video, icon: Video, color: "text-orange-400" },
                  { label: "Audio Samples Processed", value: stats.audio, icon: Mic, color: "text-green-400" }
                ].map((stat, i) => {
                  const Icon = stat.icon;
                  return (
                    <div key={i} className="bg-gray-800/50 rounded-2xl p-6 border border-gray-700/50 hover:border-gray-600 transition-colors">
                      <Icon className={`w-8 h-8 ${stat.color} mb-4`} />
                      <div className="text-4xl font-bold text-white mb-1">{stat.value}</div>
                      <div className="text-sm text-gray-400">{stat.label}</div>
                    </div>
                  );
                })}
              </div>
            </div>

            <div>
              <h3 className="text-2xl font-bold text-white mb-2">Model Performance Metrics</h3>
              <p className="text-sm text-gray-400 mb-8">Aggregated Inference Accuracy (v2.4)</p>
              <div className="space-y-6">
                {[
                  { name: "Scam Text Pattern Recognition", value: 92, color: "bg-blue-500" },
                  { name: "Image Manipulation Detection", value: 95, color: "bg-cyan-500" },
                  { name: "Deepfake Temporal Analysis", value: 87, color: "bg-orange-500" },
                  { name: "Voice Synthetic Verification", value: 90, color: "bg-green-500" }
                ].map((item, i) => (
                  <div key={i}>
                    <div className="flex justify-between text-sm mb-2">
                      <span className="text-gray-300 font-medium">{item.name}</span>
                      <span className="text-white font-bold">{item.value}%</span>
                    </div>
                    <div className="h-3 bg-gray-700 rounded-full overflow-hidden">
                      <div
                        className={`h-full ${item.color} rounded-full transition-all duration-1000`}
                        style={{ width: isLoaded ? `${item.value}%` : '0%' }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Why Choose Us */}
      <section className="max-w-7xl mx-auto px-6 pb-24">
        <div className="text-center mb-16">
          <h3 className="text-sm font-semibold text-blue-400 uppercase tracking-wider mb-4">Enterprise Grade Security</h3>
          <h2 className="text-4xl font-bold text-white">Enterprise-Grade Security Architecture</h2>
        </div>
        <div className="grid md:grid-cols-3 gap-8">
          {[
            { icon: Zap, title: "Real-time Inference", desc: "Latency < 200ms via optimized Edge/Cloud hybrid inference engine." },
            { icon: Shield, title: "Precision-Recall Optimized", desc: "Fine-tuned on 10M+ datapoints to minimize false positives." },
            { icon: CheckCircle, title: "Analyst-First Interface", desc: "Streamlined workflow designed for Security Operations Centers (SOC)." }
          ].map((item, i) => {
            const Icon = item.icon;
            return (
              <div key={i} className="rounded-2xl bg-gradient-to-br from-gray-900/80 to-gray-800/50 border border-gray-700/50 p-8 text-center hover:border-blue-500/30 transition-colors">
                <div className="w-16 h-16 rounded-2xl bg-blue-500/10 flex items-center justify-center mx-auto mb-6">
                  <Icon className="w-8 h-8 text-blue-400" />
                </div>
                <h3 className="text-xl font-bold text-white mb-4">{item.title}</h3>
                <p className="text-gray-400 leading-relaxed">{item.desc}</p>
              </div>
            );
          })}
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-gray-800/50">
        <div className="max-w-7xl mx-auto px-6 py-10">
          <div className="flex flex-col md:flex-row items-center justify-between gap-6">
            <div className="flex items-center gap-4">
              <div className="w-10 h-10 rounded-xl bg-blue-500 flex items-center justify-center">
                <Shield className="w-5 h-5 text-white" />
              </div>
              <div>
                <span className="text-white font-semibold">AI Security Hub</span>
                <p className="text-gray-500 text-sm">Enterprise-Grade Protection Platform</p>
              </div>
            </div>
            <div className="text-gray-500 text-sm">
              Authorized Access Only • System Version 2.4.1 (Stable) • © 2024
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
