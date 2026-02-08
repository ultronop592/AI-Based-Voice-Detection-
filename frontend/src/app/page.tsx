"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { HeroDitheringCard } from "@/components/ui/hero-dithering-card";
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
      title: "Text Fraud Detection",
      description: "Analyze call transcripts and messages to detect fraud patterns using advanced NLP and machine learning",
      icon: FileText,
      href: "/text",
      stat: "92% Accuracy",
      color: "from-blue-500 to-cyan-500"
    },
    {
      title: "Audio Analysis",
      description: "Record or upload audio files for real-time fraud call detection with speech recognition",
      icon: Mic,
      href: "/audio",
      stat: "Real-time",
      color: "from-green-500 to-emerald-500"
    },
    {
      title: "Image Deepfake Detection",
      description: "Upload images to detect AI-generated deepfakes using our trained EfficientNet-B0 model",
      icon: Image,
      href: "/image",
      stat: "95% Accuracy",
      color: "from-purple-500 to-violet-500"
    },
    {
      title: "Video Analysis",
      description: "Analyze videos for temporal inconsistencies, face manipulation, and deepfake indicators",
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
              <h1 className="text-xl font-bold text-white">AI Detection Hub</h1>
              <p className="text-xs text-gray-400">Fraud & Deepfake Analysis</p>
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
              <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></span>
              Online
            </span>
          </div>
        </div>
      </header>

      {/* Hero Section with Dithering Effect */}
      <HeroDitheringCard />

      {/* Features Section */}
      <section id="features" className="max-w-7xl mx-auto px-6 pb-24">
        <div className="text-center mb-16">
          <h3 className="text-sm font-semibold text-blue-400 uppercase tracking-wider mb-4">Features</h3>
          <h2 className="text-4xl font-bold text-white mb-4">Detection Services</h2>
          <p className="text-gray-400 max-w-2xl mx-auto text-lg">
            Choose a detection type to analyze your content
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-8">
          {features.map((feature, index) => {
            const Icon = feature.icon;
            return (
              <Link key={feature.href} href={feature.href}>
                <div className={`group relative overflow-hidden rounded-3xl bg-gradient-to-br from-gray-900/80 to-gray-800/50 border border-gray-700/50 p-8 h-full cursor-pointer transition-all duration-500 hover:border-blue-500/50 hover:shadow-2xl hover:shadow-blue-500/10 hover:-translate-y-2 opacity-0 animate-slide-up stagger-${index + 1}`}>
                  <div className={`absolute inset-0 bg-gradient-to-br ${feature.color} opacity-0 group-hover:opacity-5 transition-opacity duration-500`} />

                  <div className="relative z-10">
                    <div className="flex items-start justify-between mb-6">
                      <div className={`w-16 h-16 rounded-2xl bg-gradient-to-br ${feature.color} flex items-center justify-center shadow-lg`}>
                        <Icon className="w-8 h-8 text-white" />
                      </div>
                      <span className="text-xs px-4 py-2 bg-white/10 text-white rounded-full font-medium">
                        {feature.stat}
                      </span>
                    </div>

                    <h4 className="text-2xl font-bold text-white mb-4 group-hover:text-blue-300 transition-colors">
                      {feature.title}
                    </h4>
                    <p className="text-gray-400 mb-8 leading-relaxed text-base">
                      {feature.description}
                    </p>

                    <div className="flex items-center gap-2 text-blue-400 font-medium group-hover:gap-4 transition-all">
                      <span>Start Detection</span>
                      <ArrowRight className="w-5 h-5" />
                    </div>
                  </div>
                </div>
              </Link>
            );
          })}
        </div>
      </section>

      {/* Stats Section */}
      <section className="max-w-7xl mx-auto px-6 pb-24">
        <div className="rounded-3xl bg-gradient-to-br from-gray-900 to-gray-800 border border-gray-700/50 p-10">
          <div className="grid md:grid-cols-2 gap-12">
            <div>
              <h3 className="text-2xl font-bold text-white mb-8 flex items-center gap-3">
                <TrendingUp className="w-6 h-6 text-blue-400" />
                Analysis Statistics
              </h3>
              <div className="grid grid-cols-2 gap-6">
                {[
                  { label: "Text Scans", value: stats.text, icon: FileText, color: "text-blue-400" },
                  { label: "Images", value: stats.image, icon: Image, color: "text-purple-400" },
                  { label: "Videos", value: stats.video, icon: Video, color: "text-orange-400" },
                  { label: "Audio Files", value: stats.audio, icon: Mic, color: "text-green-400" }
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
              <h3 className="text-2xl font-bold text-white mb-8">Model Performance</h3>
              <div className="space-y-6">
                {[
                  { name: "Text Detection", value: 92, color: "bg-blue-500" },
                  { name: "Image Detection", value: 95, color: "bg-purple-500" },
                  { name: "Video Analysis", value: 87, color: "bg-orange-500" },
                  { name: "Audio Processing", value: 90, color: "bg-green-500" }
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
          <h3 className="text-sm font-semibold text-blue-400 uppercase tracking-wider mb-4">Why Choose Us</h3>
          <h2 className="text-4xl font-bold text-white">Powerful AI Detection</h2>
        </div>
        <div className="grid md:grid-cols-3 gap-8">
          {[
            { icon: Zap, title: "Lightning Fast", desc: "Get detection results in seconds with our optimized AI models" },
            { icon: Shield, title: "High Accuracy", desc: "95%+ accuracy on deepfake and fraud detection" },
            { icon: CheckCircle, title: "Easy to Use", desc: "Simple upload interface - no technical knowledge required" }
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
                <span className="text-white font-semibold">AI Detection Hub</span>
                <p className="text-gray-500 text-sm">Advanced Fraud & Deepfake Detection</p>
              </div>
            </div>
            <div className="text-gray-500 text-sm">
              Built with FastAPI + Next.js • © 2024
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
