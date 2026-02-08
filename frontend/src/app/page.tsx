"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import ResponsiveHeroBanner from "@/components/ui/responsive-hero-banner";
import { FileText, Mic, Image, Video, ArrowRight, Zap, Shield, Clock } from "lucide-react";

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
      title: "Text Analysis",
      description: "Analyze call transcripts to detect fraud patterns",
      icon: FileText,
      href: "/text",
      stat: "92% Accuracy"
    },
    {
      title: "Audio Detection",
      description: "Record or upload audio for fraud analysis",
      icon: Mic,
      href: "/audio",
      stat: "Real-time"
    },
    {
      title: "Image Detection",
      description: "Detect AI-generated deepfake images",
      icon: Image,
      href: "/image",
      stat: "95% Accuracy"
    },
    {
      title: "Video Analysis",
      description: "Analyze videos for manipulation artifacts",
      icon: Video,
      href: "/video",
      stat: "Heuristic AI"
    },
  ];

  return (
    <div className="min-h-screen" style={{ background: '#0F1419' }}>
      {/* Hero Banner */}
      <ResponsiveHeroBanner
        title="AI Detection Hub"
        titleLine2="Protect Yourself"
        badgeLabel="AI Powered"
        badgeText="Advanced Fraud & Deepfake Detection"
        description="Detect fraudulent calls and deepfake media using our advanced machine learning detection systems. Upload text, audio, images, or videos for instant analysis."
        primaryButtonText="Start Detection"
        primaryButtonHref="/text"
        secondaryButtonText="Learn More"
        secondaryButtonHref="#features"
        backgroundImageUrl="https://images.unsplash.com/photo-1550751827-4bd374c3f58b?w=1920&h=1080&fit=crop"
      />

      {/* Features Section */}
      <section id="features" className="max-w-7xl mx-auto px-6 py-20">
        <div className="text-center mb-12">
          <h2 className="text-3xl font-bold text-white mb-4">Detection Services</h2>
          <p className="text-gray-400 max-w-2xl mx-auto">
            Choose a detection type to analyze your content for fraud or deepfakes
          </p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
          {features.map((feature, index) => {
            const Icon = feature.icon;
            return (
              <Link key={feature.href} href={feature.href}>
                <div className={`card-3d gradient-border p-6 h-full cursor-pointer opacity-0 animate-slide-up stagger-${index + 1}`}>
                  <div className="w-12 h-12 rounded-xl bg-blue-500/20 flex items-center justify-center mb-4">
                    <Icon className="w-6 h-6 text-blue-400" />
                  </div>
                  <h4 className="text-xl font-bold text-white mb-2">{feature.title}</h4>
                  <p className="text-gray-400 text-sm mb-4">{feature.description}</p>
                  <div className="flex items-center justify-between">
                    <span className="text-xs px-3 py-1 bg-blue-500/20 text-blue-400 rounded-full">
                      {feature.stat}
                    </span>
                    <ArrowRight className="w-5 h-5 text-blue-400" />
                  </div>
                </div>
              </Link>
            );
          })}
        </div>
      </section>

      {/* Stats Section */}
      <section className="max-w-7xl mx-auto px-6 py-16">
        <div className="gradient-border p-8">
          <div className="grid md:grid-cols-2 gap-8">
            {/* Stats Cards */}
            <div className="grid grid-cols-2 gap-4">
              {[
                { label: "Text Scans", value: stats.text, icon: FileText },
                { label: "Images Analyzed", value: stats.image, icon: Image },
                { label: "Videos Checked", value: stats.video, icon: Video },
                { label: "Audio Files", value: stats.audio, icon: Mic }
              ].map((stat, i) => {
                const Icon = stat.icon;
                return (
                  <div key={i} className="stat-card bg-gray-800/50 rounded-xl p-6 text-center border border-gray-700">
                    <Icon className="w-8 h-8 text-blue-400 mx-auto mb-3" />
                    <div className="text-3xl font-bold text-white">{stat.value}</div>
                    <div className="text-sm text-gray-400">{stat.label}</div>
                  </div>
                );
              })}
            </div>

            {/* Performance Bars */}
            <div className="space-y-4">
              <h4 className="text-lg font-semibold text-white mb-4">Model Performance</h4>
              {[
                { name: "Text Fraud Detection", value: 92, color: "bg-blue-500" },
                { name: "Image Deepfake Detection", value: 95, color: "bg-blue-400" },
                { name: "Video Analysis", value: 87, color: "bg-blue-300" },
                { name: "Audio Processing", value: 90, color: "bg-blue-200" }
              ].map((item, i) => (
                <div key={i}>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-gray-300">{item.name}</span>
                    <span className="text-white font-medium">{item.value}%</span>
                  </div>
                  <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
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
      </section>

      {/* Why Choose Us */}
      <section className="max-w-7xl mx-auto px-6 py-16">
        <div className="text-center mb-12">
          <h2 className="text-3xl font-bold text-white mb-4">Why Choose Us</h2>
        </div>
        <div className="grid md:grid-cols-3 gap-6">
          {[
            { icon: Zap, title: "Fast Detection", desc: "Get results in seconds with our optimized AI models" },
            { icon: Shield, title: "High Accuracy", desc: "95%+ accuracy on deepfake and fraud detection" },
            { icon: Clock, title: "Real-time", desc: "Process audio and video in real-time streaming" }
          ].map((item, i) => {
            const Icon = item.icon;
            return (
              <div key={i} className="gradient-border p-6 text-center">
                <div className="w-14 h-14 rounded-2xl bg-blue-500/20 flex items-center justify-center mx-auto mb-4">
                  <Icon className="w-7 h-7 text-blue-400" />
                </div>
                <h3 className="text-lg font-semibold text-white mb-2">{item.title}</h3>
                <p className="text-gray-400 text-sm">{item.desc}</p>
              </div>
            );
          })}
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-gray-800">
        <div className="max-w-7xl mx-auto px-6 py-8">
          <div className="flex flex-col md:flex-row items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-blue-500 flex items-center justify-center">
                <span className="text-lg">🛡️</span>
              </div>
              <span className="text-gray-400 text-sm">AI Detection Hub</span>
            </div>
            <div className="text-gray-500 text-sm text-center">
              Built with FastAPI + Next.js • © 2024
            </div>
            <div className="flex items-center gap-4">
              <a href="http://localhost:8000/docs" target="_blank" className="text-gray-400 hover:text-white text-sm transition-colors">
                API Docs
              </a>
              <a href="http://localhost:8000/health" target="_blank" className="text-gray-400 hover:text-white text-sm transition-colors">
                Health
              </a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
