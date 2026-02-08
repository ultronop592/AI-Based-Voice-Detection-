"use client";

import React, { useState } from 'react';
import { ArrowUpRight, Menu, ArrowRight, Play } from 'lucide-react';

interface NavLink {
    label: string;
    href: string;
    isActive?: boolean;
}

interface Partner {
    logoUrl: string;
    href: string;
    name: string;
}

interface ResponsiveHeroBannerProps {
    logoUrl?: string;
    backgroundImageUrl?: string;
    navLinks?: NavLink[];
    ctaButtonText?: string;
    ctaButtonHref?: string;
    badgeText?: string;
    badgeLabel?: string;
    title?: string;
    titleLine2?: string;
    description?: string;
    primaryButtonText?: string;
    primaryButtonHref?: string;
    secondaryButtonText?: string;
    secondaryButtonHref?: string;
    partnersTitle?: string;
    partners?: Partner[];
}

const ResponsiveHeroBanner: React.FC<ResponsiveHeroBannerProps> = ({
    logoUrl = "https://images.unsplash.com/photo-1614064641938-3bbee52942c7?w=200&h=80&fit=crop",
    backgroundImageUrl = "https://images.unsplash.com/photo-1550751827-4bd374c3f58b?w=1920&h=1080&fit=crop",
    navLinks = [
        { label: "Home", href: "/", isActive: true },
        { label: "Text", href: "/text" },
        { label: "Audio", href: "/audio" },
        { label: "Image", href: "/image" },
        { label: "Video", href: "/video" }
    ],
    ctaButtonText = "API Docs",
    ctaButtonHref = "http://localhost:8000/docs",
    badgeLabel = "AI Powered",
    badgeText = "Advanced Fraud & Deepfake Detection",
    title = "AI Detection Hub",
    titleLine2 = "Protect Yourself",
    description = "Detect fraudulent calls and deepfake media using our advanced machine learning detection systems. Upload text, audio, images, or videos for instant analysis.",
    primaryButtonText = "Start Detection",
    primaryButtonHref = "/text",
    secondaryButtonText = "Watch Demo",
    secondaryButtonHref = "#demo",
    partnersTitle = "Powered by advanced AI technologies",
    partners = [
        { logoUrl: "https://images.unsplash.com/photo-1633356122544-f134324a6cee?w=120&h=36&fit=crop", href: "#", name: "TensorFlow" },
        { logoUrl: "https://images.unsplash.com/photo-1618401471353-b98afee0b2eb?w=120&h=36&fit=crop", href: "#", name: "PyTorch" },
        { logoUrl: "https://images.unsplash.com/photo-1599507593499-a3f7d7d97667?w=120&h=36&fit=crop", href: "#", name: "FastAPI" },
        { logoUrl: "https://images.unsplash.com/photo-1617042375876-a13e36732a04?w=120&h=36&fit=crop", href: "#", name: "Next.js" },
        { logoUrl: "https://images.unsplash.com/photo-1558494949-ef010cbdcc31?w=120&h=36&fit=crop", href: "#", name: "Python" }
    ]
}) => {
    const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

    return (
        <section className="w-full isolate min-h-screen overflow-hidden relative">
            <img
                src={backgroundImageUrl}
                alt="AI Detection Background"
                className="w-full h-full object-cover absolute top-0 right-0 bottom-0 left-0"
            />
            <div className="pointer-events-none absolute inset-0 bg-gradient-to-b from-black/60 via-black/40 to-black/80" />

            <header className="z-10 xl:top-4 relative">
                <div className="mx-6">
                    <div className="flex items-center justify-between pt-4">
                        <a
                            href="/"
                            className="inline-flex items-center gap-3"
                        >
                            <div className="w-10 h-10 rounded-xl bg-blue-500 flex items-center justify-center shadow-lg">
                                <span className="text-xl">🛡️</span>
                            </div>
                            <span className="text-white font-semibold text-lg">AI Detection</span>
                        </a>

                        <nav className="hidden md:flex items-center gap-2">
                            <div className="flex items-center gap-1 rounded-full bg-white/5 px-1 py-1 ring-1 ring-white/10 backdrop-blur">
                                {navLinks.map((link, index) => (
                                    <a
                                        key={index}
                                        href={link.href}
                                        className={`px-3 py-2 text-sm font-medium hover:text-white font-sans transition-colors ${link.isActive ? 'text-white/90' : 'text-white/80'
                                            }`}
                                    >
                                        {link.label}
                                    </a>
                                ))}
                                <a
                                    href={ctaButtonHref}
                                    target="_blank"
                                    className="ml-1 inline-flex items-center gap-2 rounded-full bg-white px-3.5 py-2 text-sm font-medium text-neutral-900 hover:bg-white/90 font-sans transition-colors"
                                >
                                    {ctaButtonText}
                                    <ArrowUpRight className="h-4 w-4" />
                                </a>
                            </div>
                        </nav>

                        <button
                            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
                            className="md:hidden inline-flex h-10 w-10 items-center justify-center rounded-full bg-white/10 ring-1 ring-white/15 backdrop-blur"
                            aria-expanded={mobileMenuOpen}
                            aria-label="Toggle menu"
                        >
                            <Menu className="h-5 w-5 text-white/90" />
                        </button>
                    </div>

                    {/* Mobile Menu */}
                    {mobileMenuOpen && (
                        <div className="md:hidden mt-4 rounded-2xl bg-black/80 backdrop-blur-xl p-4 ring-1 ring-white/10">
                            {navLinks.map((link, index) => (
                                <a
                                    key={index}
                                    href={link.href}
                                    className="block px-4 py-3 text-white/80 hover:text-white transition-colors"
                                >
                                    {link.label}
                                </a>
                            ))}
                        </div>
                    )}
                </div>
            </header>

            <div className="z-10 relative">
                <div className="sm:pt-28 md:pt-32 lg:pt-40 max-w-7xl mx-auto pt-28 px-6 pb-16">
                    <div className="mx-auto max-w-3xl text-center">
                        <div className="mb-6 inline-flex items-center gap-3 rounded-full bg-blue-500/20 px-2.5 py-2 ring-1 ring-blue-500/30 backdrop-blur animate-fade-slide-in-1">
                            <span className="inline-flex items-center text-xs font-medium text-white bg-blue-500 rounded-full py-0.5 px-2 font-sans">
                                {badgeLabel}
                            </span>
                            <span className="text-sm font-medium text-white/90 font-sans">
                                {badgeText}
                            </span>
                        </div>

                        <h1 className="sm:text-5xl md:text-6xl lg:text-7xl leading-tight text-4xl text-white tracking-tight font-serif font-normal animate-fade-slide-in-2">
                            {title}
                            <br className="hidden sm:block" />
                            <span className="text-blue-400">{titleLine2}</span>
                        </h1>

                        <p className="sm:text-lg animate-fade-slide-in-3 text-base text-white/80 max-w-2xl mt-6 mx-auto">
                            {description}
                        </p>

                        <div className="flex flex-col sm:flex-row sm:gap-4 mt-10 gap-3 items-center justify-center animate-fade-slide-in-4">
                            <a
                                href={primaryButtonHref}
                                className="inline-flex items-center gap-2 bg-blue-500 hover:bg-blue-600 text-sm font-medium text-white ring-blue-500/30 ring-1 rounded-full py-3 px-6 font-sans transition-all shadow-lg shadow-blue-500/30"
                            >
                                {primaryButtonText}
                                <ArrowRight className="h-4 w-4" />
                            </a>
                            <a
                                href={secondaryButtonHref}
                                className="inline-flex items-center gap-2 rounded-full bg-white/10 hover:bg-white/20 px-5 py-3 text-sm font-medium text-white/90 hover:text-white font-sans transition-colors ring-1 ring-white/10"
                            >
                                {secondaryButtonText}
                                <Play className="w-4 h-4" />
                            </a>
                        </div>
                    </div>

                    <div className="mx-auto mt-20 max-w-5xl">
                        <p className="animate-fade-slide-in-1 text-sm text-white/70 text-center">
                            {partnersTitle}
                        </p>
                        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 animate-fade-slide-in-2 text-white/70 mt-6 items-center justify-items-center gap-6">
                            {partners.map((partner, index) => (
                                <div
                                    key={index}
                                    className="flex items-center gap-2 px-4 py-2 rounded-lg bg-white/5 ring-1 ring-white/10 opacity-70 hover:opacity-100 transition-opacity"
                                >
                                    <span className="text-white/80 text-sm">{partner.name}</span>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            </div>
        </section>
    );
};

export default ResponsiveHeroBanner;
