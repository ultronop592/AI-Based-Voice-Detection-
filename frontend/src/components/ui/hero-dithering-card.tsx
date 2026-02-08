"use client";

import { ArrowRight, Shield, FileText, Mic, Image, Video } from "lucide-react"
import { useState, Suspense, lazy } from "react"
import Link from "next/link"

const Dithering = lazy(() =>
    import("@paper-design/shaders-react").then((mod) => ({ default: mod.Dithering }))
)

export function HeroDitheringCard() {
    const [isHovered, setIsHovered] = useState(false)

    return (
        <section className="py-12 w-full flex justify-center items-center px-4 md:px-6">
            <div
                className="w-full max-w-7xl relative"
                onMouseEnter={() => setIsHovered(true)}
                onMouseLeave={() => setIsHovered(false)}
            >
                <div className="relative overflow-hidden rounded-[48px] border border-gray-700/50 bg-gray-900/80 shadow-2xl min-h-[600px] md:min-h-[650px] flex flex-col items-center justify-center duration-500">
                    {/* Dithering Shader Background */}
                    <Suspense fallback={<div className="absolute inset-0 bg-blue-500/5" />}>
                        <div className="absolute inset-0 z-0 pointer-events-none opacity-50 dark:opacity-40 mix-blend-screen">
                            <Dithering
                                colorBack="#00000000"
                                colorFront="#3B82F6"
                                shape="warp"
                                type="4x4"
                                speed={isHovered ? 0.6 : 0.2}
                                className="size-full"
                                minPixelRatio={1}
                            />
                        </div>
                    </Suspense>

                    <div className="relative z-10 px-6 max-w-4xl mx-auto text-center flex flex-col items-center">

                        {/* Badge */}
                        <div className="mb-8 inline-flex items-center gap-2 rounded-full border border-blue-500/30 bg-blue-500/10 px-5 py-2 text-sm font-medium text-blue-400 backdrop-blur-sm">
                            <span className="relative flex h-2 w-2">
                                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-blue-400 opacity-75"></span>
                                <span className="relative inline-flex rounded-full h-2 w-2 bg-blue-400"></span>
                            </span>
                            AI-Powered Protection
                        </div>

                        {/* Headline */}
                        <h1 className="text-5xl md:text-7xl lg:text-8xl font-bold tracking-tight text-white mb-8 leading-[1.05]">
                            Detect Fraud &<br />
                            <span className="bg-gradient-to-r from-blue-400 via-cyan-400 to-blue-500 bg-clip-text text-transparent">
                                Deepfakes Instantly
                            </span>
                        </h1>

                        {/* Description */}
                        <p className="text-gray-400 text-lg md:text-xl max-w-2xl mb-12 leading-relaxed">
                            Protect yourself from fraudulent calls and AI-generated fake media using our
                            advanced machine learning detection systems. Upload and analyze in seconds.
                        </p>

                        {/* CTA Button */}
                        <Link href="/text">
                            <button className="group relative inline-flex h-14 items-center justify-center gap-3 overflow-hidden rounded-full bg-gradient-to-r from-blue-500 to-cyan-500 px-12 text-base font-semibold text-white transition-all duration-300 hover:from-blue-600 hover:to-cyan-600 hover:scale-105 active:scale-95 shadow-lg shadow-blue-500/30 hover:shadow-blue-500/50 hover:ring-4 hover:ring-blue-500/20">
                                <span className="relative z-10">Start Detection</span>
                                <ArrowRight className="h-5 w-5 relative z-10 transition-transform duration-300 group-hover:translate-x-1" />
                            </button>
                        </Link>

                        {/* Feature Pills */}
                        <div className="flex flex-wrap justify-center gap-3 mt-12">
                            {[
                                { icon: FileText, label: "Text Analysis" },
                                { icon: Mic, label: "Audio Detection" },
                                { icon: Image, label: "Image Deepfake" },
                                { icon: Video, label: "Video Analysis" },
                            ].map((feature, i) => {
                                const Icon = feature.icon;
                                return (
                                    <div
                                        key={i}
                                        className="flex items-center gap-2 px-4 py-2 rounded-full bg-white/5 border border-white/10 text-gray-300 text-sm backdrop-blur-sm"
                                    >
                                        <Icon className="w-4 h-4 text-blue-400" />
                                        {feature.label}
                                    </div>
                                );
                            })}
                        </div>
                    </div>
                </div>
            </div>
        </section>
    )
}
