"use client";

import { ArrowRight, FileText, Mic, Image as ImageIcon, Video, Shield } from "lucide-react";
import Image from "next/image";
import Link from "next/link";

const servicesData = [
    {
        category: "TEXT",
        description:
            "Analyze call transcripts and messages to detect fraud patterns using advanced NLP and machine learning algorithms.",
        image:
            "https://images.unsplash.com/photo-1516321318423-f06f85e504b3?w=800&q=80",
        stat: "92% Accuracy",
        href: "/text",
        title: "Text Fraud Detection",
        icon: FileText,
    },
    {
        category: "AUDIO",
        description:
            "Record or upload audio files for real-time fraud call detection with advanced speech recognition technology.",
        image:
            "https://images.unsplash.com/photo-1478737270239-2f02b77fc618?w=800&q=80",
        stat: "Real-time",
        href: "/audio",
        title: "Audio Analysis",
        icon: Mic,
    },
    {
        category: "IMAGE",
        description:
            "Upload images to detect AI-generated deepfakes using our trained EfficientNet-B0 neural network model.",
        image:
            "https://images.unsplash.com/photo-1633412802994-5c058f151b66?w=800&q=80",
        stat: "95% Accuracy",
        href: "/image",
        title: "Image Deepfake Detection",
        icon: ImageIcon,
    },
    {
        category: "VIDEO",
        description:
            "Analyze videos for temporal inconsistencies, face manipulation, and other deepfake indicators using heuristic AI.",
        image:
            "https://images.unsplash.com/photo-1492619375914-88005aa9e8fb?w=800&q=80",
        stat: "Heuristic AI",
        href: "/video",
        title: "Video Analysis",
        icon: Video,
    },
];

export default function DetectionServicesCards() {
    return (
        <section className="px-4 py-12 sm:py-16 md:py-20" style={{ background: 'transparent' }}>
            <div className="mx-auto max-w-7xl">
                <div className="mb-8 text-center sm:mb-12">
                    <p className="mb-3 font-medium text-blue-400 text-xs uppercase tracking-wider sm:mb-4">
                        FEATURES
                    </p>
                    <h2 className="font-bold text-2xl text-white tracking-tight sm:text-3xl md:text-5xl">
                        Detection Services
                    </h2>
                    <p className="mt-4 text-gray-400 max-w-2xl mx-auto">
                        Choose a detection type to analyze your content for fraud or deepfakes
                    </p>
                </div>
                <div className="grid gap-6 sm:gap-8 md:grid-cols-2">
                    {servicesData.map((service, index) => {
                        const Icon = service.icon;
                        return (
                            <Link href={service.href} key={index}>
                                <div className="group cursor-pointer rounded-2xl border border-gray-700/50 bg-gray-900/50 backdrop-blur-sm transition-all duration-300 hover:border-blue-500/50 hover:shadow-xl hover:shadow-blue-500/10 hover:-translate-y-1 overflow-hidden">
                                    <div className="p-0">
                                        <div className="relative mb-0">
                                            <Image
                                                alt={service.title}
                                                className="aspect-video h-48 w-full object-cover sm:h-56 opacity-80 group-hover:opacity-100 transition-opacity"
                                                height={400}
                                                src={service.image}
                                                width={800}
                                            />
                                            {/* Category Badge */}
                                            <div className="absolute top-3 left-3 flex items-center gap-2">
                                                <span className="rounded-full bg-blue-500/90 px-3 py-1 font-medium text-xs text-white uppercase backdrop-blur-sm">
                                                    #{service.category}
                                                </span>
                                            </div>
                                            {/* Stat Badge */}
                                            <div className="absolute top-3 right-3">
                                                <span className="rounded-full bg-white/90 px-3 py-1 font-medium text-xs text-gray-900 backdrop-blur-sm">
                                                    {service.stat}
                                                </span>
                                            </div>
                                            {/* Gradient Overlay */}
                                            <div className="absolute inset-0 bg-gradient-to-t from-gray-900 via-transparent to-transparent" />
                                        </div>
                                        <div className="px-5 pb-5 pt-4">
                                            <div className="flex items-center gap-3 mb-3">
                                                <div className="w-10 h-10 rounded-xl bg-blue-500/20 flex items-center justify-center">
                                                    <Icon className="w-5 h-5 text-blue-400" />
                                                </div>
                                                <h3 className="font-bold text-lg text-white tracking-tight sm:text-xl group-hover:text-blue-300 transition-colors">
                                                    {service.title}
                                                </h3>
                                            </div>
                                            <p className="mb-5 text-gray-400 text-sm leading-relaxed">
                                                {service.description}
                                            </p>
                                            {/* Read More Link */}
                                            <div className="flex items-center justify-between">
                                                <div className="group/btn flex items-center gap-2 font-medium text-blue-400 text-sm transition-colors hover:text-blue-300">
                                                    <span className="overflow-hidden rounded-lg border border-gray-700 p-2 transition-all duration-300 group-hover/btn:bg-blue-500 group-hover/btn:border-blue-500">
                                                        <ArrowRight className="h-4 w-4 transition-transform duration-300 group-hover/btn:translate-x-1" />
                                                    </span>
                                                    Start Detection
                                                </div>
                                                <div className="flex items-center gap-2">
                                                    <Shield className="w-4 h-4 text-green-500" />
                                                    <span className="text-xs text-gray-500">Secure</span>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </Link>
                        );
                    })}
                </div>
            </div>
        </section>
    );
}
