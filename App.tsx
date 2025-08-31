
import React from 'react';
import { STATS_DATA, MODULES_DATA, RESOURCES_DATA, ACCORDION_DATA } from './constants';
import { Stat, Module, Resource } from './types';
import Accordion from './components/Accordion';
import ScrollProgressBar from './components/ScrollProgressBar';
import FloatingActionButton from './components/FloatingActionButton';

const Header: React.FC = () => (
    <header className="header bg-white/10 backdrop-blur-lg border-b border-white/20 sticky top-0 z-50 transition-all duration-300">
        <div className="container mx-auto px-5">
            <nav className="flex justify-between items-center py-4">
                <a href="#" className="logo text-2xl font-bold text-white">🤖 GenAI Mastery</a>
                <ul className="nav-links hidden md:flex list-none gap-8">
                    <li><a href="#modules" className="text-white relative hover:text-yellow-300 after:content-[''] after:absolute after:bottom-[-5px] after:left-0 after:w-0 after:h-0.5 after:bg-yellow-300 after:transition-all after:duration-300 hover:after:w-full">Modules</a></li>
                    <li><a href="#progress" className="text-white relative hover:text-yellow-300 after:content-[''] after:absolute after:bottom-[-5px] after:left-0 after:w-0 after:h-0.5 after:bg-yellow-300 after:transition-all after:duration-300 hover:after:w-full">Progress</a></li>
                    <li><a href="#resources" className="text-white relative hover:text-yellow-300 after:content-[''] after:absolute after:bottom-[-5px] after:left-0 after:w-0 after:h-0.5 after:bg-yellow-300 after:transition-all after:duration-300 hover:after:w-full">Resources</a></li>
                </ul>
            </nav>
        </div>
    </header>
);

const Hero: React.FC = () => (
    <section className="hero py-24 text-center text-white">
        <div className="container mx-auto px-5">
            <h1 className="text-5xl md:text-6xl font-extrabold mb-4 drop-shadow-lg bg-gradient-to-r from-white to-yellow-300 bg-clip-text text-transparent">Master Generative AI</h1>
            <p className="text-lg opacity-90 max-w-xl mx-auto mb-8">A comprehensive, progressive learning path from beginner to expert level</p>
            <a href="#modules" className="cta-button inline-block px-8 py-3 bg-gradient-to-r from-red-500 to-orange-400 text-white rounded-full font-semibold shadow-lg hover:shadow-xl hover:-translate-y-1 transition-all duration-300">Start Learning Journey</a>
        </div>
    </section>
);

const Stats: React.FC = () => (
    <section className="stats container mx-auto px-5 py-8 grid grid-cols-2 md:grid-cols-4 gap-6">
        {STATS_DATA.map((stat: Stat, index: number) => (
            <div key={index} className="stat-card text-center p-8 bg-gradient-to-br from-pink-400 to-red-500 rounded-2xl text-white shadow-lg hover:-translate-y-2 transition-transform duration-300">
                <div className="stat-number text-4xl font-bold mb-2">{stat.number}</div>
                <div className="stat-label opacity-90">{stat.label}</div>
            </div>
        ))}
    </section>
);

const Modules: React.FC = () => (
    <section id="modules" className="py-16">
        <h2 className="text-center text-4xl font-bold mb-12 text-gray-800">Learning Modules</h2>
        <div className="container mx-auto px-5 grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            {MODULES_DATA.map((module: Module, index: number) => (
                <div key={index} className="module-card bg-white rounded-2xl p-8 shadow-lg hover:shadow-xl hover:-translate-y-2 transition-all duration-300 border border-gray-100 relative overflow-hidden before:content-[''] before:absolute before:top-0 before:left-0 before:right-0 before:h-1 before:bg-gradient-to-r from-indigo-500 to-purple-600">
                    <div className="flex items-start justify-between mb-4">
                        <div className="flex items-center">
                            <span className="text-3xl mr-4">{module.icon}</span>
                            <div>
                                <h3 className="text-xl font-bold text-gray-800">{module.title}</h3>
                            </div>
                        </div>
                        <span className="bg-indigo-100 text-indigo-700 px-3 py-1 rounded-full text-xs font-semibold whitespace-nowrap">{module.duration}</span>
                    </div>
                    <p className="text-gray-600 mb-6">{module.description}</p>
                    <ul className="space-y-2 mb-6">
                        {module.skills.map(skill => (
                            <li key={skill} className="relative pl-6 text-gray-700 before:content-['✓'] before:absolute before:left-0 before:text-green-500 before:font-bold">{skill}</li>
                        ))}
                    </ul>
                    <span className={`difficulty-badge inline-block px-4 py-1.5 rounded-full text-xs font-bold uppercase tracking-wider ${module.difficultyClass}`}>{module.difficulty}</span>
                </div>
            ))}
        </div>
    </section>
);

const ProgressTracker: React.FC = () => {
    const steps = ['Foundations', 'Core Concepts', 'Intermediate', 'Advanced', 'Expert'];
    const completedStep = 1;
    return (
        <section id="progress" className="bg-gradient-to-br from-indigo-500 to-purple-600 py-16 text-white rounded-3xl my-16 mx-5">
            <div className="container mx-auto px-5 text-center">
                <h2 className="text-4xl font-bold mb-8">Your Learning Journey</h2>
                <div className="relative flex justify-between items-center max-w-4xl mx-auto mb-8 px-4">
                     <div className="absolute top-1/2 left-0 w-full h-0.5 bg-white/30 -translate-y-1/2"></div>
                    {steps.map((step, index) => (
                        <div key={step} className="progress-step relative z-10 flex flex-col items-center">
                            <div className={`step-circle w-10 h-10 rounded-full flex items-center justify-center font-bold mb-2 transition-colors duration-300 ${index < completedStep ? 'bg-green-500' : 'bg-white/20'}`}>
                                {index < completedStep ? '✓' : index + 1}
                            </div>
                            <div className="step-label text-sm text-center">{step}</div>
                        </div>
                    ))}
                </div>
                <p className="opacity-90">Track your progress through each module and celebrate your achievements!</p>
            </div>
        </section>
    );
};

const Resources: React.FC = () => (
    <section id="resources" className="bg-gray-50 py-16 rounded-3xl my-16 mx-5">
        <div className="container mx-auto px-5">
            <h2 className="text-center text-4xl font-bold mb-12 text-gray-800">Learning Resources</h2>
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
                {RESOURCES_DATA.map((resource: Resource, index: number) => (
                    <div key={index} className="resource-card bg-white p-8 rounded-2xl shadow-md hover:shadow-lg hover:-translate-y-1 transition-all duration-300">
                        <span className="resource-type bg-gradient-to-r from-indigo-500 to-purple-600 text-white px-4 py-1.5 rounded-full text-xs font-semibold inline-block mb-4">{resource.type}</span>
                        <h3 className="text-xl font-bold text-gray-800 mb-2">{resource.title}</h3>
                        <p className="text-gray-600">{resource.description}</p>
                    </div>
                ))}
            </div>
        </div>
    </section>
);


const DetailedModules: React.FC = () => (
    <section className="my-16 container mx-auto px-5">
         <h2 className="text-center text-4xl font-bold mb-12 text-gray-800">Hands-On Projects</h2>
        {ACCORDION_DATA.map((item, index) => (
            <Accordion key={index} item={item} startOpen={index === 0} />
        ))}
    </section>
);


const Footer: React.FC = () => (
    <footer className="footer bg-gray-800 text-white py-12 text-center">
        <div className="container mx-auto px-5">
            <p className="opacity-80 mb-4">&copy; 2024 GenAI Mastery Guide. All rights reserved.</p>
            <div className="flex justify-center gap-8 mb-8">
                <a href="#" className="hover:text-yellow-300 transition-colors">About</a>
                <a href="#" className="hover:text-yellow-300 transition-colors">Contact</a>
                <a href="#" className="hover:text-yellow-300 transition-colors">Privacy Policy</a>
            </div>
        </div>
    </footer>
);

export default function App() {
    return (
        <div className="bg-gradient-to-br from-indigo-500 to-purple-600 min-h-screen">
            <ScrollProgressBar />
            <Header />
            <main>
                <Hero />
                <div className="main-content bg-white rounded-t-3xl md:rounded-t-[50px] -mt-8 relative z-10 pt-12 pb-8">
                    <Stats />
                    <Modules />
                    <div className="container mx-auto">
                        <ProgressTracker />
                        <Resources />
                    </div>
                    <DetailedModules />
                </div>
            </main>
            <Footer />
            <FloatingActionButton />
        </div>
    );
}
