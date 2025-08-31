
import React, { useState, useEffect } from 'react';

const FloatingActionButton: React.FC = () => {
    const [isVisible, setIsVisible] = useState(false);

    const toggleVisibility = () => {
        if (window.pageYOffset > 300) {
            setIsVisible(true);
        } else {
            setIsVisible(false);
        }
    };

    const scrollToTop = () => {
        window.scrollTo({
            top: 0,
            behavior: 'smooth'
        });
    };

    useEffect(() => {
        window.addEventListener('scroll', toggleVisibility);
        return () => {
            window.removeEventListener('scroll', toggleVisibility);
        };
    }, []);

    return (
        <button
            onClick={scrollToTop}
            className={`fixed bottom-8 right-8 w-14 h-14 bg-gradient-to-br from-red-500 to-orange-400 rounded-full flex items-center justify-center text-white text-2xl shadow-lg hover:-translate-y-1 hover:shadow-xl transition-all duration-300 z-50 ${isVisible ? 'opacity-100' : 'opacity-0'}`}
        >
            ▲
        </button>
    );
};

export default FloatingActionButton;
