
import React, { useState } from 'react';
import { AccordionItemData, Project } from '../types';

interface AccordionProps {
  item: AccordionItemData;
  startOpen?: boolean;
}

const Accordion: React.FC<AccordionProps> = ({ item, startOpen = false }) => {
  const [isOpen, setIsOpen] = useState(startOpen);
  const [visibleCodeBlocks, setVisibleCodeBlocks] = useState<Record<string, boolean>>({});

  const toggleAccordion = () => setIsOpen(!isOpen);
  const toggleCodeBlock = (id: string) => {
    setVisibleCodeBlocks(prev => ({ ...prev, [id]: !prev[id] }));
  };

  return (
    <div className="accordion bg-white rounded-lg shadow-md mb-4 overflow-hidden">
      <button
        onClick={toggleAccordion}
        className="accordion-header w-full text-left bg-gradient-to-r from-indigo-500 to-purple-600 text-white p-4 flex justify-between items-center font-semibold hover:brightness-110 transition-all duration-300"
      >
        <span>{item.icon} {item.title}</span>
        <span className={`accordion-toggle text-xl transition-transform duration-300 ${isOpen ? 'rotate-180' : ''}`}>▼</span>
      </button>
      {isOpen && (
        <div className="accordion-content p-6 animate-fadeInUp">
          <h4 className="font-bold text-lg mb-2">Learning Objectives</h4>
          <p className="text-gray-700 mb-4">{item.learningObjectives}</p>
          
          <h4 className="font-bold text-lg mb-2">Key Topics</h4>
          <ul className="list-disc list-inside space-y-1 mb-6">
            {item.keyTopics.map((topic, index) => (
              <li key={index} className="text-gray-700">{topic}</li>
            ))}
          </ul>

          <h4 className="font-bold text-xl text-gray-800 border-b pb-2 mb-4">Hands-On Projects</h4>
          {item.projects.map((project: Project) => (
            <div key={project.codeId} className="project-section py-4 border-t">
              <h5 className="text-lg font-semibold text-gray-800">{project.title}</h5>
              <p className="text-gray-600 mt-2">{project.description}</p>
              {project.code && (
                <>
                  <button 
                    onClick={() => toggleCodeBlock(project.codeId)}
                    className="mt-4 px-4 py-2 bg-indigo-500 text-white rounded-lg hover:bg-indigo-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-400 transition-colors"
                  >
                    {visibleCodeBlocks[project.codeId] ? 'Hide' : 'Show'} Code Template
                  </button>
                  {visibleCodeBlocks[project.codeId] && (
                    <div className="code-block mt-4 bg-gray-900 text-gray-300 rounded-lg p-4 overflow-x-auto font-mono text-sm">
                      <pre><code>{project.code.trim()}</code></pre>
                    </div>
                  )}
                </>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default Accordion;
