import React, { useState, useEffect } from 'react';
import { Globe, FileText, Filter, CheckCircle2, Bot, Database, Loader2 } from 'lucide-react';

const PIPELINE_STAGES = [
  { id: 1, name: 'Web Discovery', desc: 'Searching credible reporting', icon: Globe },
  { id: 2, name: 'Text Extraction', desc: 'Parsing article paragraphs', icon: FileText },
  { id: 3, name: 'Model A Filter', desc: 'RoBERTa relevance scoring', icon: Filter },
  { id: 4, name: 'Model B Verification', desc: 'SciFact neural veracity', icon: CheckCircle2 },
  { id: 5, name: 'Editorial Synthesis', desc: 'Grounded factual summary', icon: Bot },
  { id: 6, name: 'Dossier Storage', desc: 'Archiving report in MongoDB', icon: Database }
];

export default function PipelineProgress({ claim }) {
  const [currentStep, setCurrentStep] = useState(1);

  useEffect(() => {
    const timer1 = setTimeout(() => setCurrentStep(2), 800);
    const timer2 = setTimeout(() => setCurrentStep(3), 1600);
    const timer3 = setTimeout(() => setCurrentStep(4), 2400);
    const timer4 = setTimeout(() => setCurrentStep(5), 3200);
    const timer5 = setTimeout(() => setCurrentStep(6), 4000);

    return () => {
      clearTimeout(timer1);
      clearTimeout(timer2);
      clearTimeout(timer3);
      clearTimeout(timer4);
      clearTimeout(timer5);
    };
  }, []);

  return (
    <div className="bg-white border border-[#ebe4d8] rounded-xl p-6 mb-8 shadow-sm">
      <div className="flex items-center justify-between flex-wrap gap-4 pb-4 border-b border-[#f4efe6]">
        <div>
          <div className="flex items-center gap-2 text-xs font-mono font-semibold uppercase tracking-wider text-[#1c1d22]">
            <Loader2 size={15} className="animate-spin text-[#1c1d22]" />
            <span>Investigation In Progress</span>
          </div>
          <h3 className="font-serif text-lg font-bold text-[#1c1d22] mt-1">
            "{claim}"
          </h3>
        </div>
        <div className="font-mono text-xs bg-[#f4efe6] px-2.5 py-1 rounded border border-[#dfd5c6] text-[#5d6069]">
          Phase {currentStep} of {PIPELINE_STAGES.length}
        </div>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2.5 mt-5">
        {PIPELINE_STAGES.map((step) => {
          const StepIcon = step.icon;
          const isCompleted = currentStep > step.id;
          const isActive = currentStep === step.id;

          let cardClasses = 'bg-white border-[#ebe4d8] text-[#787b85]';
          let iconClasses = 'bg-[#f4efe6] text-[#787b85]';

          if (isCompleted) {
            cardClasses = 'bg-[#f0fdf4] border-[#bbf7d0] text-[#15803d]';
            iconClasses = 'bg-white text-[#15803d] border border-[#bbf7d0]';
          } else if (isActive) {
            cardClasses = 'bg-[#faf7f0] border-[#1c1d22] text-[#1c1d22] ring-1 ring-[#1c1d22]/10';
            iconClasses = 'bg-[#1c1d22] text-white';
          }

          return (
            <div
              key={step.id}
              className={`flex flex-col items-center text-center p-3 rounded-lg border transition-all ${cardClasses}`}
            >
              <div className={`w-8 h-8 rounded-full flex items-center justify-center mb-2 text-xs font-semibold ${iconClasses}`}>
                {isCompleted ? (
                  <CheckCircle2 size={16} />
                ) : isActive ? (
                  <Loader2 size={16} className="animate-spin" />
                ) : (
                  <StepIcon size={15} />
                )}
              </div>
              <div className="text-xs font-bold font-serif leading-tight">{step.name}</div>
              <div className="text-[11px] text-[#787b85] mt-1 leading-snug">{step.desc}</div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
