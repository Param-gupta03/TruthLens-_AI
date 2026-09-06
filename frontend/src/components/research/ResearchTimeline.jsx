import React from 'react';
import { Search, Filter, CheckCircle2, Bot, ShieldCheck } from 'lucide-react';

const HIERARCHY_STEPS = [
  {
    num: '1',
    title: 'Search & Harvest',
    subtitle: 'Archival & News Ingress',
    desc: 'Query verified reporting databases and scrape full-body paragraph passages.',
    icon: Search,
    iconColor: 'text-[#1c1d22]'
  },
  {
    num: '2',
    title: 'Relevance Filter',
    subtitle: 'Model A (RoBERTa)',
    desc: 'Calculates claim-passage relevance probabilities, removing noise and unrelated text.',
    icon: Filter,
    iconColor: 'text-[#1c1d22]'
  },
  {
    num: '3',
    title: 'Entailment Verdict',
    subtitle: 'Model B (SciFact)',
    desc: 'Determines authoritative verdict: SUPPORTS, REFUTES, or NOT_ENOUGH_INFO.',
    icon: CheckCircle2,
    iconColor: 'text-[#15803d]'
  },
  {
    num: '4',
    title: 'Factual Synthesis',
    subtitle: 'Attributed Summary Layer',
    desc: 'Gathers verified quotations and transparently articulates verdict rationales.',
    icon: Bot,
    iconColor: 'text-[#1c1d22]'
  }
];

export default function ResearchTimeline() {
  return (
    <div className="bg-white border border-[#ebe4d8] rounded-xl p-6 sm:p-7 mb-8 shadow-[0_1px_4px_rgba(0,0,0,0.03)]">
      <div className="flex items-center justify-between pb-4 mb-5 border-b border-[#f4efe6] flex-wrap gap-3">
        <div>
          <div className="flex items-center gap-1.5 text-xs font-mono font-bold uppercase tracking-wider text-[#787b85]">
            <ShieldCheck size={15} className="text-[#15803d]" />
            <span>Verification Methodology & Decision Hierarchy</span>
          </div>
          <h3 className="font-serif text-xl font-bold text-[#1c1d22] mt-0.5">
            How TruthLens Investigates Claims
          </h3>
        </div>

        <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-[#f4efe6] border border-[#dfd5c6] text-xs font-mono text-[#44474f]">
          <span className="font-bold text-[#1c1d22]">Model B = Final Authority</span>
          <span>•</span>
          <span>Citations Required</span>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-5">
        {HIERARCHY_STEPS.map((step) => {
          const StepIcon = step.icon;
          return (
            <div
              key={step.num}
              className="bg-[#fcfbf8] border border-[#ebe4d8] rounded-xl p-4 sm:p-5 flex flex-col justify-between hover:border-[#dfd5c6] transition-all"
            >
              <div>
                <div className="flex items-center justify-between mb-3">
                  <div className={`w-8 h-8 rounded-lg bg-white border border-[#dfd5c6] flex items-center justify-center ${step.iconColor}`}>
                    <StepIcon size={16} />
                  </div>
                  <span className="font-mono text-xs font-bold text-[#9b9ea7]">
                    STAGE 0{step.num}
                  </span>
                </div>

                <div className="font-serif text-base font-bold text-[#1c1d22]">
                  {step.title}
                </div>
                <div className="text-[11px] font-mono text-[#5d6069] font-medium mt-0.5 mb-2">
                  {step.subtitle}
                </div>
              </div>
              <div className="text-xs text-[#5d6069] leading-relaxed">
                {step.desc}
              </div>
            </div>
          );
        })}
      </div>

      <div className="bg-[#f0fdf4] border border-[#bbf7d0] rounded-xl p-4 sm:p-4.5 flex items-start sm:items-center gap-3.5 text-xs text-[#14532d] leading-relaxed">
        <ShieldCheck size={20} className="text-[#15803d] shrink-0 mt-0.5 sm:mt-0" />
        <div>
          <strong className="font-bold font-serif text-[13px] text-[#14532d] mr-1.5">Editorial Integrity Standard:</strong>
          Generative summaries never alter or fabricate model verdicts. Entailment determinations are calculated strictly by the fine-tuned SciFact transformer on the GPU; all factual explanations require cited web passages.
        </div>
      </div>
    </div>
  );
}
