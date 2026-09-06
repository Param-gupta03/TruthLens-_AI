import React from 'react';
import { Bot, CheckCircle, ShieldAlert, HelpCircle } from 'lucide-react';

export default function ExplanationCard({ explanation, llm }) {
  if (!explanation) {
    const isRejected = llm?.status === 'inconsistent_rejected';
    return (
      <div className="bg-white border-l-4 border-l-[#b45309] border border-[#ebe4d8] rounded-xl p-6 mb-6 shadow-sm">
        <div className="flex items-center justify-between pb-3 mb-3 border-b border-[#f4efe6]">
          <div className="flex items-center gap-2 font-serif font-bold text-base text-[#1c1d22]">
            <Bot size={18} className="text-[#b45309]" />
            <span>Editorial Synthesis Status</span>
          </div>
          <span className="px-2.5 py-0.5 rounded text-xs font-mono bg-[#fffbeb] text-[#b45309] border border-[#fde68a]">
            {isRejected ? 'Filtered for Consistency' : 'Standard Verification'}
          </span>
        </div>
        <p className="text-sm text-[#5d6069] leading-relaxed">
          {isRejected
            ? 'The generated summary was excluded to protect model accuracy, as it diverged from authoritative Model B neural entailment. Model B predictions and verified citations remain fully authoritative.'
            : 'Automated natural language synthesis is currently in offline mode. The fine-tuned Model A & B neural predictions and verified web citations below remain fully authoritative.'}
        </p>
      </div>
    );
  }

  const { summary, verdictExplanation, keyEvidence = [], contradictoryEvidence = [], uncertainty, sourceNotes = [] } = explanation;

  return (
    <div className="bg-white border border-[#ebe4d8] rounded-xl p-6 sm:p-7 mb-6 shadow-[0_1px_4px_rgba(0,0,0,0.03)]">
      <div className="flex items-center justify-between pb-4 mb-5 border-b border-[#f4efe6] flex-wrap gap-2">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-[#f4efe6] text-[#1c1d22] flex items-center justify-center">
            <Bot size={18} />
          </div>
          <h3 className="font-serif text-xl font-bold text-[#1c1d22]">
            Investigative Synthesis & Findings
          </h3>
        </div>
        <span className="px-2.5 py-1 rounded-full text-xs font-mono bg-[#f4efe6] text-[#5d6069] border border-[#dfd5c6]">
          Grounded in Retrieved Evidence
        </span>
      </div>

      {summary && (
        <div className="mb-6">
          <div className="text-[11px] font-mono font-bold uppercase tracking-wider text-[#787b85] mb-1.5">
            Executive Summary
          </div>
          <p className="font-serif text-base sm:text-lg text-[#1c1d22] leading-relaxed bg-[#fbf9f5] p-4 rounded-lg border border-[#f4efe6]">
            {summary}
          </p>
        </div>
      )}

      {verdictExplanation && (
        <div className="mb-6">
          <div className="text-[11px] font-mono font-bold uppercase tracking-wider text-[#787b85] mb-1.5">
            Verdict Rationale
          </div>
          <p className="text-sm text-[#44474f] leading-relaxed">
            {verdictExplanation}
          </p>
        </div>
      )}

      {keyEvidence.length > 0 && (
        <div className="mb-6">
          <div className="flex items-center gap-1.5 text-[11px] font-mono font-bold uppercase tracking-wider text-[#15803d] mb-2.5">
            <CheckCircle size={14} />
            <span>Key Corroborating Findings</span>
          </div>
          <ul className="space-y-2">
            {keyEvidence.map((point, index) => (
              <li key={index} className="flex items-start gap-2.5 text-sm text-[#14532d] bg-[#f0fdf4] border border-[#bbf7d0] rounded-lg p-3">
                <span className="text-xs font-mono font-bold text-[#15803d] shrink-0 mt-0.5">0{index + 1}</span>
                <span className="leading-relaxed">{point}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {contradictoryEvidence.length > 0 && (
        <div className="mb-6 bg-[#fef2f2] border border-[#fecaca] rounded-xl p-4 sm:p-5">
          <div className="flex items-center gap-2 text-xs font-mono font-bold uppercase tracking-wider text-[#b91c1c] mb-2.5">
            <ShieldAlert size={15} />
            <span>Counter-Evidence & Conflicting Perspectives</span>
          </div>
          <ul className="space-y-1.5 pl-5 list-disc text-sm text-[#7f1d1d] leading-relaxed">
            {contradictoryEvidence.map((point, index) => (
              <li key={index}>
                {point}
              </li>
            ))}
          </ul>
        </div>
      )}

      {uncertainty && (
        <div className="mb-6">
          <div className="flex items-center gap-1.5 text-[11px] font-mono font-bold uppercase tracking-wider text-[#b45309] mb-1.5">
            <HelpCircle size={14} />
            <span>Evidentiary Uncertainty & Limitations</span>
          </div>
          <p className="text-sm text-[#5d6069] leading-relaxed bg-[#fffbeb] border border-[#fde68a] p-3.5 rounded-lg">
            {uncertainty}
          </p>
        </div>
      )}

      {sourceNotes.length > 0 && (
        <div className="pt-4 border-t border-[#f4efe6] flex items-center gap-2 flex-wrap">
          <span className="text-xs font-mono text-[#787b85] font-semibold">Attributed Citations:</span>
          {sourceNotes.map((note, index) => (
            <span key={index} className="text-xs font-mono text-[#1c1d22] bg-[#f4efe6] px-2 py-0.5 rounded border border-[#dfd5c6]">
              {note}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
