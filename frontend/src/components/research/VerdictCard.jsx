import React from 'react';
import { CheckCircle2, XCircle, HelpCircle, Cpu } from 'lucide-react';

export default function VerdictCard({ verdict = 'NOT_ENOUGH_INFO', confidence = 0 }) {
  const normalizedVerdict = (verdict || 'NOT_ENOUGH_INFO').toUpperCase();
  const confidencePercent = ((confidence || 0) * 100).toFixed(1);

  let statusIcon = <HelpCircle size={28} />;
  let containerClasses = 'bg-[#fffbeb] border-2 border-[#b45309]/30 text-[#78350f]';
  let badgeClasses = 'bg-[#b45309] text-white';
  let badgeLabel = 'INSUFFICIENT EVIDENCE';
  let titleColor = 'text-[#78350f]';

  if (normalizedVerdict === 'SUPPORTS') {
    statusIcon = <CheckCircle2 size={28} />;
    containerClasses = 'bg-[#f0fdf4] border-2 border-[#15803d]/40 text-[#14532d]';
    badgeClasses = 'bg-[#15803d] text-white';
    badgeLabel = 'VERIFIED SUPPORTED';
    titleColor = 'text-[#14532d]';
  } else if (normalizedVerdict === 'REFUTES') {
    statusIcon = <XCircle size={28} />;
    containerClasses = 'bg-[#fef2f2] border-2 border-[#b91c1c]/40 text-[#7f1d1d]';
    badgeClasses = 'bg-[#b91c1c] text-white';
    badgeLabel = 'VERIFIED REFUTED';
    titleColor = 'text-[#7f1d1d]';
  }

  return (
    <div className={`rounded-xl p-6 sm:p-7 mb-6 flex items-center justify-between flex-wrap gap-6 shadow-[0_2px_8px_rgba(28,29,34,0.04)] ${containerClasses}`}>
      <div className="flex items-center gap-4 sm:gap-5">
        <div className="shrink-0 p-3 rounded-xl bg-white shadow-xs">
          {statusIcon}
        </div>

        <div>
          <span className={`inline-block px-2.5 py-0.5 rounded text-[11px] font-mono font-bold uppercase tracking-wider mb-1.5 ${badgeClasses}`}>
            {badgeLabel}
          </span>
          <h3 className={`font-serif text-2xl sm:text-3xl font-bold tracking-tight ${titleColor}`}>
            {normalizedVerdict === 'NOT_ENOUGH_INFO' ? 'Insufficient Evidence' : normalizedVerdict.replace(/_/g, ' ')}
          </h3>
          {normalizedVerdict === 'NOT_ENOUGH_INFO' && (
            <p className="text-xs text-[#78350f]/80 mt-1 max-w-md">
              The retrieved credible web sources do not provide sufficient corroborating or refuting evidence.
            </p>
          )}
          <div className="flex items-center gap-1.5 text-xs text-[#5d6069] mt-2 font-mono">
            <Cpu size={13} className="text-[#1c1d22]" />
            <span>Authoritative Entailment Assessment via Model B</span>
          </div>
        </div>
      </div>

      <div className="text-right sm:border-l sm:border-current/15 sm:pl-6">
        <div className={`font-mono text-3xl sm:text-4xl font-black ${titleColor}`}>
          {confidencePercent}%
        </div>
        <div className="text-xs uppercase tracking-wider font-semibold text-[#5d6069] mt-0.5">
          Model Confidence
        </div>
      </div>
    </div>
  );
}
