import React from 'react';

export default function ProbabilityDistribution({ probabilities = {} }) {
  const probSupports = Number(probabilities.SUPPORTS || 0);
  const probRefutes = Number(probabilities.REFUTES || 0);
  const probNeutral = Number(probabilities.NOT_ENOUGH_INFO || 0);

  const distributionItems = [
    { label: 'SUPPORTS', value: probSupports, percent: (probSupports * 100).toFixed(1), fillClass: 'bg-[#15803d]' },
    { label: 'REFUTES', value: probRefutes, percent: (probRefutes * 100).toFixed(1), fillClass: 'bg-[#b91c1c]' },
    { label: 'INSUFFICIENT INFO', value: probNeutral, percent: (probNeutral * 100).toFixed(1), fillClass: 'bg-[#b45309]' }
  ];

  return (
    <div className="bg-white border border-[#ebe4d8] rounded-xl p-5 sm:p-6 mb-6 shadow-[0_1px_4px_rgba(0,0,0,0.03)]">
      <div className="flex items-center justify-between mb-4 pb-2 border-b border-[#f4efe6]">
        <h4 className="text-xs font-mono font-bold uppercase tracking-wider text-[#5d6069]">
          Neural Model B Probability Distribution
        </h4>
        <span className="font-mono text-[11px] text-[#787b85] bg-[#f4efe6] px-2 py-0.5 rounded">
          Softmax Metrics
        </span>
      </div>

      <div className="space-y-3">
        {distributionItems.map((item, index) => (
          <div key={index} className="flex items-center gap-4">
            <div className="w-36 text-xs font-mono font-bold text-[#1c1d22] shrink-0">
              {item.label}
            </div>
            <div className="flex-1 h-2 bg-[#f4efe6] rounded-full overflow-hidden">
              <div
                className={`h-full rounded-full transition-all duration-500 ease-out ${item.fillClass}`}
                style={{ width: `${Math.max(item.value * 100, item.value > 0 ? 1 : 0)}%` }}
              />
            </div>
            <div className="w-14 text-right font-mono text-xs font-bold text-[#1c1d22]">
              {item.percent}%
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
