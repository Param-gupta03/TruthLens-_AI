import React from 'react';

export default function ConfidenceBar({ confidence = 0, verdict = 'SUPPORTS' }) {
  const formattedPercent = Math.min(100, Math.max(0, confidence * 100)).toFixed(1);
  const normalizedVerdict = verdict.toLowerCase().replace(/_/g, '-');
  
  const fillClass = normalizedVerdict === 'supports'
    ? 'bg-[#15803d]'
    : normalizedVerdict === 'refutes'
      ? 'bg-[#b91c1c]'
      : 'bg-[#b45309]';

  return (
    <div className="w-full">
      <div className="flex items-center justify-between text-xs mb-1.5 text-[#5d6069]">
        <span className="font-medium">Verification Confidence</span>
        <span className="font-mono font-bold text-[#1c1d22]">{formattedPercent}%</span>
      </div>
      <div className="w-full h-2 bg-[#ebe4d8] rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-500 ease-out ${fillClass}`}
          style={{ width: `${formattedPercent}%` }}
        />
      </div>
    </div>
  );
}
