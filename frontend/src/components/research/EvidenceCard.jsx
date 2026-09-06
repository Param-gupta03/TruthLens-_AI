import React from 'react';
import { ExternalLink, Quote } from 'lucide-react';

export default function EvidenceCard({ evidence, index }) {
  const { text, relevanceScore, verification, source } = evidence;
  const verificationLabel = verification?.label || 'NOT_ENOUGH_INFO';
  const confidencePercent = ((verification?.confidence || 0) * 100).toFixed(1);

  let verdictPillClass = 'bg-[#fffbeb] text-[#b45309] border-[#fde68a]';
  if (verificationLabel === 'SUPPORTS') {
    verdictPillClass = 'bg-[#f0fdf4] text-[#15803d] border-[#bbf7d0]';
  } else if (verificationLabel === 'REFUTES') {
    verdictPillClass = 'bg-[#fef2f2] text-[#b91c1c] border-[#fecaca]';
  }

  const formattedRelevance = typeof relevanceScore === 'number'
    ? relevanceScore.toFixed(3)
    : relevanceScore;

  return (
    <div className="bg-[#fcfbf8] border border-[#ebe4d8] rounded-xl p-5 mb-3.5 transition-all hover:border-[#dfd5c6] hover:bg-white hover:shadow-xs">
      <div className="flex items-center justify-between flex-wrap gap-2 mb-2.5">
        <div className="flex items-center gap-1.5 text-[11px] font-mono font-bold uppercase tracking-wider text-[#787b85]">
          <Quote size={13} />
          <span>PASSAGE RECORD #{index + 1}</span>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-[11px] font-mono bg-white border border-[#dfd5c6] text-[#44474f] px-2 py-0.5 rounded shadow-xs" title="Model A Relevance Probability">
            Model A: {formattedRelevance}
          </span>
          <span className={`text-[11px] font-mono font-semibold border px-2 py-0.5 rounded ${verdictPillClass}`} title="Model B Classification">
            Model B: {verificationLabel} ({confidencePercent}%)
          </span>
        </div>
      </div>

      <div className="font-serif text-[15px] sm:text-base text-[#1c1d22] italic pl-3.5 border-l-2 border-[#dfd5c6] my-3 leading-relaxed">
        "{text}"
      </div>

      <div className="flex items-center justify-between flex-wrap gap-2 pt-2 border-t border-[#f4efe6] text-xs text-[#787b85]">
        <div>
          <span>Source: </span>
          <strong className="text-[#1c1d22] font-medium font-sans">
            {source?.title || source?.name || 'Retrieved Web Article'}
          </strong>
        </div>

        {source?.url && (
          <a
            href={source.url}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1 font-medium text-[#1c1d22] hover:text-[#15803d] transition-colors underline decoration-[#dfd5c6] hover:decoration-[#15803d]"
            title="Read full article at source"
          >
            <span>View source article</span>
            <ExternalLink size={12} />
          </a>
        )}
      </div>
    </div>
  );
}
