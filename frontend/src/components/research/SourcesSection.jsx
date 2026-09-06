import React from 'react';
import { Globe } from 'lucide-react';
import SourceCard from './SourceCard';

export default function SourcesSection({ sources = [] }) {
  if (sources.length === 0) return null;

  return (
    <div className="bg-white border border-[#ebe4d8] rounded-xl p-6 sm:p-7 mb-6 shadow-[0_1px_4px_rgba(0,0,0,0.03)]">
      <div className="flex items-center justify-between pb-4 mb-5 border-b border-[#f4efe6] flex-wrap gap-2">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-[#f4efe6] text-[#1c1d22] flex items-center justify-center">
            <Globe size={18} />
          </div>
          <h3 className="font-serif text-xl font-bold text-[#1c1d22]">
            Retrieved News & Web Sources ({sources.length})
          </h3>
        </div>
        <span className="font-mono text-xs text-[#787b85] bg-[#f4efe6] px-2.5 py-1 rounded border border-[#dfd5c6]">
          Real-Time Web Crawler
        </span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3.5">
        {sources.map((sourceItem, index) => (
          <SourceCard key={index} source={sourceItem} />
        ))}
      </div>
    </div>
  );
}
