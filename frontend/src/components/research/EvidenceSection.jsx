import React, { useState } from 'react';
import { Layers, CheckCircle2, XCircle, HelpCircle } from 'lucide-react';
import EvidenceCard from './EvidenceCard';

export default function EvidenceSection({ evidence = [], categorized = {} }) {
  const [activeTab, setActiveTab] = useState('all');

  const { supporting = [], refuting = [], uncertain = [] } = categorized;

  let displayList = evidence;
  if (activeTab === 'supporting') displayList = supporting;
  else if (activeTab === 'refuting') displayList = refuting;
  else if (activeTab === 'uncertain') displayList = uncertain;

  return (
    <div className="bg-white border border-[#ebe4d8] rounded-xl p-6 sm:p-7 mb-6 shadow-[0_1px_4px_rgba(0,0,0,0.03)]">
      <div className="flex items-center justify-between pb-4 mb-5 border-b border-[#f4efe6] flex-wrap gap-3">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-[#f4efe6] text-[#1c1d22] flex items-center justify-center">
            <Layers size={18} />
          </div>
          <h3 className="font-serif text-xl font-bold text-[#1c1d22]">
            Evaluated Evidence Passages ({evidence.length})
          </h3>
        </div>

        <div className="flex items-center gap-1 bg-[#f4efe6] p-1 rounded-lg border border-[#dfd5c6] text-xs">
          <button
            type="button"
            onClick={() => setActiveTab('all')}
            className={`px-3 py-1 rounded-md font-medium transition-colors ${
              activeTab === 'all'
                ? 'bg-white text-[#1c1d22] shadow-xs font-bold'
                : 'text-[#5d6069] hover:text-[#1c1d22]'
            }`}
          >
            All ({evidence.length})
          </button>

          <button
            type="button"
            onClick={() => setActiveTab('supporting')}
            className={`inline-flex items-center gap-1 px-3 py-1 rounded-md font-medium transition-colors ${
              activeTab === 'supporting'
                ? 'bg-white text-[#15803d] shadow-xs font-bold'
                : 'text-[#5d6069] hover:text-[#15803d]'
            }`}
          >
            <CheckCircle2 size={12} className="text-[#15803d]" />
            <span>Supporting ({supporting.length})</span>
          </button>

          <button
            type="button"
            onClick={() => setActiveTab('refuting')}
            className={`inline-flex items-center gap-1 px-3 py-1 rounded-md font-medium transition-colors ${
              activeTab === 'refuting'
                ? 'bg-white text-[#b91c1c] shadow-xs font-bold'
                : 'text-[#5d6069] hover:text-[#b91c1c]'
            }`}
          >
            <XCircle size={12} className="text-[#b91c1c]" />
            <span>Refuting ({refuting.length})</span>
          </button>

          <button
            type="button"
            onClick={() => setActiveTab('uncertain')}
            className={`inline-flex items-center gap-1 px-3 py-1 rounded-md font-medium transition-colors ${
              activeTab === 'uncertain'
                ? 'bg-white text-[#b45309] shadow-xs font-bold'
                : 'text-[#5d6069] hover:text-[#b45309]'
            }`}
          >
            <HelpCircle size={12} className="text-[#b45309]" />
            <span>Inconclusive ({uncertain.length})</span>
          </button>
        </div>
      </div>

      {displayList.length === 0 ? (
        <div className="py-12 text-center text-sm font-serif italic text-[#787b85]">
          {activeTab === 'refuting' && 'No refuting passages identified for this claim in retrieved web articles.'}
          {activeTab === 'supporting' && 'No supporting passages identified for this claim in retrieved web articles.'}
          {activeTab === 'uncertain' && 'No inconclusive passages survived relevance filtering.'}
          {activeTab === 'all' && 'No readable passages were extracted from the retrieved sources.'}
        </div>
      ) : (
        <div className="space-y-3">
          {displayList.map((evidenceItem, index) => (
            <EvidenceCard key={index} evidence={evidenceItem} index={index} />
          ))}
        </div>
      )}
    </div>
  );
}
