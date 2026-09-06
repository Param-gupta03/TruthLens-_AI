import React, { useState } from 'react';
import { Cpu, ChevronDown, ChevronUp } from 'lucide-react';

export default function TechnicalMetadata({ retrieval, llm, id }) {
  const [expanded, setExpanded] = useState(false);
  const timings = retrieval?.timings || {};

  return (
    <div className="bg-white border border-[#ebe4d8] rounded-xl p-5 mb-6 shadow-[0_1px_4px_rgba(0,0,0,0.03)]">
      <button
        type="button"
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between text-left group"
      >
        <div className="flex items-center gap-2.5 font-serif font-bold text-sm text-[#1c1d22]">
          <div className="w-6 h-6 rounded bg-[#f4efe6] flex items-center justify-center text-[#1c1d22]">
            <Cpu size={14} />
          </div>
          <span>Pipeline Telemetry & Execution Latency Breakdown</span>
        </div>
        <div className="flex items-center gap-2.5 text-xs font-mono text-[#787b85]">
          <span className="bg-[#f4efe6] px-2 py-0.5 rounded border border-[#dfd5c6]">
            {timings.totalTimeSec ? `${timings.totalTimeSec}s Total` : 'View Telemetry'}
          </span>
          {expanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
        </div>
      </button>

      {expanded && (
        <div className="mt-4 pt-4 border-t border-[#f4efe6]">
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-4">
            <div className="bg-[#fcfbf8] border border-[#ebe4d8] p-3 rounded-lg">
              <div className="text-[11px] font-mono uppercase text-[#787b85]">Search Queries</div>
              <div className="font-mono text-lg font-bold text-[#1c1d22] mt-0.5">
                {timings.searchTimeSec || 0}s
              </div>
            </div>

            <div className="bg-[#fcfbf8] border border-[#ebe4d8] p-3 rounded-lg">
              <div className="text-[11px] font-mono uppercase text-[#787b85]">Article Parsing</div>
              <div className="font-mono text-lg font-bold text-[#1c1d22] mt-0.5">
                {timings.extractionTimeSec || 0}s
              </div>
            </div>

            <div className="bg-[#fcfbf8] border border-[#ebe4d8] p-3 rounded-lg">
              <div className="text-[11px] font-mono uppercase text-[#787b85]">CUDA Neural Models</div>
              <div className="font-mono text-lg font-bold text-[#1c1d22] mt-0.5">
                {timings.mlTimeSec || 0}s
              </div>
            </div>

            <div className="bg-[#fcfbf8] border border-[#ebe4d8] p-3 rounded-lg">
              <div className="text-[11px] font-mono uppercase text-[#787b85]">Editorial Synthesis</div>
              <div className="font-mono text-lg font-bold text-[#1c1d22] mt-0.5">
                {timings.llmTimeSec || 0}s
              </div>
            </div>
          </div>

          <div className="bg-[#fbf9f5] border border-[#f4efe6] rounded-lg p-4 font-mono text-xs text-[#5d6069] space-y-1">
            <div>Archive Record ID: <span className="font-bold text-[#1c1d22]">{id || 'N/A'}</span></div>
            <div>Articles Inspected: <span className="font-bold text-[#1c1d22]">{retrieval?.articlesProcessed || 0} of {retrieval?.articlesFound || 0} found</span></div>
            <div>Passages Generated: <span className="font-bold text-[#1c1d22]">{retrieval?.passagesCreated || 0} ({retrieval?.relevantPassages || 0} relevant)</span></div>
            <div>Search Provider: <span className="font-bold text-[#1c1d22]">{retrieval?.provider || 'duckduckgo'}</span></div>
            <div>Synthesis Service: <span className="font-bold text-[#1c1d22]">{llm?.provider || 'openai'} / {llm?.model || 'gpt-4o-mini'}</span></div>
          </div>
        </div>
      )}
    </div>
  );
}
