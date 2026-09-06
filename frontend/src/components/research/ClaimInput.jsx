import React, { useState } from 'react';
import { Search, X, ArrowRight, CornerDownLeft } from 'lucide-react';

const SAMPLE_CLAIMS = [
  'Smoking causes lung cancer.',
  'Vaccines cause autism.',
  'Human activity contributes to global climate change.',
  'A secret species named Xylok-9 was discovered in 2024.'
];

const MAX_CLAIM_LENGTH = 1000;

export default function ClaimInput({ onResearch, loading }) {
  const [claimText, setClaimText] = useState('');

  const handleSubmit = (e) => {
    e?.preventDefault();
    const trimmedClaim = claimText.trim();
    if (!trimmedClaim || loading) return;
    onResearch(trimmedClaim);
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleSampleClick = (sampleText) => {
    setClaimText(sampleText);
    onResearch(sampleText);
  };

  return (
    <div>
      <form onSubmit={handleSubmit} className="bg-white border border-[#ebe4d8] rounded-xl p-5 sm:p-6 shadow-[0_2px_8px_rgba(28,29,34,0.04)] transition-all focus-within:border-[#1c1d22] focus-within:shadow-md mb-6">
        <label htmlFor="claim-input" className="sr-only">Claim to investigate</label>
        <textarea
          id="claim-input"
          className="w-full min-h-[95px] bg-transparent border-0 outline-none text-[#1c1d22] placeholder:text-[#9b9ea7] font-serif text-lg sm:text-xl leading-relaxed resize-y"
          placeholder="Enter a factual statement or claim to investigate (e.g. 'Smoking causes lung cancer.')..."
          value={claimText}
          onChange={(e) => setClaimText(e.target.value.slice(0, MAX_CLAIM_LENGTH))}
          onKeyDown={handleKeyDown}
          disabled={loading}
          rows={3}
        />

        <div className="flex items-center justify-between pt-3.5 border-t border-[#f4efe6] flex-wrap gap-3 mt-2">
          <div className="text-xs font-mono text-[#787b85] flex items-center gap-2">
            <span>{claimText.length} / {MAX_CLAIM_LENGTH} characters</span>
            <span className="hidden sm:inline text-[#c2c4cb]">•</span>
            <span className="hidden sm:inline text-[11px] text-[#9b9ea7]">Press Enter ↵ to investigate</span>
          </div>

          <div className="flex items-center gap-2">
            {claimText.length > 0 && !loading && (
              <button
                type="button"
                onClick={() => setClaimText('')}
                className="px-3 py-1.5 rounded-md bg-[#f4efe6] hover:bg-[#ebe4d8] text-xs font-medium text-[#5d6069] flex items-center gap-1 transition-colors"
                title="Clear input"
              >
                <X size={14} />
                <span>Clear</span>
              </button>
            )}

            <button
              type="submit"
              className="px-5 py-2.5 rounded-md bg-[#1c1d22] hover:bg-[#2e3036] text-white text-sm font-semibold tracking-wide flex items-center gap-2 shadow-sm transition-all disabled:opacity-50 disabled:cursor-not-allowed"
              disabled={!claimText.trim() || loading}
            >
              <Search size={15} />
              <span>{loading ? 'Investigating...' : 'Verify Claim'}</span>
              {!loading && <CornerDownLeft size={13} className="opacity-70" />}
            </button>
          </div>
        </div>
      </form>

      <div className="flex items-center gap-2 flex-wrap mb-8">
        <span className="text-xs font-semibold uppercase tracking-wider text-[#787b85] mr-1">
          Sample Inquiries:
        </span>
        {SAMPLE_CLAIMS.map((sampleText, index) => (
          <button
            key={index}
            type="button"
            className="px-3 py-1.5 rounded-full bg-white border border-[#dfd5c6] text-xs font-medium text-[#44474f] hover:border-[#1c1d22] hover:bg-[#f4efe6] hover:text-[#1c1d22] transition-all shadow-[0_1px_2px_rgba(0,0,0,0.02)]"
            onClick={() => handleSampleClick(sampleText)}
            disabled={loading}
          >
            "{sampleText}"
          </button>
        ))}
      </div>
    </div>
  );
}
