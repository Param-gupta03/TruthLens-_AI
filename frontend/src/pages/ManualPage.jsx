import React, { useState } from 'react';
import { Plus, Trash2, ArrowRight, Sliders, CheckCircle2 } from 'lucide-react';
import VerdictCard from '../components/research/VerdictCard';
import ProbabilityDistribution from '../components/research/ProbabilityDistribution';
import EvidenceCard from '../components/research/EvidenceCard';
import ErrorBanner from '../components/common/ErrorBanner';
import { submitManualFactCheck } from '../services/api';

const DEFAULT_CLAIM = 'Smoking causes lung cancer.';
const DEFAULT_PASSAGES = [
  'Cigarette smoking is the leading risk factor for lung cancer, causing approximately 85 percent of all cases.'
];

export default function ManualPage() {
  const [claim, setClaim] = useState(DEFAULT_CLAIM);
  const [passages, setPassages] = useState(DEFAULT_PASSAGES);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const addPassage = () => {
    if (passages.length >= 10) return;
    setPassages([...passages, '']);
  };

  const removePassage = (index) => {
    if (passages.length <= 1) return;
    setPassages(passages.filter((_, idx) => idx !== index));
  };

  const updatePassage = (index, value) => {
    const updated = [...passages];
    updated[index] = value;
    setPassages(updated);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const validPassages = passages.map(p => p.trim()).filter(Boolean);
    if (!claim.trim() || validPassages.length === 0) {
      setError('Please provide a claim and at least one non-empty evidence passage.');
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const data = await submitManualFactCheck(claim.trim(), validPassages);
      setResult(data);
    } catch (err) {
      setError(err.message || 'Manual fact-check request failed.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="editorial-container py-8 md:py-12">
      {/* Header */}
      <div className="mb-8 border-b border-cream-300/80 pb-6">
        <div className="inline-flex items-center gap-2 px-2.5 py-1 rounded-full bg-cream-200/80 border border-cream-300 text-xs font-mono font-medium text-ink-700 mb-3">
          <Sliders size={13} className="text-ink-600" />
          <span>LABORATORY BENCHMARK MODE</span>
        </div>
        <h1 className="text-2xl md:text-3xl font-serif font-bold text-ink-950 tracking-tight">
          Manual Evidence Fact-Checking
        </h1>
        <p className="text-sm md:text-base text-ink-600 max-w-2xl mt-2 leading-relaxed">
          Evaluate a claim against custom, user-provided evidence passages directly using Model A (RoBERTa Relevance) and Model B (SciFact DeBERTa Claim Verification).
        </p>
      </div>

      <form onSubmit={handleSubmit} className="editorial-card p-6 md:p-8 space-y-6 shadow-xs">
        <div>
          <label className="block text-xs font-mono uppercase tracking-wider font-semibold text-ink-700 mb-2">
            Claim to Verify
          </label>
          <textarea
            rows={2}
            className="w-full px-4 py-3 text-sm md:text-base bg-cream-100/60 border border-cream-300 rounded-lg text-ink-950 placeholder:text-ink-400 focus:outline-none focus:ring-2 focus:ring-ink-900/10 focus:border-ink-900 transition-colors resize-none leading-relaxed font-serif"
            value={claim}
            onChange={(e) => setClaim(e.target.value)}
            placeholder="Enter claim to verify..."
            required
          />
        </div>

        <div>
          <div className="flex items-center justify-between pb-3 mb-3 border-b border-cream-200">
            <label className="text-xs font-mono uppercase tracking-wider font-semibold text-ink-700">
              Candidate Evidence Passages ({passages.length})
            </label>
            <button
              type="button"
              onClick={addPassage}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg bg-white border border-cream-300 text-ink-700 hover:text-ink-950 hover:bg-cream-100 transition-colors shadow-2xs"
            >
              <Plus size={13} />
              <span>Add Passage</span>
            </button>
          </div>

          <div className="space-y-3">
            {passages.map((passageText, index) => (
              <div key={index} className="flex items-start gap-3">
                <span className="shrink-0 w-7 h-7 rounded-md bg-cream-200 text-ink-600 text-xs font-mono font-medium flex items-center justify-center mt-1 border border-cream-300">
                  {index + 1}
                </span>
                <textarea
                  rows={2}
                  value={passageText}
                  onChange={(e) => updatePassage(index, e.target.value)}
                  placeholder={`Evidence passage #${index + 1}...`}
                  className="flex-1 px-3.5 py-2.5 text-sm bg-cream-100/40 border border-cream-300 rounded-lg text-ink-900 placeholder:text-ink-400 focus:outline-none focus:ring-2 focus:ring-ink-900/10 focus:border-ink-900 transition-colors resize-y leading-relaxed font-sans"
                />
                {passages.length > 1 && (
                  <button
                    type="button"
                    onClick={() => removePassage(index)}
                    className="p-2 text-ink-400 hover:text-false-red hover:bg-false-red-bg rounded-lg transition-colors mt-1"
                    title="Remove passage"
                  >
                    <Trash2 size={16} />
                  </button>
                )}
              </div>
            ))}
          </div>
        </div>

        <div className="pt-2 flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-t border-cream-200">
          <button
            type="submit"
            className="inline-flex items-center justify-center gap-2 px-6 py-2.5 rounded-lg bg-ink-950 text-cream-50 font-medium text-sm hover:bg-ink-900 disabled:opacity-50 transition-colors shadow-xs"
            disabled={loading}
          >
            {loading ? (
              <>
                <div className="w-4 h-4 border-2 border-cream-100/30 border-t-cream-100 rounded-full animate-spin" />
                <span>Evaluating via Models...</span>
              </>
            ) : (
              <>
                <span>Run Manual Verification</span>
                <ArrowRight size={15} />
              </>
            )}
          </button>
          <span className="text-xs font-mono text-ink-500">
            Dual-model neural inference (CUDA accelerated)
          </span>
        </div>
      </form>

      <ErrorBanner message={error} />

      {result && (
        <div className="mt-10 space-y-6">
          <div className="border-b border-cream-300 pb-3">
            <h2 className="text-xl font-serif font-bold text-ink-950">
              Verification Outcome
            </h2>
          </div>

          <VerdictCard verdict={result.verdict} confidence={result.confidence} />
          <ProbabilityDistribution probabilities={result.probabilities} />

          <div className="editorial-card p-6 space-y-4">
            <div className="flex items-center justify-between border-b border-cream-200 pb-3">
              <h3 className="text-sm font-mono uppercase tracking-wider font-semibold text-ink-700">
                Model A Relevance Evaluation
              </h3>
              <span className="text-xs font-mono text-ink-500">
                {result.evidence?.length || 0} candidate passages analyzed
              </span>
            </div>
            <div className="space-y-3">
              {(result.evidence || []).map((evidenceItem, index) => (
                <EvidenceCard key={index} evidence={evidenceItem} index={index} />
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
