import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowUpRight, History, Newspaper } from 'lucide-react';
import ClaimInput from '../components/research/ClaimInput';
import PipelineProgress from '../components/research/PipelineProgress';
import VerdictCard from '../components/research/VerdictCard';
import ProbabilityDistribution from '../components/research/ProbabilityDistribution';
import ExplanationCard from '../components/research/ExplanationCard';
import EvidenceSection from '../components/research/EvidenceSection';
import SourcesSection from '../components/research/SourcesSection';
import TechnicalMetadata from '../components/research/TechnicalMetadata';
import ResearchTimeline from '../components/research/ResearchTimeline';
import ErrorBanner from '../components/common/ErrorBanner';
import { researchClaim, getFactChecks } from '../services/api';

export default function ResearchPage() {
  const [loading, setLoading] = useState(false);
  const [activeClaim, setActiveClaim] = useState('');
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [recentHistory, setRecentHistory] = useState([]);
  const navigate = useNavigate();

  useEffect(() => {
    getFactChecks(1, 3)
      .then(res => setRecentHistory(res.records || []))
      .catch(() => {});
  }, []);

  const handleResearch = async (claim) => {
    setError(null);
    setActiveClaim(claim);
    setLoading(true);
    setResult(null);

    try {
      const data = await researchClaim(claim, true);
      setResult(data);
    } catch (err) {
      setError(err.message || 'An unexpected error occurred during research. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="editorial-container pt-8 sm:pt-12">
      {/* Editorial Masthead / Hero */}
      <div className="text-center mb-10 sm:mb-12">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-[#f4efe6] border border-[#dfd5c6] text-xs font-mono font-bold tracking-wider text-[#1c1d22] uppercase mb-4 shadow-xs">
          <Newspaper size={13} className="text-[#1c1d22]" />
          <span>Independent Evidence Investigation</span>
        </div>
        <h1 className="font-serif text-3xl sm:text-5xl font-bold tracking-tight text-[#1c1d22] max-w-3xl mx-auto leading-tight sm:leading-[1.15]">
          Investigate factual claims with credible web evidence
        </h1>
        <p className="text-[#5d6069] text-base sm:text-lg max-w-2xl mx-auto mt-3.5 leading-relaxed font-sans">
          Cross-reference news reporting, extract verifiable passages with fine-tuned transformers, and generate evidence-grounded dossiers without generative hallucination.
        </p>
      </div>

      <ClaimInput onResearch={handleResearch} loading={loading} />

      <ErrorBanner message={error} onRetry={() => activeClaim && handleResearch(activeClaim)} />

      {loading && <PipelineProgress claim={activeClaim} />}

      {result && !loading && (
        <div className="mt-8">
          <div className="flex items-center justify-between mb-5 flex-wrap gap-3 pb-3 border-b border-[#ebe4d8]">
            <div>
              <span className="text-xs font-mono font-bold uppercase tracking-wider text-[#787b85]">
                INVESTIGATION DOSSIER
              </span>
              <h2 className="font-serif text-2xl font-bold text-[#1c1d22] mt-0.5">
                "{result.claim}"
              </h2>
            </div>

            {result.id && (
              <button
                onClick={() => navigate(`/report/${result.id}`)}
                className="px-3.5 py-1.5 rounded-md bg-white border border-[#dfd5c6] hover:bg-[#f4efe6] text-xs font-semibold text-[#1c1d22] flex items-center gap-1.5 shadow-xs transition-colors"
              >
                <span>Open Printable Dossier</span>
                <ArrowUpRight size={13} />
              </button>
            )}
          </div>

          <VerdictCard verdict={result.verdict} confidence={result.confidence} />

          <ProbabilityDistribution probabilities={result.probabilities} />

          <ExplanationCard explanation={result.explanation} llm={result.llm} />

          <EvidenceSection evidence={result.evidence} categorized={result.categorizedEvidence} />

          <SourcesSection sources={result.sources} />

          <TechnicalMetadata retrieval={result.retrieval} llm={result.llm} id={result.id} />
        </div>
      )}

      {!result && !loading && (
        <>
          <ResearchTimeline />

          {recentHistory.length > 0 && (
            <div className="bg-white border border-[#ebe4d8] rounded-xl p-6 sm:p-7 shadow-[0_1px_4px_rgba(0,0,0,0.03)]">
              <div className="flex items-center justify-between pb-4 mb-4 border-b border-[#f4efe6]">
                <div className="flex items-center gap-2 font-serif font-bold text-lg text-[#1c1d22]">
                  <History size={18} className="text-[#1c1d22]" />
                  <span>Recent Public Inquiries</span>
                </div>
                <button
                  onClick={() => navigate('/history')}
                  className="text-xs font-semibold text-[#1c1d22] hover:text-[#15803d] transition-colors"
                >
                  View Full Archive →
                </button>
              </div>

              <div className="space-y-2.5">
                {recentHistory.map((historyItem, historyIndex) => {
                  const normalizedVerdict = (historyItem.verdict || 'not-enough-info').toLowerCase().replace(/_/g, '-');
                  let verdictBadgeClass = 'bg-[#fffbeb] text-[#b45309] border-[#fde68a]';
                  if (normalizedVerdict === 'supports') {
                    verdictBadgeClass = 'bg-[#f0fdf4] text-[#15803d] border-[#bbf7d0]';
                  } else if (normalizedVerdict === 'refutes') {
                    verdictBadgeClass = 'bg-[#fef2f2] text-[#b91c1c] border-[#fecaca]';
                  }

                  return (
                    <div
                      key={historyIndex}
                      className="bg-[#fcfbf8] border border-[#ebe4d8] hover:border-[#dfd5c6] hover:bg-white p-4 rounded-xl transition-all cursor-pointer flex items-center justify-between gap-4"
                      onClick={() => navigate(`/report/${historyItem.id}`)}
                    >
                      <div>
                        <div className="font-serif text-[15px] font-bold text-[#1c1d22]">
                          "{historyItem.claim}"
                        </div>
                        <div className="flex items-center gap-2 text-xs text-[#787b85] font-mono mt-1">
                          <span>{new Date(historyItem.createdAt).toLocaleDateString()}</span>
                          <span>•</span>
                          <span>{historyItem.evidence?.length || 0} cited passages</span>
                        </div>
                      </div>
                      <span className={`px-2.5 py-1 rounded text-xs font-mono font-bold border ${verdictBadgeClass} shrink-0`}>
                        {historyItem.verdict} ({historyItem.confidencePercent}%)
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
