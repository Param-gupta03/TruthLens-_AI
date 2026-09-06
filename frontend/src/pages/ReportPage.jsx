import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  ArrowLeft,
  PlusCircle,
  Printer,
  Globe,
  Bot,
  Shield,
  FileText,
  AlertTriangle,
  HelpCircle
} from 'lucide-react';
import VerdictCard from '../components/research/VerdictCard';
import ProbabilityDistribution from '../components/research/ProbabilityDistribution';
import EvidenceCard from '../components/research/EvidenceCard';
import SourcesSection from '../components/research/SourcesSection';
import TechnicalMetadata from '../components/research/TechnicalMetadata';
import ErrorBanner from '../components/common/ErrorBanner';
import { getResearchReport, getFactCheckById } from '../services/api';

export default function ReportPage() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);
  const [report, setReport] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!id) return;
    setLoading(true);
    setError(null);

    getResearchReport(id)
      .then(data => {
        setReport(data);
        setLoading(false);
      })
      .catch(() => {
        getFactCheckById(id)
          .then(data => {
            setReport(data);
            setLoading(false);
          })
          .catch(() => {
            setError('Unable to locate this investigative dossier. The ID may be invalid or deleted.');
            setLoading(false);
          });
      });
  }, [id]);

  const handlePrint = () => {
    window.print();
  };

  if (loading) {
    return (
      <div className="editorial-container text-center pt-24">
        <div className="w-10 h-10 rounded-full border-2 border-[#1c1d22] border-t-transparent animate-spin mx-auto mb-4" />
        <h3 className="font-serif text-xl font-bold text-[#1c1d22]">Loading Investigative Dossier...</h3>
        <p className="text-sm text-[#787b85] mt-1 font-sans">
          Retrieving verified evidentiary records from archive database
        </p>
      </div>
    );
  }

  if (error || !report) {
    return (
      <div className="editorial-container pt-10">
        <button
          onClick={() => navigate('/')}
          className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-md bg-white border border-[#dfd5c6] text-xs font-semibold text-[#1c1d22] mb-6 hover:bg-[#f4efe6] transition-colors"
        >
          <ArrowLeft size={14} />
          <span>Back to Investigation</span>
        </button>
        <ErrorBanner message={error || 'Dossier not found.'} />
      </div>
    );
  }

  const explanation = report.explanation;
  const contradictoryEvidence = explanation?.contradictoryEvidence || [];
  const uncertainty = explanation?.uncertainty;

  return (
    <div className="editorial-container pt-8 sm:pt-10">
      {/* Dossier Action Bar */}
      <div className="flex items-center justify-between mb-6 flex-wrap gap-3 pb-4 border-b border-[#ebe4d8]">
        <div className="flex items-center gap-2">
          <button
            onClick={() => navigate('/')}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-white border border-[#dfd5c6] text-xs font-semibold text-[#1c1d22] hover:bg-[#f4efe6] transition-colors shadow-xs"
          >
            <ArrowLeft size={13} />
            <span>Back</span>
          </button>
          <button
            onClick={() => navigate('/')}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-[#1c1d22] text-xs font-semibold text-white hover:bg-[#2e3036] transition-colors shadow-xs"
          >
            <PlusCircle size={13} />
            <span>New Research</span>
          </button>
        </div>

        <button
          onClick={handlePrint}
          className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-md bg-white border border-[#dfd5c6] text-xs font-semibold text-[#1c1d22] hover:bg-[#f4efe6] transition-colors shadow-xs"
        >
          <Printer size={13} />
          <span>Print / Export PDF</span>
        </button>
      </div>

      {/* 1. Claim Masthead */}
      <div className="mb-6 pb-5 border-b border-[#ebe4d8]">
        <div className="flex items-center gap-1.5 text-xs font-mono font-bold uppercase tracking-wider text-[#15803d] mb-1.5">
          <Shield size={14} />
          <span>TruthLens Verified Intelligence Dossier</span>
        </div>
        <h1 className="font-serif text-2xl sm:text-4xl font-bold text-[#1c1d22] leading-tight mb-2">
          "{report.claim}"
        </h1>
        <div className="font-mono text-xs text-[#787b85]">
          Record ID: {report.id} • Evaluated: {new Date(report.createdAt).toLocaleString()}
        </div>
      </div>

      {/* 2. Verdict & 3. Confidence */}
      <VerdictCard verdict={report.verdict} confidence={report.confidence} />

      {/* 4. Probability Distribution */}
      <ProbabilityDistribution probabilities={report.probabilities} />

      {/* 5. Executive Summary & 6. Verdict Rationale */}
      {explanation ? (
        <div className="bg-white border border-[#ebe4d8] rounded-xl p-6 sm:p-7 mb-6 shadow-[0_1px_4px_rgba(0,0,0,0.03)]">
          <div className="flex items-center justify-between pb-4 mb-5 border-b border-[#f4efe6] flex-wrap gap-2">
            <div className="flex items-center gap-2 font-serif font-bold text-xl text-[#1c1d22]">
              <Bot size={20} className="text-[#1c1d22]" />
              <span>Evidence-Grounded Factual Synthesis</span>
            </div>
            <span className="px-2.5 py-0.5 rounded-full text-xs font-mono bg-[#f0fdf4] text-[#15803d] border border-[#bbf7d0]">
              Grounded in Citations
            </span>
          </div>

          {explanation.summary && (
            <div className="mb-6">
              <div className="text-[11px] font-mono font-bold uppercase tracking-wider text-[#787b85] mb-1.5">
                Executive Summary
              </div>
              <p className="font-serif text-base sm:text-lg text-[#1c1d22] leading-relaxed bg-[#fbf9f5] p-4 rounded-lg border border-[#f4efe6]">
                {explanation.summary}
              </p>
            </div>
          )}

          {explanation.verdictExplanation && (
            <div>
              <div className="text-[11px] font-mono font-bold uppercase tracking-wider text-[#787b85] mb-1.5">
                Verdict Rationale
              </div>
              <p className="text-sm text-[#44474f] leading-relaxed">
                {explanation.verdictExplanation}
              </p>
            </div>
          )}
        </div>
      ) : (
        <div className="bg-white border-l-4 border-l-[#b45309] border border-[#ebe4d8] rounded-xl p-6 mb-6 shadow-sm">
          <div className="flex items-center justify-between pb-3 mb-3 border-b border-[#f4efe6]">
            <div className="flex items-center gap-2 font-serif font-bold text-base text-[#1c1d22]">
              <Bot size={18} className="text-[#b45309]" />
              <span>Synthesis Layer Status</span>
            </div>
            <span className="px-2.5 py-0.5 rounded text-xs font-mono bg-[#fffbeb] text-[#b45309] border border-[#fde68a]">
              Neural Verdict Active
            </span>
          </div>
          <p className="text-sm text-[#5d6069] leading-relaxed">
            Automated LLM explanation is currently unconfigured or unavailable. The neural Model B verdict, confidence probabilities, and verified web evidence remain fully authoritative.
          </p>
        </div>
      )}

      {/* 7. Evidence Passages */}
      {report.evidenceBySource && report.evidenceBySource.length > 0 ? (
        <div className="bg-white border border-[#ebe4d8] rounded-xl p-6 sm:p-7 mb-6 shadow-[0_1px_4px_rgba(0,0,0,0.03)]">
          <div className="flex items-center justify-between pb-4 mb-5 border-b border-[#f4efe6] flex-wrap gap-2">
            <div className="flex items-center gap-2 font-serif font-bold text-xl text-[#1c1d22]">
              <Globe size={19} className="text-[#1c1d22]" />
              <span>Verified Evidence Passages Grouped by Source</span>
            </div>
            <span className="px-2.5 py-0.5 rounded-full text-xs font-mono bg-[#f4efe6] text-[#5d6069] border border-[#dfd5c6]">
              {report.evidence?.length || 0} Passages Evaluated
            </span>
          </div>

          <div className="space-y-4">
            {report.evidenceBySource.map((sourceGroup, groupIndex) => (
              <div key={groupIndex} className="bg-[#fcfbf8] border border-[#ebe4d8] rounded-xl p-4 sm:p-5">
                <div className="flex items-center justify-between pb-2 mb-3 border-b border-[#ebe4d8] flex-wrap gap-2">
                  <div>
                    <div className="text-[11px] font-mono font-bold uppercase tracking-wider text-[#787b85]">
                      {sourceGroup.sourceName || 'News Article'}
                    </div>
                    <h4 className="font-serif text-base font-bold text-[#1c1d22]">{sourceGroup.title || 'Source Title'}</h4>
                  </div>
                  {sourceGroup.url && (
                    <a
                      href={sourceGroup.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-xs font-semibold text-[#1c1d22] hover:text-[#15803d] transition-colors underline decoration-[#dfd5c6]"
                    >
                      Open Source ↗
                    </a>
                  )}
                </div>

                <div className="space-y-2.5">
                  {sourceGroup.passages.map((passage, passageIndex) => {
                    const isSupport = passage.verification?.label === 'SUPPORTS';
                    const isRefute = passage.verification?.label === 'REFUTES';
                    const badgeClass = isSupport
                      ? 'bg-[#f0fdf4] text-[#15803d] border-[#bbf7d0]'
                      : isRefute
                        ? 'bg-[#fef2f2] text-[#b91c1c] border-[#fecaca]'
                        : 'bg-[#fffbeb] text-[#b45309] border-[#fde68a]';

                    return (
                      <div key={passageIndex} className="bg-white border border-[#ebe4d8] rounded-lg p-3.5">
                        <p className="font-serif text-sm text-[#1c1d22] italic leading-relaxed mb-2.5">
                          "{passage.text}"
                        </p>
                        <div className="flex items-center gap-2 text-xs flex-wrap">
                          <span className="font-mono text-[11px] bg-[#f4efe6] px-2 py-0.5 rounded border border-[#dfd5c6] text-[#44474f]">
                            Model A: {passage.relevanceScore?.toFixed(3)}
                          </span>
                          <span className={`font-mono text-[11px] font-semibold px-2 py-0.5 rounded border ${badgeClass}`}>
                            Model B: {passage.verification?.label} ({((passage.verification?.confidence || 0) * 100).toFixed(1)}%)
                          </span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>
        </div>
      ) : (
        <div className="bg-white border border-[#ebe4d8] rounded-xl p-6 sm:p-7 mb-6 shadow-sm">
          <div className="flex items-center gap-2 font-serif font-bold text-xl text-[#1c1d22] pb-4 mb-5 border-b border-[#f4efe6]">
            <FileText size={18} />
            <span>Verified Evidence Passages ({report.evidence?.length || 0})</span>
          </div>
          <div className="space-y-3">
            {(report.evidence || []).map((evidenceItem, index) => (
              <EvidenceCard key={index} evidence={evidenceItem} index={index} />
            ))}
          </div>
        </div>
      )}

      {/* 8. Contradictory Evidence */}
      {contradictoryEvidence.length > 0 && (
        <div className="bg-[#fef2f2] border border-[#fecaca] rounded-xl p-5 sm:p-6 mb-6 shadow-xs">
          <div className="flex items-center gap-2 text-xs font-mono font-bold uppercase tracking-wider text-[#b91c1c] mb-3 pb-2 border-b border-[#fecaca]">
            <AlertTriangle size={15} />
            <span>Counter-Claims & Contradictory Information</span>
          </div>
          <ul className="space-y-1.5 pl-5 list-disc text-sm text-[#7f1d1d] leading-relaxed">
            {contradictoryEvidence.map((point, index) => (
              <li key={index}>
                {point}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* 9. Discovered Sources */}
      <SourcesSection sources={report.sources} />

      {/* 10. Evidentiary Uncertainty */}
      {uncertainty && (
        <div className="bg-white border border-[#ebe4d8] rounded-xl p-6 mb-6 shadow-sm">
          <div className="flex items-center gap-2 text-xs font-mono font-bold uppercase tracking-wider text-[#b45309] pb-3 mb-3 border-b border-[#f4efe6]">
            <HelpCircle size={15} />
            <span>Investigative Limitations & Uncertainty</span>
          </div>
          <p className="text-sm text-[#5d6069] leading-relaxed bg-[#fffbeb] p-3.5 rounded-lg border border-[#fde68a]">
            {uncertainty}
          </p>
        </div>
      )}

      {/* 11. Technical Metadata */}
      <TechnicalMetadata retrieval={report.retrieval} llm={report.llm} id={report.id} />
    </div>
  );
}
