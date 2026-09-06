import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Search, ArrowUpRight, ChevronLeft, ChevronRight, History } from 'lucide-react';
import ErrorBanner from '../components/common/ErrorBanner';
import EmptyState from '../components/common/EmptyState';
import { getFactChecks } from '../services/api';

const VERDICT_FILTERS = ['ALL', 'SUPPORTS', 'REFUTES', 'NOT_ENOUGH_INFO'];

export default function HistoryPage() {
  const [records, setRecords] = useState([]);
  const [pagination, setPagination] = useState({ page: 1, limit: 10, total: 0, pages: 1 });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [activeFilter, setActiveFilter] = useState('ALL');
  const navigate = useNavigate();

  const loadHistory = (page = 1) => {
    setLoading(true);
    setError(null);
    getFactChecks(page, 10)
      .then(res => {
        setRecords(res.records || []);
        setPagination(res.pagination || { page: 1, limit: 10, total: 0, pages: 1 });
        setLoading(false);
      })
      .catch(err => {
        setError(err.message || 'Failed to load research history.');
        setLoading(false);
      });
  };

  useEffect(() => {
    loadHistory(1);
  }, []);

  const filteredRecords = records.filter(record => {
    const matchesSearch = !searchTerm || record.claim.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesVerdict = activeFilter === 'ALL' || record.verdict === activeFilter;
    return matchesSearch && matchesVerdict;
  });

  return (
    <div className="editorial-container py-8 md:py-12">
      {/* Header */}
      <div className="mb-8 border-b border-cream-300/80 pb-6">
        <div className="inline-flex items-center gap-2 px-2.5 py-1 rounded-full bg-cream-200/80 border border-cream-300 text-xs font-mono font-medium text-ink-700 mb-3">
          <History size={13} className="text-ink-600" />
          <span>ARCHIVAL DOSSIER LEDGER</span>
        </div>
        <h1 className="text-2xl md:text-3xl font-serif font-bold text-ink-950 tracking-tight">
          Previous Fact-Check Investigations
        </h1>
        <p className="text-sm md:text-base text-ink-600 max-w-2xl mt-2 leading-relaxed">
          Browse historical factual claims, dual-model verification verdicts, confidence levels, and evidentiary reporting dossiers.
        </p>
      </div>

      {/* Controls: Search + Verdict Filters */}
      <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-4 mb-6">
        <div className="relative flex-1 max-w-md">
          <Search size={16} className="absolute left-3.5 top-1/2 -translate-y-1/2 text-ink-400 pointer-events-none" />
          <input
            type="text"
            placeholder="Search claims in archive..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-10 pr-4 py-2 text-sm bg-white border border-cream-300 rounded-lg text-ink-900 placeholder:text-ink-400 focus:outline-none focus:ring-2 focus:ring-ink-900/10 focus:border-ink-900 transition-colors shadow-2xs"
          />
        </div>

        <div className="inline-flex items-center gap-1 p-1 bg-cream-200/80 rounded-lg border border-cream-300/80 self-start sm:self-auto overflow-x-auto max-w-full">
          {VERDICT_FILTERS.map((verdictFilter) => {
            const isActive = activeFilter === verdictFilter;
            return (
              <button
                key={verdictFilter}
                type="button"
                onClick={() => setActiveFilter(verdictFilter)}
                className={`px-3 py-1.5 text-xs font-medium rounded-md whitespace-nowrap transition-all ${
                  isActive
                    ? 'bg-white text-ink-950 shadow-2xs font-semibold border border-cream-300/90'
                    : 'text-ink-600 hover:text-ink-900 hover:bg-white/50'
                }`}
              >
                {verdictFilter.replace(/_/g, ' ')}
              </button>
            );
          })}
        </div>
      </div>

      <ErrorBanner message={error} onRetry={() => loadHistory(pagination.page)} />

      {loading ? (
        <div className="editorial-card p-12 text-center my-6">
          <div className="inline-block w-6 h-6 border-2 border-ink-900/20 border-t-ink-900 rounded-full animate-spin mb-3" />
          <p className="text-xs font-mono text-ink-500 uppercase tracking-wider">Loading archive ledger...</p>
        </div>
      ) : filteredRecords.length === 0 ? (
        <EmptyState
          title="No records found"
          message={searchTerm || activeFilter !== 'ALL' ? 'No historical records match your filter criteria.' : 'No fact checks recorded yet. Run your first research query on the Research page!'}
        />
      ) : (
        <>
          <div className="space-y-3">
            {filteredRecords.map((recordItem) => {
              const normalizedVerdict = (recordItem.verdict || 'not-enough-info').toLowerCase().replace(/_/g, '-');
              
              let verdictBadgeClass = 'bg-amber-neutral-bg text-amber-neutral-dark border-amber-neutral-border';
              if (normalizedVerdict === 'supports') {
                verdictBadgeClass = 'bg-truth-green-bg text-truth-green-dark border-truth-green-border';
              } else if (normalizedVerdict === 'refutes') {
                verdictBadgeClass = 'bg-false-red-bg text-false-red-dark border-false-red-border';
              }

              return (
                <div
                  key={recordItem.id}
                  className="editorial-card-interactive p-4 md:p-5 cursor-pointer flex flex-col sm:flex-row sm:items-center justify-between gap-4 group"
                  onClick={() => navigate(`/report/${recordItem.id}`)}
                >
                  <div className="flex-1 min-w-0">
                    <div className="font-serif text-base md:text-lg font-semibold text-ink-950 group-hover:text-black leading-snug line-clamp-2">
                      "{recordItem.claim}"
                    </div>
                    <div className="flex flex-wrap items-center gap-2 mt-2 text-xs font-mono text-ink-500">
                      <span>{new Date(recordItem.createdAt).toLocaleDateString()} at {new Date(recordItem.createdAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
                      <span>•</span>
                      <span>{recordItem.evidence?.length || 0} evidence passages</span>
                      <span>•</span>
                      <span>{recordItem.sourcesCount || 0} sources</span>
                      {recordItem.llm?.status === 'generated' && (
                        <>
                          <span>•</span>
                          <span className="inline-flex items-center px-1.5 py-0.5 rounded bg-cream-200 text-ink-700 font-sans font-medium text-[11px]">
                            Synthesized
                          </span>
                        </>
                      )}
                    </div>
                  </div>

                  <div className="flex items-center justify-between sm:justify-end gap-3 shrink-0 pt-2 sm:pt-0 border-t sm:border-t-0 border-cream-200">
                    <span className={`inline-flex items-center px-2.5 py-1 rounded-full text-xs font-mono font-semibold border ${verdictBadgeClass}`}>
                      {recordItem.verdict} ({recordItem.confidencePercent}%)
                    </span>
                    <div className="p-1.5 rounded-md text-ink-400 group-hover:text-ink-950 group-hover:translate-x-0.5 group-hover:-translate-y-0.5 transition-all">
                      <ArrowUpRight size={16} />
                    </div>
                  </div>
                </div>
              );
            })}
          </div>

          {pagination.pages > 1 && (
            <div className="flex items-center justify-between mt-8 pt-4 border-t border-cream-300">
              <button
                type="button"
                disabled={pagination.page <= 1}
                onClick={() => loadHistory(pagination.page - 1)}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg bg-white border border-cream-300 text-ink-700 hover:text-ink-950 hover:bg-cream-100 disabled:opacity-40 disabled:cursor-not-allowed transition-colors shadow-2xs"
              >
                <ChevronLeft size={15} />
                <span>Previous</span>
              </button>

              <span className="text-xs font-mono text-ink-500">
                Page {pagination.page} of {pagination.pages}
              </span>

              <button
                type="button"
                disabled={pagination.page >= pagination.pages}
                onClick={() => loadHistory(pagination.page + 1)}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg bg-white border border-cream-300 text-ink-700 hover:text-ink-950 hover:bg-cream-100 disabled:opacity-40 disabled:cursor-not-allowed transition-colors shadow-2xs"
              >
                <span>Next</span>
                <ChevronRight size={15} />
              </button>
            </div>
          )}
        </>
      )}
    </div>
  );
}
