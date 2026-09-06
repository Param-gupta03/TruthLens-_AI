import React from 'react';
import { AlertTriangle, RefreshCw } from 'lucide-react';

export default function ErrorBanner({ message, onRetry }) {
  if (!message) return null;

  return (
    <div className="bg-[#fef2f2] border border-[#fecaca] rounded-lg p-4 mb-6 flex items-center justify-between gap-4 shadow-sm" role="alert">
      <div className="flex items-center gap-3 text-sm text-[#b91c1c]">
        <AlertTriangle size={18} className="shrink-0 text-[#b91c1c]" />
        <span className="font-medium">{message}</span>
      </div>
      {onRetry && (
        <button
          onClick={onRetry}
          className="px-3 py-1.5 rounded-md bg-white border border-[#fecaca] text-[#b91c1c] text-xs font-semibold hover:bg-[#fee2e2] transition-colors flex items-center gap-1.5 shrink-0"
        >
          <RefreshCw size={13} />
          <span>Retry</span>
        </button>
      )}
    </div>
  );
}
