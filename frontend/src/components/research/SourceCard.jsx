import React from 'react';
import { ExternalLink, Calendar } from 'lucide-react';

export default function SourceCard({ source }) {
  const { title, url, source: publisher, publishedAt } = source;

  return (
    <div className="bg-[#fcfbf8] border border-[#ebe4d8] rounded-xl p-4 flex flex-col justify-between hover:border-[#dfd5c6] hover:bg-white hover:shadow-xs transition-all">
      <div>
        <div className="text-[11px] font-mono font-bold uppercase tracking-wider text-[#787b85] mb-1">
          {publisher || 'Archival Reporting'}
        </div>
        <h4 className="font-serif text-[15px] font-semibold text-[#1c1d22] leading-snug mb-3 line-clamp-2">
          {title || 'Untitled Source'}
        </h4>
      </div>

      <div className="flex items-center justify-between pt-2.5 border-t border-[#f4efe6] text-xs text-[#787b85]">
        {publishedAt ? (
          <div className="flex items-center gap-1 text-[11px] font-mono">
            <Calendar size={12} className="text-[#9b9ea7]" />
            <span>{new Date(publishedAt).toLocaleDateString()}</span>
          </div>
        ) : (
          <span />
        )}

        {url ? (
          <a
            href={url}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1 font-medium text-[#1c1d22] hover:text-[#15803d] text-xs transition-colors"
          >
            <span>Open Article</span>
            <ExternalLink size={11} />
          </a>
        ) : (
          <span className="text-[11px] text-[#9b9ea7]">Direct Record</span>
        )}
      </div>
    </div>
  );
}
