import React from 'react';
import { FileQuestion } from 'lucide-react';

export default function EmptyState({
  title = 'No records found',
  message = 'Try searching with different claim keywords or submit a new inquiry.',
  icon: IconComponent = FileQuestion
}) {
  return (
    <div className="bg-white border border-[#ebe4d8] rounded-xl p-10 text-center shadow-[0_1px_3px_rgba(0,0,0,0.03)]">
      <div className="w-12 h-12 rounded-full bg-[#f4efe6] text-[#787b85] flex items-center justify-center mx-auto mb-3.5 border border-[#dfd5c6]">
        <IconComponent size={22} />
      </div>
      <h3 className="font-serif text-lg font-semibold text-[#1c1d22]">{title}</h3>
      <p className="text-sm text-[#5d6069] max-w-md mx-auto mt-1.5 leading-relaxed">{message}</p>
    </div>
  );
}
