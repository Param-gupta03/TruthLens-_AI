import React from 'react';
import { ShieldCheck, Cpu } from 'lucide-react';

export default function Footer() {
  return (
    <footer className="border-t border-[#ebe4d8] bg-[#f4efe6]/60 py-10 mt-auto">
      <div className="editorial-container flex flex-col md:flex-row items-start md:items-center justify-between gap-6">
        <div>
          <div className="flex items-center gap-2 text-base font-serif font-bold text-[#1c1d22]">
            <ShieldCheck size={18} className="text-[#15803d]" />
            <span>TruthLens Evidence & Investigation Platform</span>
          </div>
          <p className="text-xs text-[#5d6069] mt-1.5 max-w-xl leading-relaxed">
            Evidence-grounded claim verification utilizing RoBERTa sequence filtering (Model A) and SciFact transformer entailment (Model B) with attributable web citations.
          </p>
        </div>
      </div>
    </footer>
  );
}
