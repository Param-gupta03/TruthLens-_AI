import React, { useState, useEffect } from 'react';
import { NavLink, Link } from 'react-router-dom';
import { ShieldCheck, Search, History, Edit3, Cpu } from 'lucide-react';
import { checkHealth } from '../../services/api';

export default function Navbar() {
  const [healthStatus, setHealthStatus] = useState({ backendOnline: true, mlOnline: true });

  useEffect(() => {
    checkHealth().then(setHealthStatus);
    const healthInterval = setInterval(() => {
      checkHealth().then(setHealthStatus);
    }, 20000);
    return () => clearInterval(healthInterval);
  }, []);

  const isSystemOnline = healthStatus.backendOnline && healthStatus.mlOnline;

  return (
    <header className="sticky top-0 z-50 bg-[#faf7f0]/95 backdrop-blur-md border-b border-[#ebe4d8]">
      <div className="editorial-container flex items-center justify-between h-16">
        <Link to="/" className="flex items-center gap-3 group">
          <div className="w-8 h-8 rounded bg-[#1c1d22] text-[#faf7f0] flex items-center justify-center shadow-sm group-hover:bg-[#2e3036] transition-colors">
            <ShieldCheck size={18} />
          </div>
          <div className="flex flex-col">
            <span className="font-serif text-xl font-bold tracking-tight text-[#1c1d22] leading-none">
              TruthLens
            </span>
            <span className="text-[10px] tracking-wider uppercase font-mono text-[#787b85] mt-0.5">
              Evidence Research
            </span>
          </div>
        </Link>

        <nav className="flex items-center gap-1.5 sm:gap-2">
          <NavLink
            to="/"
            className={({ isActive }) =>
              `flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
                isActive
                  ? 'bg-white text-[#1c1d22] shadow-sm border border-[#dfd5c6]'
                  : 'text-[#5d6069] hover:text-[#1c1d22] hover:bg-[#f4efe6]'
              }`
            }
          >
            <Search size={14} />
            <span className="hidden sm:inline">Research</span>
          </NavLink>

          <NavLink
            to="/history"
            className={({ isActive }) =>
              `flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
                isActive
                  ? 'bg-white text-[#1c1d22] shadow-sm border border-[#dfd5c6]'
                  : 'text-[#5d6069] hover:text-[#1c1d22] hover:bg-[#f4efe6]'
              }`
            }
          >
            <History size={14} />
            <span className="hidden sm:inline">Archive</span>
          </NavLink>

          <NavLink
            to="/manual"
            className={({ isActive }) =>
              `flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
                isActive
                  ? 'bg-white text-[#1c1d22] shadow-sm border border-[#dfd5c6]'
                  : 'text-[#5d6069] hover:text-[#1c1d22] hover:bg-[#f4efe6]'
              }`
            }
          >
            <Edit3 size={14} />
            <span className="hidden sm:inline">Manual Check</span>
          </NavLink>

          <NavLink
            to="/architecture"
            className={({ isActive }) =>
              `flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
                isActive
                  ? 'bg-white text-[#1c1d22] shadow-sm border border-[#dfd5c6]'
                  : 'text-[#5d6069] hover:text-[#1c1d22] hover:bg-[#f4efe6]'
              }`
            }
          >
            <Cpu size={14} />
            <span className="hidden sm:inline">Methodology</span>
          </NavLink>

          <div
            className="ml-2 flex items-center gap-2 px-2.5 py-1 rounded-full bg-white border border-[#dfd5c6] text-[11px] font-mono text-[#5d6069]"
            title="Model Inference Server Health"
          >
            <span
              className={`w-2 h-2 rounded-full ${
                isSystemOnline ? 'bg-[#15803d]' : 'bg-[#b91c1c]'
              }`}
            />
            <span className="hidden md:inline font-semibold">
              {isSystemOnline ? 'ENGINE READY' : 'OFFLINE'}
            </span>
          </div>
        </nav>
      </div>
    </header>
  );
}
