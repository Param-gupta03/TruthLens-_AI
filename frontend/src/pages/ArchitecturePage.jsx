import React from 'react';
import { Cpu, CheckCircle2, Bot, Globe, Shield, Database, Activity, Lock, Layers, Zap, Server, BarChart3, ArrowDown } from 'lucide-react';
import ResearchTimeline from '../components/research/ResearchTimeline';

export default function ArchitecturePage() {
  return (
    <div className="editorial-container py-8 md:py-12 space-y-10">
      {/* Header */}
      <div className="border-b border-cream-300/80 pb-6">
        <div className="inline-flex items-center gap-2 px-2.5 py-1 rounded-full bg-cream-200/80 border border-cream-300 text-xs font-mono font-medium text-ink-700 mb-3">
          <Layers size={13} className="text-ink-600" />
          <span>SYSTEM ARCHITECTURE & SPECIFICATION</span>
        </div>
        <h1 className="text-2xl md:text-4xl font-serif font-bold text-ink-950 tracking-tight">
          TruthLens AI Architecture & Metrics
        </h1>
        <p className="text-sm md:text-base text-ink-600 max-w-3xl mt-2 leading-relaxed">
          Comprehensive technical specification of the dual-model neural verification engine, web retrieval crawler, LLM explanation layer, and production security controls.
        </p>
      </div>

      {/* Trust & Transparency Pipeline */}
      <ResearchTimeline />

      {/* Visual System Architecture Diagram */}
      <div className="editorial-card p-6 md:p-8 space-y-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-cream-200 pb-4">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-cream-200 text-ink-800">
              <Server size={18} />
            </div>
            <div>
              <h2 className="font-serif text-lg font-bold text-ink-950">End-to-End Pipeline Flow</h2>
              <p className="text-xs text-ink-500 font-sans">Distributed orchestration between Node.js and FastAPI PyTorch</p>
            </div>
          </div>
          <span className="self-start sm:self-auto text-xs font-mono font-semibold px-2.5 py-1 rounded-full bg-cream-200 border border-cream-300 text-ink-700">
            MERN + Python ML + CUDA
          </span>
        </div>

        {/* Structured Pipeline Steps */}
        <div className="p-5 md:p-6 rounded-lg bg-cream-100/50 border border-cream-300 font-mono text-xs md:text-sm text-ink-800 space-y-2 overflow-x-auto">
          <div className="min-w-[640px] space-y-1.5 leading-relaxed">
            <div className="text-ink-950 font-bold flex items-center gap-2">
              <span className="inline-block w-2 h-2 rounded-full bg-ink-900" />
              <span>User Claim Input (Browser)</span>
            </div>
            <div className="text-ink-400 pl-4">│</div>
            <div className="pl-4 flex items-center gap-2 text-ink-900">
              <span>▼</span>
              <span className="font-semibold text-ink-950">React 18 SPA</span>
              <span className="text-ink-500 text-xs font-sans">(Vite, Responsive Editorial UI, State Management)</span>
            </div>
            <div className="text-ink-400 pl-4">│ <span className="text-xs font-sans text-ink-500">HTTP POST /api/research</span></div>
            <div className="pl-4 flex items-center gap-2 text-ink-900">
              <span>▼</span>
              <span className="font-semibold text-ink-950">Node.js + Express Backend</span>
              <span className="text-ink-500 text-xs font-sans">(Helmet, CORS, Rate Limiting, Input Validation)</span>
            </div>
            <div className="text-ink-400 pl-4">│</div>
            <div className="pl-4 flex items-center gap-2 text-ink-900">
              <span>▼</span>
              <span className="font-semibold text-ink-950">Automated Web & News Retrieval</span>
              <span className="text-ink-500 text-xs font-sans">(Multi-query search, SSRF-safe scraper, sentence chunking)</span>
            </div>
            <div className="text-ink-400 pl-4">│ <span className="text-xs font-sans text-ink-500">HTTP POST to FastAPI (:8000)</span></div>
            <div className="pl-4 flex items-center gap-2 text-ink-900">
              <span>▼</span>
              <span className="font-semibold text-truth-green-dark">Model A: Evidence Relevance</span>
              <span className="text-ink-500 text-xs font-sans">(RoBERTa-base on NVIDIA RTX 2050 CUDA, Threshold 0.35)</span>
            </div>
            <div className="text-ink-400 pl-4">│ <span className="text-xs font-sans text-ink-500">Passages with P(RELEVANT) ≥ 0.35</span></div>
            <div className="pl-4 flex items-center gap-2 text-ink-900">
              <span>▼</span>
              <span className="font-semibold text-truth-green-dark">Model B: Claim Verification</span>
              <span className="text-xs font-mono font-bold px-1.5 py-0.5 rounded bg-truth-green-bg text-truth-green-dark border border-truth-green-border">FINAL VERDICT AUTHORITY</span>
            </div>
            <div className="text-ink-400 pl-4">│ <span className="text-xs font-sans text-ink-500">Fixed Verdict: SUPPORTS / REFUTES / NOT_ENOUGH_INFO</span></div>
            <div className="pl-4 flex items-center gap-2 text-ink-900">
              <span>▼</span>
              <span className="font-semibold text-ink-950">LLM Explanation Layer</span>
              <span className="text-ink-500 text-xs font-sans">(gpt-4o-mini / Gemini, &lt;UNTRUSTED_EVIDENCE&gt; boundary, Explanation Only)</span>
            </div>
            <div className="text-ink-400 pl-4">│</div>
            <div className="pl-4 flex items-center gap-2 text-ink-900">
              <span>▼</span>
              <span className="font-semibold text-ink-950">MongoDB Database</span>
              <span className="text-ink-500 text-xs font-sans">(Indexed FactCheck collection, 2-hour TTL cache)</span>
            </div>
            <div className="text-ink-400 pl-4">│</div>
            <div className="pl-4 flex items-center gap-2 text-ink-950 font-bold">
              <span>▼</span>
              <span>Interactive Research Report & Dossier (React UI)</span>
            </div>
          </div>
        </div>
      </div>

      {/* Model Information Section */}
      <div className="space-y-6">
        <div className="border-b border-cream-300 pb-3">
          <h2 className="text-xl md:text-2xl font-serif font-bold text-ink-950">
            Machine Learning Models & Hardware
          </h2>
          <p className="text-xs md:text-sm text-ink-600 font-sans mt-1">
            Dual-transformer verification pipeline with strict authority separation.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Model A */}
          <div className="editorial-card p-6 flex flex-col justify-between space-y-4">
            <div>
              <div className="flex items-center justify-between pb-3 border-b border-cream-200 mb-3">
                <div className="flex items-center gap-2">
                  <Cpu size={18} className="text-ink-700" />
                  <span className="font-serif font-bold text-ink-950 text-base">Model A: Relevance</span>
                </div>
                <span className="text-xs font-mono font-medium px-2 py-0.5 rounded bg-cream-200 border border-cream-300 text-ink-700">
                  RoBERTa-base
                </span>
              </div>
              <p className="text-xs md:text-sm text-ink-600 leading-relaxed mb-4">
                Determines whether candidate web passages are semantically relevant to the claim, acting as an empirical filter to eliminate retrieval noise.
              </p>
              <div className="space-y-2 text-xs border-t border-cream-200 pt-3">
                <div className="flex justify-between py-1 border-b border-cream-100">
                  <span className="text-ink-500 font-mono">Architecture:</span>
                  <span className="font-mono font-medium text-ink-900">Sequence Classifier</span>
                </div>
                <div className="flex justify-between py-1 border-b border-cream-100">
                  <span className="text-ink-500 font-mono">Threshold:</span>
                  <span className="font-mono font-semibold text-truth-green-dark">0.35 (Tuned)</span>
                </div>
                <div className="flex justify-between py-1 border-b border-cream-100">
                  <span className="text-ink-500 font-mono">Test Split:</span>
                  <span className="font-mono text-ink-900">1,565 pairs</span>
                </div>
                <div className="flex justify-between py-1">
                  <span className="text-ink-500 font-mono">Throughput:</span>
                  <span className="font-mono font-semibold text-ink-950">150.04 samples / s</span>
                </div>
              </div>
            </div>

            <div className="p-3 bg-cream-100/50 rounded-lg border border-cream-300 text-xs text-ink-700 space-y-1 font-sans">
              <div className="font-medium text-ink-900">Empirical Benchmarks:</div>
              <div>• <strong>Accuracy:</strong> 90.35% (+7.15% over baseline)</div>
              <div>• <strong>Macro F1:</strong> 78.41% | <strong>Weighted F1:</strong> 90.46%</div>
              <div>• <strong>ROC-AUC:</strong> 0.9032 | <strong>PR-AUC:</strong> 0.6534</div>
              <div>• <strong>Early Exit:</strong> Zero-passages triggers fast return</div>
            </div>
          </div>

          {/* Model B */}
          <div className="editorial-card p-6 flex flex-col justify-between space-y-4">
            <div>
              <div className="flex items-center justify-between pb-3 border-b border-cream-200 mb-3">
                <div className="flex items-center gap-2">
                  <CheckCircle2 size={18} className="text-truth-green-dark" />
                  <span className="font-serif font-bold text-ink-950 text-base">Model B: Verification</span>
                </div>
                <span className="text-xs font-mono font-semibold px-2 py-0.5 rounded bg-truth-green-bg text-truth-green-dark border border-truth-green-border">
                  SciFact RoBERTa
                </span>
              </div>
              <p className="text-xs md:text-sm text-ink-600 leading-relaxed mb-4">
                The authoritative verdict decision-maker. Evaluates claim-evidence entailment to classify: <span className="font-semibold text-truth-green-dark">SUPPORTS</span>, <span className="font-semibold text-false-red-dark">REFUTES</span>, or <span className="font-semibold text-amber-neutral-dark">NOT_ENOUGH_INFO</span>.
              </p>
              <div className="space-y-2 text-xs border-t border-cream-200 pt-3">
                <div className="flex justify-between py-1 border-b border-cream-100">
                  <span className="text-ink-500 font-mono">Pretrained Base:</span>
                  <span className="font-mono font-medium text-ink-900">RoBERTa Fine-tuned</span>
                </div>
                <div className="flex justify-between py-1 border-b border-cream-100">
                  <span className="text-ink-500 font-mono">Benchmark:</span>
                  <span className="font-mono text-ink-900">SciFact Benchmark</span>
                </div>
                <div className="flex justify-between py-1 border-b border-cream-100">
                  <span className="text-ink-500 font-mono">Validation Size:</span>
                  <span className="font-mono text-ink-900">448 grounded pairs</span>
                </div>
                <div className="flex justify-between py-1">
                  <span className="text-ink-500 font-mono">Decision Rule:</span>
                  <span className="font-mono font-bold text-truth-green-dark">Final Verdict Authority</span>
                </div>
              </div>
            </div>

            <div className="p-3 bg-cream-100/50 rounded-lg border border-cream-300 text-xs text-ink-700 space-y-1 font-sans">
              <div className="font-medium text-ink-900">Empirical Benchmarks:</div>
              <div>• <strong>Validation Accuracy:</strong> 78.57%</div>
              <div>• <strong>Macro F1:</strong> 79.00% | <strong>Weighted:</strong> 78.31%</div>
              <div>• <strong>Macro Precision:</strong> 79.38% | <strong>Recall:</strong> 78.76%</div>
              <div>• <strong>SUPPORTS F1:</strong> 78.38% | <strong>REFUTES:</strong> 58.62%</div>
            </div>
          </div>

          {/* LLM Layer */}
          <div className="editorial-card p-6 flex flex-col justify-between space-y-4">
            <div>
              <div className="flex items-center justify-between pb-3 border-b border-cream-200 mb-3">
                <div className="flex items-center gap-2">
                  <Bot size={18} className="text-ink-700" />
                  <span className="font-serif font-bold text-ink-950 text-base">LLM Synthesis Layer</span>
                </div>
                <span className="text-xs font-mono font-medium px-2 py-0.5 rounded bg-cream-200 border border-cream-300 text-ink-700">
                  gpt-4o-mini / Gemini
                </span>
              </div>
              <p className="text-xs md:text-sm text-ink-600 leading-relaxed mb-4">
                Converts verified evidence and neural predictions into a structured human dossier (Executive Summary, Verdict Rationale, Counter-evidence, Uncertainty).
              </p>
              <div className="space-y-2 text-xs border-t border-cream-200 pt-3">
                <div className="flex justify-between py-1 border-b border-cream-100">
                  <span className="text-ink-500 font-mono">Default Engine:</span>
                  <span className="font-mono font-medium text-ink-900">OpenAI gpt-4o-mini</span>
                </div>
                <div className="flex justify-between py-1 border-b border-cream-100">
                  <span className="text-ink-500 font-mono">Temperature:</span>
                  <span className="font-mono text-ink-900">0.2 (Low Hallucination)</span>
                </div>
                <div className="flex justify-between py-1 border-b border-cream-100">
                  <span className="text-ink-500 font-mono">Context Budget:</span>
                  <span className="font-mono text-ink-900">Top-5 Verified Passages</span>
                </div>
                <div className="flex justify-between py-1">
                  <span className="text-ink-500 font-mono">Supremacy:</span>
                  <span className="font-mono font-semibold text-false-red-dark">Cannot Alter Verdict</span>
                </div>
              </div>
            </div>

            <div className="p-3 bg-cream-100/50 rounded-lg border border-cream-300 text-xs text-ink-700 space-y-1 font-sans">
              <div className="font-medium text-ink-900">Operational Defenses:</div>
              <div>• <strong>Prompt Boundary:</strong> Tagged with &lt;UNTRUSTED_EVIDENCE&gt;</div>
              <div>• <strong>Consistency Guard:</strong> Mismatches rejected automatically</div>
              <div>• <strong>Graceful Fallback:</strong> Neural verification works offline</div>
            </div>
          </div>
        </div>
      </div>

      {/* Hardware & GPU Acceleration */}
      <div className="editorial-card p-6 md:p-8 space-y-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-cream-200 pb-4">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-cream-200 text-ink-800">
              <Zap size={18} />
            </div>
            <div>
              <h2 className="font-serif text-lg font-bold text-ink-950">Hardware Infrastructure & GPU Acceleration</h2>
              <p className="text-xs text-ink-500 font-sans">Empirical inference configuration and resource allocation</p>
            </div>
          </div>
          <span className="self-start sm:self-auto text-xs font-mono font-semibold px-2.5 py-1 rounded-full bg-cream-200 border border-cream-300 text-ink-700">
            NVIDIA CUDA Runtime
          </span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="p-4 rounded-lg bg-cream-100/50 border border-cream-300 space-y-1">
            <div className="text-xs font-mono text-ink-500 uppercase tracking-wider">GPU Accelerator</div>
            <div className="font-serif text-base font-bold text-ink-950">NVIDIA RTX 2050</div>
            <div className="text-xs text-ink-600 font-mono">Dedicated 4.0 GB VRAM</div>
          </div>
          <div className="p-4 rounded-lg bg-cream-100/50 border border-cream-300 space-y-1">
            <div className="text-xs font-mono text-ink-500 uppercase tracking-wider">Compute Stack</div>
            <div className="font-serif text-base font-bold text-ink-950">CUDA 12.4 + cuDNN</div>
            <div className="text-xs text-ink-600 font-mono">PyTorch FP16 Precision</div>
          </div>
          <div className="p-4 rounded-lg bg-cream-100/50 border border-cream-300 space-y-1">
            <div className="text-xs font-mono text-ink-500 uppercase tracking-wider">Inference Footprint</div>
            <div className="font-serif text-base font-bold text-ink-950">~953 MB Combined</div>
            <div className="text-xs text-ink-600 font-mono">Models Loaded Once</div>
          </div>
          <div className="p-4 rounded-lg bg-cream-100/50 border border-cream-300 space-y-1">
            <div className="text-xs font-mono text-ink-500 uppercase tracking-wider">Fault Tolerance</div>
            <div className="font-serif text-base font-bold text-ink-950">CPU Auto-Fallback</div>
            <div className="text-xs text-ink-600 font-mono">Non-GPU Environments</div>
          </div>
        </div>
      </div>

      {/* Latency Benchmarks & Empirical Table */}
      <div className="editorial-card p-6 md:p-8 space-y-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-cream-200 pb-4">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-cream-200 text-ink-800">
              <BarChart3 size={18} />
            </div>
            <div>
              <h2 className="font-serif text-lg font-bold text-ink-950">Verified Latency & Throughput Metrics</h2>
              <p className="text-xs text-ink-500 font-sans">End-to-end benchmark timings measured across test suites</p>
            </div>
          </div>
          <span className="self-start sm:self-auto text-xs font-mono font-semibold px-2.5 py-1 rounded-full bg-cream-200 border border-cream-300 text-ink-700">
            Empirical Benchmarks
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs md:text-sm">
            <thead>
              <tr className="border-b-2 border-cream-300 text-ink-950 font-mono uppercase text-xs">
                <th className="py-2.5 px-3">Pipeline Component</th>
                <th className="py-2.5 px-3">Measured Latency</th>
                <th className="py-2.5 px-3">Operational Characteristics</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-cream-200">
              <tr>
                <td className="py-3 px-3 font-semibold text-ink-900">Multi-Query Web Search</td>
                <td className="py-3 px-3 font-mono text-ink-950">0.8s – 1.4s</td>
                <td className="py-3 px-3 text-ink-600">DuckDuckGo & Google News querying with candidate aggregation</td>
              </tr>
              <tr>
                <td className="py-3 px-3 font-semibold text-ink-900">SSRF-Safe Article Extraction</td>
                <td className="py-3 px-3 font-mono text-ink-950">1.0s – 1.8s</td>
                <td className="py-3 px-3 text-ink-600">Cheerio DOM parser with 8s per-article timeout and snippet fallback</td>
              </tr>
              <tr>
                <td className="py-3 px-3 font-semibold text-ink-900">Model A Relevance (50 Passages)</td>
                <td className="py-3 px-3 font-mono text-ink-950">0.4s – 0.8s</td>
                <td className="py-3 px-3 text-ink-600">RoBERTa-base inference on RTX 2050 GPU (150 samples/sec)</td>
              </tr>
              <tr>
                <td className="py-3 px-3 font-semibold text-ink-900">Model B Entailment Verification</td>
                <td className="py-3 px-3 font-mono text-ink-950">0.5s – 1.0s</td>
                <td className="py-3 px-3 text-ink-600">SciFact transformer inference on RTX 2050 GPU with softmax distribution</td>
              </tr>
              <tr>
                <td className="py-3 px-3 font-semibold text-ink-900">LLM Factual Synthesis</td>
                <td className="py-3 px-3 font-mono text-ink-950">0.8s – 1.5s</td>
                <td className="py-3 px-3 text-ink-600">Structured JSON generation (0.00s if unconfigured or fallback active)</td>
              </tr>
              <tr className="bg-cream-100/70 font-semibold">
                <td className="py-3 px-3 text-ink-950">Total End-to-End Latency</td>
                <td className="py-3 px-3 font-mono text-truth-green-dark font-bold">2.5s – 4.0s</td>
                <td className="py-3 px-3 text-ink-900">Full lifecycle from raw claim input to stored research report</td>
              </tr>
              <tr className="bg-cream-100/70 font-semibold">
                <td className="py-3 px-3 text-ink-950">Cached Query Latency</td>
                <td className="py-3 px-3 font-mono text-truth-green-dark font-bold">&lt; 15 ms</td>
                <td className="py-3 px-3 text-ink-900">Sub-15ms retrieval from MongoDB 2-hour TTL cache</td>
              </tr>
            </tbody>
          </table>
        </div>

        {/* 3 Summary Badges */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-2">
          <div className="p-4 rounded-lg bg-cream-100/50 border border-cream-300">
            <div className="text-xs font-mono text-ink-500 uppercase">Automated Test Suite</div>
            <div className="text-xl md:text-2xl font-bold font-serif text-truth-green-dark mt-1">39 / 39 PASSED</div>
            <div className="text-xs text-ink-600 mt-1">100% test pass rate across regression suite</div>
          </div>
          <div className="p-4 rounded-lg bg-cream-100/50 border border-cream-300">
            <div className="text-xs font-mono text-ink-500 uppercase">Model B Validation Accuracy</div>
            <div className="text-xl md:text-2xl font-bold font-serif text-ink-950 mt-1">78.57%</div>
            <div className="text-xs text-ink-600 mt-1">Macro F1: 79.00% | Weighted F1: 78.31%</div>
          </div>
          <div className="p-4 rounded-lg bg-cream-100/50 border border-cream-300">
            <div className="text-xs font-mono text-ink-500 uppercase">Model A Test Accuracy</div>
            <div className="text-xl md:text-2xl font-bold font-serif text-ink-950 mt-1">90.35%</div>
            <div className="text-xs text-ink-600 mt-1">Macro F1: 78.41% | ROC-AUC: 0.9032</div>
          </div>
        </div>
      </div>

      {/* Security Highlights */}
      <div className="editorial-card p-6 md:p-8 space-y-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-cream-200 pb-4">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-cream-200 text-ink-800">
              <Shield size={18} />
            </div>
            <div>
              <h2 className="font-serif text-lg font-bold text-ink-950">Security Protections & Hardening</h2>
              <p className="text-xs text-ink-500 font-sans">Multi-layered perimeter defense and prompt isolation standards</p>
            </div>
          </div>
          <span className="self-start sm:self-auto text-xs font-mono font-semibold px-2.5 py-1 rounded-full bg-truth-green-bg text-truth-green-dark border border-truth-green-border">
            Phase 10 Hardened
          </span>
        </div>

        <p className="text-xs md:text-sm text-ink-600 leading-relaxed">
          TruthLens AI enforces multi-layered perimeter defenses, input validation boundaries, and prompt isolation to protect system availability, backend resources, and data integrity:
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          <div className="p-4 rounded-lg bg-cream-100/50 border border-cream-300 space-y-2">
            <div className="flex items-center gap-2 text-ink-950 font-semibold text-sm">
              <Shield size={16} className="text-ink-700" />
              <span>HTTP Perimeter & Headers</span>
            </div>
            <p className="text-xs text-ink-600 leading-relaxed">
              Helmet security headers (HSTS, X-Content-Type-Options, Frameguard), configurable CORS origin whitelisting, and 60-second request timeout enforcement.
            </p>
          </div>

          <div className="p-4 rounded-lg bg-cream-100/50 border border-cream-300 space-y-2">
            <div className="flex items-center gap-2 text-ink-950 font-semibold text-sm">
              <Activity size={16} className="text-truth-green-dark" />
              <span>Rate Limiting & Payloads</span>
            </div>
            <p className="text-xs text-ink-600 leading-relaxed">
              Research endpoint throttled to 30 req / 15 min / IP; general APIs at 120 req / 15 min / IP. Body payload clamped strictly at 1MB to prevent buffer exhaustion.
            </p>
          </div>

          <div className="p-4 rounded-lg bg-cream-100/50 border border-cream-300 space-y-2">
            <div className="flex items-center gap-2 text-ink-950 font-semibold text-sm">
              <Globe size={16} className="text-ink-700" />
              <span>SSRF Ingress Filter</span>
            </div>
            <p className="text-xs text-ink-600 leading-relaxed">
              17 attack vectors blocked: loopbacks, private IPv4 CIDRs, cloud metadata endpoints (169.254.169.254, GCP metadata), decimal/hex IPs, and non-HTML binaries.
            </p>
          </div>

          <div className="p-4 rounded-lg bg-cream-100/50 border border-cream-300 space-y-2">
            <div className="flex items-center gap-2 text-ink-950 font-semibold text-sm">
              <Database size={16} className="text-ink-700" />
              <span>Database & Input Defense</span>
            </div>
            <p className="text-xs text-ink-600 leading-relaxed">
              Strict 400 Bad Request validation schemas, 24-character hexadecimal MongoDB ObjectId validation, pagination bounds, and compound indexing.
            </p>
          </div>

          <div className="p-4 rounded-lg bg-cream-100/50 border border-cream-300 space-y-2">
            <div className="flex items-center gap-2 text-ink-950 font-semibold text-sm">
              <Bot size={16} className="text-ink-700" />
              <span>LLM Boundary Isolation</span>
            </div>
            <p className="text-xs text-ink-600 leading-relaxed">
              Retrieved web content tagged with &lt;UNTRUSTED_EVIDENCE&gt; boundary to prevent prompt injection. Model B verdict supremacy strictly enforced.
            </p>
          </div>

          <div className="p-4 rounded-lg bg-cream-100/50 border border-cream-300 space-y-2">
            <div className="flex items-center gap-2 text-ink-950 font-semibold text-sm">
              <Lock size={16} className="text-truth-green-dark" />
              <span>Client-Side Hygiene</span>
            </div>
            <p className="text-xs text-ink-600 leading-relaxed">
              0 dangerouslySetInnerHTML usages across entire frontend, safe external links with rel="noopener noreferrer", and top-level React ErrorBoundary.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
