const mongoose = require('mongoose');

const EvidenceItemSchema = new mongoose.Schema(
  {
    text: {
      type: String,
      required: true
    },
    relevanceScore: {
      type: Number,
      required: true
    },
    relevanceLabel: {
      type: String,
      required: true
    },
    verification: {
      label: { type: String, default: null },
      confidence: { type: Number, default: 0 },
      probabilities: { type: Object, default: {} }
    },
    source: {
      title: { type: String, default: null },
      url: { type: String, default: null },
      name: { type: String, default: null },
      publishedAt: { type: Date, default: null }
    }
  },
  { _id: false }
);

const ProbabilitiesSchema = new mongoose.Schema(
  {
    SUPPORTS: { type: Number, required: true, default: 0 },
    REFUTES: { type: Number, required: true, default: 0 },
    NOT_ENOUGH_INFO: { type: Number, required: true, default: 0 }
  },
  { _id: false }
);

const FactCheckSchema = new mongoose.Schema(
  {
    claim: {
      type: String,
      required: true,
      trim: true,
      index: true
    },
    verdict: {
      type: String,
      required: true,
      enum: ['SUPPORTS', 'REFUTES', 'NOT_ENOUGH_INFO']
    },
    confidence: {
      type: Number,
      required: true
    },
    probabilities: {
      type: ProbabilitiesSchema,
      required: true
    },
    evidence: [EvidenceItemSchema],
    sources: {
      type: Array,
      default: []
    },
    sourceSummary: {
      supporting: { type: Number, default: 0 },
      refuting: { type: Number, default: 0 },
      uncertain: { type: Number, default: 0 }
    },
    retrieval: {
      type: Object,
      default: null
    },
    explanation: {
      type: Object,
      default: null
    },
    llmExplanation: {
      type: Object,
      default: null
    },
    llm: {
      type: Object,
      default: null
    },
    createdAt: {
      type: Date,
      default: Date.now,
      index: true
    }
  },
  {
    timestamps: true,
    strict: false // keeps schema flexible for future news/source/citations/LLM explanation
  }
);

// Compound index for verdict filtering and date sorting (Phase 10 Hardening)
FactCheckSchema.index({ verdict: 1, createdAt: -1 });

module.exports = mongoose.model('FactCheck', FactCheckSchema);
