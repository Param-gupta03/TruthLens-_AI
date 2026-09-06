const mongoose = require('mongoose');
const FactCheck = require('../models/FactCheck');
const mlService = require('./mlService');
const { AppError } = require('../middleware/errorHandler');

/**
 * Executes fact-checking via ML Service and persists record to MongoDB
 */
const createFactCheck = async ({ claim, evidence }) => {
  // Step 1: Call Python ML service (Model A & Model B)
  const mlResult = await mlService.predict(claim, evidence);

  // Step 2: Format evidence items for Mongoose schema
  const formattedEvidence = (mlResult.evidenceResults || []).map(item => ({
    text: item.evidence,
    relevanceScore: item.relevanceScore,
    relevanceLabel: item.relevanceLabel,
    verificationLabel: null
  }));

  // Step 3: Create document and persist in MongoDB
  const factCheckDoc = new FactCheck({
    claim: mlResult.claim || claim,
    verdict: mlResult.verification.label,
    confidence: mlResult.verification.confidence,
    probabilities: {
      SUPPORTS: mlResult.verification.probabilities.SUPPORTS,
      REFUTES: mlResult.verification.probabilities.REFUTES,
      NOT_ENOUGH_INFO: mlResult.verification.probabilities.NOT_ENOUGH_INFO
    },
    evidence: formattedEvidence,
    createdAt: new Date()
  });

  const saved = await factCheckDoc.save();

  // Step 4: Return response object conforming to specification
  return {
    id: saved._id.toString(),
    claim: saved.claim,
    verdict: saved.verdict,
    confidence: saved.confidence,
    probabilities: {
      SUPPORTS: saved.probabilities.SUPPORTS,
      REFUTES: saved.probabilities.REFUTES,
      NOT_ENOUGH_INFO: saved.probabilities.NOT_ENOUGH_INFO
    },
    evidence: saved.evidence.map(ev => ({
      text: ev.text,
      relevanceScore: ev.relevanceScore,
      relevanceLabel: ev.relevanceLabel
    })),
    createdAt: saved.createdAt
  };
};

/**
 * Retrieves paginated fact-check records from MongoDB
 */
const getFactChecks = async ({ page = 1, limit = 10 }) => {
  const pageNum = Math.max(1, parseInt(page, 10) || 1);
  const limitNum = Math.min(50, Math.max(1, parseInt(limit, 10) || 10));
  const skip = (pageNum - 1) * limitNum;

  const [records, total] = await Promise.all([
    FactCheck.find()
      .sort({ createdAt: -1 })
      .skip(skip)
      .limit(limitNum)
      .lean(),
    FactCheck.countDocuments()
  ]);

  const formatted = records.map(rec => ({
    id: rec._id.toString(),
    claim: rec.claim,
    verdict: rec.verdict,
    confidence: rec.confidence,
    probabilities: rec.probabilities,
    sourcesCount: (rec.sources || []).length,
    evidenceCount: (rec.evidence || []).length,
    explanation: rec.explanation || rec.llmExplanation || null,
    llm: rec.llm || null,
    evidence: (rec.evidence || []).map(ev => ({
      text: ev.text,
      relevanceScore: ev.relevanceScore,
      relevanceLabel: ev.relevanceLabel,
      verification: ev.verification || null,
      source: ev.source || null
    })),
    createdAt: rec.createdAt
  }));

  return {
    records: formatted,
    pagination: {
      total,
      page: pageNum,
      limit: limitNum,
      pages: Math.ceil(total / limitNum) || 1
    }
  };
};

/**
 * Retrieves a single fact-check record by ID
 */
const getFactCheckById = async (id) => {
  if (!mongoose.Types.ObjectId.isValid(id)) {
    throw new AppError('Invalid fact check record ID format.', 400);
  }

  const record = await FactCheck.findById(id).lean();
  if (!record) {
    throw new AppError('Fact check record not found.', 404);
  }

  return {
    id: record._id.toString(),
    claim: record.claim,
    verdict: record.verdict,
    confidence: record.confidence,
    probabilities: record.probabilities,
    explanation: record.explanation || record.llmExplanation || null,
    sourceSummary: record.sourceSummary || null,
    sources: record.sources || [],
    evidence: (record.evidence || []).map(ev => ({
      text: ev.text,
      relevanceScore: ev.relevanceScore,
      relevanceLabel: ev.relevanceLabel,
      verification: ev.verification || null,
      source: ev.source || null
    })),
    retrieval: record.retrieval || null,
    llm: record.llm || null,
    createdAt: record.createdAt
  };
};

/**
 * Deletes a fact-check record by ID
 */
const deleteFactCheckById = async (id) => {
  if (!mongoose.Types.ObjectId.isValid(id)) {
    throw new AppError('Invalid fact check record ID format.', 400);
  }

  const deleted = await FactCheck.findByIdAndDelete(id);
  if (!deleted) {
    throw new AppError('Fact check record not found.', 404);
  }

  return true;
};

module.exports = {
  createFactCheck,
  getFactChecks,
  getFactCheckById,
  deleteFactCheckById
};
