/**
 * TruthLens AI — Centralized Frontend API Service
 * Manages all HTTP communication with the Express backend (:5000)
 */

const API_BASE_URL = (typeof import.meta !== 'undefined' && import.meta.env && import.meta.env.VITE_API_URL) || 'http://localhost:5000/api';

/**
 * Normalizes backend responses into a consistent, robust schema
 */
export const normalizeResearchResult = (raw) => {
  if (!raw) return null;
  const data = raw.data || raw;

  const probabilities = data.probabilities || {};
  const evidenceList = (data.evidence || []).map(ev => ({
    text: ev.text || '',
    relevanceScore: typeof ev.relevanceScore === 'number' ? ev.relevanceScore : 0,
    relevanceLabel: ev.relevanceLabel || 'RELEVANT',
    verification: ev.verification || {
      label: data.verdict || 'NOT_ENOUGH_INFO',
      confidence: data.confidence || 0
    },
    source: ev.source || {
      title: 'Web Source',
      url: null,
      name: 'Web'
    }
  }));

  // Separate evidence into categorized groups
  const supporting = [];
  const refuting = [];
  const uncertain = [];

  for (const item of evidenceList) {
    const label = item.verification?.label;
    if (label === 'SUPPORTS') supporting.push(item);
    else if (label === 'REFUTES') refuting.push(item);
    else uncertain.push(item);
  }

  // Explanation normalization
  const expl = data.explanation || data.llmExplanation || {};
  const hasExplanation = Boolean(expl && expl.summary);

  return {
    id: data.id || data._id || data.reportId || '',
    claim: data.claim || '',
    verdict: data.verdict || 'NOT_ENOUGH_INFO',
    confidence: typeof data.confidence === 'number' ? data.confidence : 0,
    confidencePercent: ((data.confidence || 0) * 100).toFixed(1),
    probabilities: {
      SUPPORTS: probabilities.SUPPORTS || 0,
      REFUTES: probabilities.REFUTES || 0,
      NOT_ENOUGH_INFO: probabilities.NOT_ENOUGH_INFO || 0,
      supportsPercent: ((probabilities.SUPPORTS || 0) * 100).toFixed(1),
      refutesPercent: ((probabilities.REFUTES || 0) * 100).toFixed(1),
      neutralPercent: ((probabilities.NOT_ENOUGH_INFO || 0) * 100).toFixed(1)
    },
    sourceSummary: data.sourceSummary || {
      supporting: supporting.length,
      refuting: refuting.length,
      uncertain: uncertain.length
    },
    sources: data.sources || [],
    evidence: evidenceList,
    categorizedEvidence: {
      supporting,
      refuting,
      uncertain
    },
    evidenceBySource: data.evidenceBySource || [],
    explanation: hasExplanation ? {
      summary: expl.summary || '',
      verdictExplanation: expl.verdictExplanation || '',
      keyEvidence: expl.keyEvidence || [],
      contradictoryEvidence: expl.contradictoryEvidence || [],
      uncertainty: expl.uncertainty || '',
      sourceNotes: expl.sourceNotes || []
    } : null,
    llm: data.llm || {
      status: hasExplanation ? 'generated' : 'unavailable',
      reason: hasExplanation ? null : 'LLM explanation not provided'
    },
    retrieval: data.retrieval || data.retrievalStatistics || {
      articlesFound: (data.sources || []).length,
      articlesProcessed: (data.sources || []).length,
      passagesCreated: evidenceList.length,
      relevantPassages: evidenceList.length,
      timings: {}
    },
    createdAt: data.createdAt || data.timestamp || new Date().toISOString()
  };
};

/**
 * Initiates automated news & web research for a claim
 */
export const researchClaim = async (claim, refresh = false) => {
  const response = await fetch(`${API_BASE_URL}/research`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ claim, refresh })
  });

  const json = await response.json();
  if (!response.ok || !json.success) {
    throw new Error(json.message || 'Failed to complete research for this claim.');
  }

  return normalizeResearchResult(json);
};

/**
 * Retrieves a detailed, complete research report by record ID
 */
export const getResearchReport = async (id) => {
  const response = await fetch(`${API_BASE_URL}/research/${id}/report`);
  const json = await response.json();

  if (!response.ok || !json.success) {
    throw new Error(json.message || 'Failed to retrieve research report.');
  }

  return normalizeResearchResult(json);
};

/**
 * Retrieves historical fact-checks with pagination
 */
export const getFactChecks = async (page = 1, limit = 10) => {
  const response = await fetch(`${API_BASE_URL}/fact-checks?page=${page}&limit=${limit}`);
  const json = await response.json();

  if (!response.ok || !json.success) {
    throw new Error(json.message || 'Failed to load research history.');
  }

  return {
    records: (json.data || []).map(normalizeResearchResult),
    pagination: json.pagination || { total: 0, page: 1, limit, pages: 1 }
  };
};

/**
 * Retrieves a single fact-check record by ID
 */
export const getFactCheckById = async (id) => {
  const response = await fetch(`${API_BASE_URL}/fact-checks/${id}`);
  const json = await response.json();

  if (!response.ok || !json.success) {
    throw new Error(json.message || 'Fact check record not found.');
  }

  return normalizeResearchResult(json);
};

/**
 * Executes a manual fact check with user-provided evidence
 */
export const submitManualFactCheck = async (claim, evidenceList) => {
  const response = await fetch(`${API_BASE_URL}/fact-check`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ claim, evidence: evidenceList })
  });

  const json = await response.json();
  if (!response.ok || !json.success) {
    throw new Error(json.message || 'Manual verification failed.');
  }

  return normalizeResearchResult(json);
};

/**
 * Checks system operational health (Express backend and FastAPI ML service)
 */
export const checkHealth = async () => {
  try {
    const [backendRes, mlRes] = await Promise.all([
      fetch(`${API_BASE_URL}/health`).then(r => r.json()).catch(() => null),
      fetch(`${API_BASE_URL}/health/ml`).then(r => r.json()).catch(() => null)
    ]);

    return {
      backendOnline: backendRes?.status === 'healthy',
      mlOnline: mlRes?.mlService === 'healthy'
    };
  } catch (e) {
    return { backendOnline: false, mlOnline: false };
  }
};
