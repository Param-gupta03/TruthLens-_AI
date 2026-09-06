/**
 * Builds a structured, comprehensive fact-checking & research report
 * Combines claim, ML model results, retrieved sources, evidence-by-source, and LLM explanation (Step 10)
 */

const groupEvidenceBySource = (evidenceList = []) => {
  const groups = new Map();

  for (const item of evidenceList) {
    const srcKey = item.source?.url || item.source?.title || 'Unknown Source';

    if (!groups.has(srcKey)) {
      groups.set(srcKey, {
        sourceName: item.source?.name || 'Web',
        title: item.source?.title || 'Untitled',
        url: item.source?.url || null,
        publishedAt: item.source?.publishedAt || null,
        passages: []
      });
    }

    groups.get(srcKey).passages.push({
      text: item.text,
      relevanceScore: item.relevanceScore,
      relevanceLabel: item.relevanceLabel,
      verification: item.verification || null
    });
  }

  return Array.from(groups.values());
};

const buildResearchReport = (data) => {
  if (!data) return null;

  const explanation = data.explanation || {};
  const evidenceList = data.evidence || [];
  const sources = data.sources || [];
  const retrieval = data.retrieval || {};
  const probabilities = data.probabilities || {};

  return {
    reportId: data.id || (data._id ? data._id.toString() : null),
    claim: data.claim,
    verdict: data.verdict,
    confidence: data.confidence,
    probabilityBreakdown: {
      SUPPORTS: probabilities.SUPPORTS || 0,
      REFUTES: probabilities.REFUTES || 0,
      NOT_ENOUGH_INFO: probabilities.NOT_ENOUGH_INFO || 0
    },
    shortExplanation: explanation.summary || null,
    verdictExplanation: explanation.verdictExplanation || null,
    keyEvidence: explanation.keyEvidence || [],
    contradictoryEvidence: explanation.contradictoryEvidence || [],
    uncertainty: explanation.uncertainty || null,
    sources: sources,
    evidenceBySource: groupEvidenceBySource(evidenceList),
    retrievalStatistics: {
      articlesFound: retrieval.articlesFound || 0,
      articlesProcessed: retrieval.articlesProcessed || 0,
      passagesCreated: retrieval.passagesCreated || 0,
      relevantPassages: retrieval.relevantPassages || 0,
      provider: retrieval.provider || 'unknown',
      timings: retrieval.timings || {}
    },
    llm: data.llm || { status: explanation.summary ? 'generated' : 'unavailable' },
    timestamp: data.createdAt || new Date().toISOString()
  };
};

module.exports = {
  buildResearchReport,
  groupEvidenceBySource
};
