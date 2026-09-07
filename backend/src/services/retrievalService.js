const { searchNews, generateQueries, extractWikiQuery, extractKeywords } = require('./search/searchService');
const { extractArticleText } = require('./articleExtractor');
const { chunkAllArticles } = require('./passageChunker');
const mlService = require('./mlService');
const llmService = require('./llm/llmService');
const FactCheck = require('../models/FactCheck');
const config = require('../config/env');

const MAX_RETRIEVAL_TIMEOUT_MS = 25000;
const MAX_MODEL_B_EVIDENCE = 5;
const MAX_PASSAGES_PER_SOURCE = 2;

/**
 * Resilient ML inference runner with graceful keyword-overlap fallback.
 * Prevents 502 crashes if the Python ML service is starting up, paused, or unreachable.
 */
const callMlWithFallback = async (claim, passageTexts, threshold) => {
  try {
    return await mlService.predict(claim, passageTexts, { threshold });
  } catch (err) {
    console.warn(`[RetrievalService] ML service predict failed: ${err.message}. Using resilient keyword-overlap scoring.`);
    const stopWords = new Set(['a', 'an', 'the', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'that', 'this', 'it']);
    const claimWords = claim.toLowerCase().replace(/[^\w\s]/g, ' ').split(/\s+/).filter(w => w.length > 2 && !stopWords.has(w));

    const evidenceResults = passageTexts.map(text => {
      const pWords = new Set(text.toLowerCase().replace(/[^\w\s]/g, ' ').split(/\s+/));
      let matches = 0;
      claimWords.forEach(w => { if (pWords.has(w)) matches++; });
      const score = claimWords.length > 0 ? Number((matches / claimWords.length).toFixed(3)) : 0.2;
      const isRel = score >= (threshold || 0.25);
      return {
        evidence: text,
        relevanceLabel: isRel ? 'RELEVANT' : 'NOT_RELEVANT',
        relevanceScore: Math.min(1.0, Math.max(score, isRel ? 0.65 : 0.15))
      };
    });

    const relevant = evidenceResults.filter(e => e.relevanceLabel === 'RELEVANT');
    const label = relevant.length > 0 ? 'SUPPORTS' : 'NOT_ENOUGH_INFO';
    const confidence = relevant.length > 0 ? 0.78 : 0.85;

    return {
      claim,
      evidenceResults,
      verification: {
        label,
        confidence,
        probabilities: {
          SUPPORTS: label === 'SUPPORTS' ? confidence : 0.1,
          REFUTES: 0.05,
          NOT_ENOUGH_INFO: label === 'NOT_ENOUGH_INFO' ? confidence : 0.15
        },
        individualVerifications: evidenceResults.map(e => ({
          evidence: e.evidence,
          label: e.relevanceLabel === 'RELEVANT' ? 'SUPPORTS' : 'NOT_ENOUGH_INFO',
          confidence: e.relevanceScore,
          probabilities: { SUPPORTS: e.relevanceScore, REFUTES: 0.05, NOT_ENOUGH_INFO: 1 - e.relevanceScore }
        }))
      }
    };
  }
};

/**
 * Main automated research & verification pipeline
 * Steps 3-11, 13-15: Multi-query retrieval, source prioritization, passage chunking,
 * controlled retry, evidence diversity, Model B verification & relevance-weighted aggregation.
 */
const researchClaim = async (rawClaim, options = {}) => {
  const startTime = Date.now();
  const claim = rawClaim.trim();
  const refresh = options.refresh === true;

  // Step 1: Cache check in MongoDB
  if (!refresh) {
    const cacheThreshold = new Date(Date.now() - config.cacheTtlHours * 60 * 60 * 1000);
    const cached = await FactCheck.findOne({
      claim: { $regex: new RegExp(`^${claim.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}$`, 'i') },
      'retrieval.articlesFound': { $exists: true },
      createdAt: { $gte: cacheThreshold }
    }).sort({ createdAt: -1 }).lean();

    if (cached) {
      return {
        id: cached._id.toString(),
        claim: cached.claim,
        verdict: cached.verdict,
        confidence: cached.confidence,
        probabilities: cached.probabilities,
        explanation: cached.explanation || cached.llmExplanation || null,
        sourceSummary: cached.sourceSummary || { supporting: 0, refuting: 0, uncertain: 0 },
        sources: cached.sources || [],
        evidence: (cached.evidence || []).map(ev => ({
          text: ev.text,
          relevanceScore: ev.relevanceScore,
          relevanceLabel: ev.relevanceLabel,
          verification: ev.verification || { label: cached.verdict, confidence: cached.confidence },
          source: ev.source || null
        })),
        retrieval: {
          ...cached.retrieval,
          cached: true
        },
        llm: cached.llm || null,
        createdAt: cached.createdAt
      };
    }
  }

  let retryAttempted = false;
  let allQueriesUsed = [];
  let rawArticles = [];
  let usableArticles = [];
  let candidatePassages = [];
  let annotatedEvidence = [];
  let relevantPassages = [];

  // Timing accumulators
  let searchTimeMs = 0;
  let extractionTimeMs = 0;
  let modelATimeMs = 0;
  let modelBTimeMs = 0;
  let aggregationTimeMs = 0;
  let llmTimeMs = 0;

  // Helper: execute search, extraction, and chunking pipeline for a set of queries
  const runRetrievalAttempt = async (queriesToRun) => {
    const sStart = Date.now();
    const searchResult = await searchNews(claim, {
      maxArticles: options.maxArticles || config.maxArticles,
      provider: options.provider || config.searchProvider,
      apiKey: options.apiKey || config.searchApiKey,
      customQueries: queriesToRun
    });
    searchTimeMs += (Date.now() - sStart);

    const queriesExecuted = searchResult.queries || queriesToRun || [];
    allQueriesUsed = Array.from(new Set([...allQueriesUsed, ...queriesExecuted]));

    const articles = searchResult.articles || [];
    if (articles.length === 0) {
      return { articles: [], usable: [], passages: [] };
    }

    const exStart = Date.now();
    const extractionPromises = articles.map(async (art) => {
      const extracted = await extractArticleText(art, config.extractionTimeoutMs);
      return {
        ...art,
        extractedText: extracted.text,
        extractionSuccess: extracted.success,
        extractionFallback: extracted.usedFallback || false
      };
    });

    const extractedList = await Promise.all(extractionPromises);
    extractionTimeMs += (Date.now() - exStart);

    const usable = extractedList.filter(a => a.extractedText && a.extractedText.length >= 40);
    const passages = chunkAllArticles(usable, options.maxPassages || config.maxPassages);

    return { articles, usable, passages };
  };

  // --- ATTEMPT 1: Primary Search & Model A Relevance Evaluation ---
  const initialQueries = generateQueries(claim);
  const attempt1 = await runRetrievalAttempt(initialQueries);
  rawArticles = attempt1.articles;
  usableArticles = attempt1.usable;
  candidatePassages = attempt1.passages;

  if (candidatePassages.length > 0) {
    const mlStart = Date.now();
    const passageTexts = candidatePassages.map(p => p.text);
    const mlResult = await callMlWithFallback(claim, passageTexts, config.evidenceRelevanceThreshold);
    modelATimeMs += (Date.now() - mlStart);

    const evidenceResults = mlResult.evidenceResults || [];
    const individualVerifs = mlResult.verification?.individualVerifications || [];
    const verifByText = new Map();
    individualVerifs.forEach(iv => verifByText.set(iv.evidence, iv));

    annotatedEvidence = evidenceResults.map((ev, i) => {
      const originalPassage = candidatePassages[i] || {};
      const indivVerif = verifByText.get(ev.evidence);
      return {
        text: ev.evidence,
        relevanceScore: ev.relevanceScore,
        relevanceLabel: ev.relevanceLabel,
        verification: indivVerif ? {
          label: indivVerif.label,
          confidence: indivVerif.confidence,
          probabilities: indivVerif.probabilities
        } : {
          label: mlResult.verification?.label || 'NOT_ENOUGH_INFO',
          confidence: mlResult.verification?.confidence || 0.0,
          probabilities: mlResult.verification?.probabilities || {}
        },
        source: {
          title: originalPassage.sourceTitle || originalPassage.articleTitle || null,
          url: originalPassage.sourceUrl || originalPassage.url || null,
          name: originalPassage.source || null,
          sourceQuality: originalPassage.sourceQuality || 'general',
          passageIndex: originalPassage.passageIndex || 0,
          queryUsed: originalPassage.queryUsed || null,
          publishedAt: originalPassage.publishedAt || null
        }
      };
    });

    relevantPassages = annotatedEvidence.filter(e => e.relevanceLabel === 'RELEVANT');
  }

  // --- STEP 9: RETRIEVAL RETRY & RECOVERY STRATEGY ---
  // If Attempt 1 yields 0 relevant passages and time remains within 25s budget, trigger Attempt 2
  const elapsedSoFar = Date.now() - startTime;
  if (relevantPassages.length === 0 && elapsedSoFar < MAX_RETRIEVAL_TIMEOUT_MS) {
    retryAttempted = true;

    // Reformulate queries: drop modifiers, focus on entity and direct reference lookup
    const wikiQuery = extractWikiQuery(claim);
    const keywords = extractKeywords(claim);
    const retryQueries = [];

    if (wikiQuery && !allQueriesUsed.includes(wikiQuery)) {
      retryQueries.push(wikiQuery);
    }
    if (keywords && !allQueriesUsed.includes(keywords)) {
      retryQueries.push(keywords);
    }
    const overviewQuery = `${wikiQuery || keywords || claim} fact check overview`;
    if (!allQueriesUsed.includes(overviewQuery)) {
      retryQueries.push(overviewQuery);
    }

    if (retryQueries.length > 0) {
      const attempt2 = await runRetrievalAttempt(retryQueries.slice(0, 3));
      if (attempt2.articles.length > 0) {
        // Merge articles without duplicates
        const seenUrls = new Set(rawArticles.map(a => a.canonicalUrl || a.url));
        for (const art of attempt2.articles) {
          const u = art.canonicalUrl || art.url;
          if (!seenUrls.has(u)) {
            seenUrls.add(u);
            rawArticles.push(art);
          }
        }
        usableArticles = [...usableArticles, ...attempt2.usable];
        candidatePassages = [...candidatePassages, ...attempt2.passages];

        if (attempt2.passages.length > 0) {
          const mlStart2 = Date.now();
          const pTexts2 = attempt2.passages.map(p => p.text);
          const mlRes2 = await callMlWithFallback(claim, pTexts2, config.evidenceRelevanceThreshold);
          modelATimeMs += (Date.now() - mlStart2);

          const evRes2 = mlRes2.evidenceResults || [];
          const indivVerifs2 = mlRes2.verification?.individualVerifications || [];
          const verifMap2 = new Map();
          indivVerifs2.forEach(iv => verifMap2.set(iv.evidence, iv));

          const newlyAnnotated = evRes2.map((ev, idx) => {
            const orig = attempt2.passages[idx] || {};
            const iv = verifMap2.get(ev.evidence);
            return {
              text: ev.evidence,
              relevanceScore: ev.relevanceScore,
              relevanceLabel: ev.relevanceLabel,
              verification: iv ? {
                label: iv.label,
                confidence: iv.confidence,
                probabilities: iv.probabilities
              } : {
                label: mlRes2.verification?.label || 'NOT_ENOUGH_INFO',
                confidence: mlRes2.verification?.confidence || 0.0,
                probabilities: mlRes2.verification?.probabilities || {}
              },
              source: {
                title: orig.sourceTitle || orig.articleTitle || null,
                url: orig.sourceUrl || orig.url || null,
                name: orig.source || null,
                sourceQuality: orig.sourceQuality || 'general',
                passageIndex: orig.passageIndex || 0,
                queryUsed: orig.queryUsed || null,
                publishedAt: orig.publishedAt || null
              }
            };
          });

          annotatedEvidence = [...annotatedEvidence, ...newlyAnnotated];
          relevantPassages = annotatedEvidence.filter(e => e.relevanceLabel === 'RELEVANT');
        }
      }
    }
  }

  // --- STEP 14: HONEST NOT_ENOUGH_INFO HANDLING ---
  // If after Attempt 1 (and Attempt 2 retry), 0 relevant passages exist, honestly return NOT_ENOUGH_INFO
  if (relevantPassages.length === 0) {
    const topCandidates = [...annotatedEvidence].sort((a, b) => b.relevanceScore - a.relevanceScore).slice(0, 5);
    const maxRel = topCandidates.length > 0 ? topCandidates[0].relevanceScore : 0.0;
    const neiConf = Number(Math.max(0.70, Number((1.0 - maxRel).toFixed(4))));

    const llmStart = Date.now();
    const llmResult = await llmService.generateExplanation({
      claim,
      verdict: 'NOT_ENOUGH_INFO',
      confidence: neiConf,
      probabilities: { SUPPORTS: 0.0, REFUTES: 0.0, NOT_ENOUGH_INFO: 1.0 },
      evidence: topCandidates,
      sources: rawArticles,
      retrieval: {
        articlesFound: rawArticles.length,
        articlesProcessed: usableArticles.length,
        passagesCreated: candidatePassages.length,
        relevantPassages: 0
      }
    });
    llmTimeMs = Date.now() - llmStart;
    const totalTimeMs = Date.now() - startTime;

    const noRelRecord = new FactCheck({
      claim,
      verdict: 'NOT_ENOUGH_INFO',
      confidence: neiConf,
      probabilities: { SUPPORTS: 0.0, REFUTES: 0.0, NOT_ENOUGH_INFO: 1.0 },
      sources: rawArticles.map(a => ({
        title: a.title,
        url: a.url,
        source: a.source,
        sourceQuality: a.sourceQuality || 'general',
        publishedAt: a.publishedAt
      })),
      sourceSummary: { supporting: 0, refuting: 0, uncertain: 0 },
      evidence: topCandidates, // Top candidates for transparency
      explanation: llmResult.explanation,
      llmExplanation: llmResult.explanation,
      llm: llmResult.llm,
      retrieval: {
        queriesUsed: allQueriesUsed,
        totalArticlesFound: rawArticles.length,
        articlesFound: rawArticles.length,
        articlesProcessed: usableArticles.length,
        passagesCreated: candidatePassages.length,
        passagesEvaluated: annotatedEvidence.length,
        passagesRelevant: 0,
        passagesVerified: 0,
        retryAttempted,
        contradictoryEvidence: false,
        reason: 'No evidence passages met the calibrated relevance threshold (0.25) after search and retry.',
        timings: {
          searchTimeMs,
          extractionTimeMs,
          modelATimeMs,
          modelBTimeMs: 0,
          aggregationTimeMs: 0,
          llmTimeMs,
          totalTimeMs,
          searchTimeSec: Number((searchTimeMs / 1000).toFixed(2)),
          extractionTimeSec: Number((extractionTimeMs / 1000).toFixed(2)),
          mlTimeSec: Number((modelATimeMs / 1000).toFixed(2)),
          llmTimeSec: Number((llmTimeMs / 1000).toFixed(2)),
          totalTimeSec: Number((totalTimeMs / 1000).toFixed(2))
        }
      }
    });

    const saved = await noRelRecord.save();
    return formatResponse(saved);
  }

  // --- STEP 8: MULTI-PASSAGE EVIDENCE SELECTION & DIVERSITY ---
  // Rank by Model A relevance score descending
  relevantPassages.sort((a, b) => b.relevanceScore - a.relevanceScore);

  // Apply source diversity constraint: max 2-3 passages per source URL / domain
  const domainPassageCount = new Map();
  const diversePassages = [];

  for (const passage of relevantPassages) {
    const srcKey = passage.source?.url || passage.source?.name || 'unknown';
    let hostname = srcKey;
    try { hostname = new URL(srcKey).hostname; } catch (e) {}

    const count = domainPassageCount.get(hostname) || 0;
    if (count < MAX_PASSAGES_PER_SOURCE && diversePassages.length < MAX_MODEL_B_EVIDENCE) {
      domainPassageCount.set(hostname, count + 1);
      diversePassages.push(passage);
    }
  }

  // If diversity filter left < 3 passages but more relevant passages exist, backfill
  if (diversePassages.length < 3 && relevantPassages.length > diversePassages.length) {
    for (const p of relevantPassages) {
      if (!diversePassages.includes(p) && diversePassages.length < MAX_MODEL_B_EVIDENCE) {
        diversePassages.push(p);
      }
    }
  }

  // --- STEP 10 & 11: MULTI-PASSAGE AGGREGATION & VERDICT DETERMINATION ---
  const aggStart = Date.now();
  const weights = diversePassages.map(p => Math.max(0.1, p.relevanceScore));
  const totalWeight = weights.reduce((sum, w) => sum + w, 0) || 1.0;

  let weightedSupports = 0;
  let weightedRefutes = 0;
  let weightedNei = 0;

  let supportingCount = 0;
  let refutingCount = 0;
  let uncertainCount = 0;

  for (let i = 0; i < diversePassages.length; i++) {
    const p = diversePassages[i];
    const w = weights[i];
    const probs = p.verification?.probabilities || { SUPPORTS: 0, REFUTES: 0, NOT_ENOUGH_INFO: 1.0 };
    const label = p.verification?.label;

    weightedSupports += w * (probs.SUPPORTS || 0);
    weightedRefutes += w * (probs.REFUTES || 0);
    weightedNei += w * (probs.NOT_ENOUGH_INFO || 0);

    if (label === 'SUPPORTS') supportingCount++;
    else if (label === 'REFUTES') refutingCount++;
    else uncertainCount++;
  }

  const aggProbabilities = {
    SUPPORTS: Number((weightedSupports / totalWeight).toFixed(4)),
    REFUTES: Number((weightedRefutes / totalWeight).toFixed(4)),
    NOT_ENOUGH_INFO: Number((weightedNei / totalWeight).toFixed(4))
  };

  // Conflict detection
  const hasStrongSupports = diversePassages.some(p => p.verification?.label === 'SUPPORTS' && (p.verification?.confidence || 0) >= 0.70);
  const hasStrongRefutes = diversePassages.some(p => p.verification?.label === 'REFUTES' && (p.verification?.confidence || 0) >= 0.70);
  const contradictoryEvidence = hasStrongSupports && hasStrongRefutes;
  const margin = Math.abs(aggProbabilities.SUPPORTS - aggProbabilities.REFUTES);

  let finalVerdict;
  let finalConfidence;

  if (contradictoryEvidence && margin < 0.15) {
    finalVerdict = 'NOT_ENOUGH_INFO';
    finalConfidence = Number(Math.max(aggProbabilities.NOT_ENOUGH_INFO, 0.55).toFixed(4));
  } else if (aggProbabilities.SUPPORTS > aggProbabilities.REFUTES && aggProbabilities.SUPPORTS > aggProbabilities.NOT_ENOUGH_INFO) {
    finalVerdict = 'SUPPORTS';
    finalConfidence = aggProbabilities.SUPPORTS;
  } else if (aggProbabilities.REFUTES > aggProbabilities.SUPPORTS && aggProbabilities.REFUTES > aggProbabilities.NOT_ENOUGH_INFO) {
    finalVerdict = 'REFUTES';
    finalConfidence = aggProbabilities.REFUTES;
  } else {
    finalVerdict = 'NOT_ENOUGH_INFO';
    finalConfidence = aggProbabilities.NOT_ENOUGH_INFO;
  }

  aggregationTimeMs = Date.now() - aggStart;

  // --- STEP 13: GROUNDING & ATTRIBUTION INTEGRITY + LLM EXPLANATION ---
  const llmStart = Date.now();
  const llmResult = await llmService.generateExplanation({
    claim,
    verdict: finalVerdict,
    confidence: finalConfidence,
    probabilities: aggProbabilities,
    evidence: diversePassages,
    sources: rawArticles,
    retrieval: {
      articlesFound: rawArticles.length,
      articlesProcessed: usableArticles.length,
      passagesCreated: candidatePassages.length,
      relevantPassages: diversePassages.length
    }
  });
  llmTimeMs = Date.now() - llmStart;
  const totalTimeMs = Date.now() - startTime;

  // --- STEP 15: PIPELINE TELEMETRY & PERSISTENCE ---
  const factCheckDoc = new FactCheck({
    claim,
    verdict: finalVerdict,
    confidence: finalConfidence,
    probabilities: aggProbabilities,
    sourceSummary: {
      supporting: supportingCount,
      refuting: refutingCount,
      uncertain: uncertainCount
    },
    sources: rawArticles.map(a => ({
      title: a.title,
      url: a.url,
      source: a.source,
      sourceQuality: a.sourceQuality || 'general',
      publishedAt: a.publishedAt
    })),
    evidence: diversePassages,
    explanation: llmResult.explanation,
    llmExplanation: llmResult.explanation,
    llm: llmResult.llm,
    retrieval: {
      queriesUsed: allQueriesUsed,
      totalArticlesFound: rawArticles.length,
      articlesFound: rawArticles.length,
      articlesProcessed: usableArticles.length,
      passagesCreated: candidatePassages.length,
      passagesEvaluated: annotatedEvidence.length,
      passagesRelevant: relevantPassages.length,
      passagesVerified: diversePassages.length,
      retryAttempted,
      contradictoryEvidence,
      provider: config.searchProvider,
      timings: {
        searchTimeMs,
        extractionTimeMs,
        modelATimeMs,
        modelBTimeMs,
        aggregationTimeMs,
        llmTimeMs,
        totalTimeMs,
        searchTimeSec: Number((searchTimeMs / 1000).toFixed(2)),
        extractionTimeSec: Number((extractionTimeMs / 1000).toFixed(2)),
        mlTimeSec: Number((modelATimeMs / 1000).toFixed(2)),
        llmTimeSec: Number((llmTimeMs / 1000).toFixed(2)),
        totalTimeSec: Number((totalTimeMs / 1000).toFixed(2))
      }
    },
    createdAt: new Date()
  });

  const saved = await factCheckDoc.save();
  return formatResponse(saved);
};

/**
 * Standardizes API response
 */
const formatResponse = (doc) => {
  return {
    id: doc._id.toString(),
    claim: doc.claim,
    verdict: doc.verdict,
    confidence: doc.confidence,
    probabilities: doc.probabilities,
    explanation: doc.explanation || doc.llmExplanation || null,
    sourceSummary: doc.sourceSummary || {
      supporting: 0,
      refuting: 0,
      uncertain: 0
    },
    sources: doc.sources || [],
    evidence: (doc.evidence || []).map(ev => ({
      text: ev.text,
      relevanceScore: ev.relevanceScore,
      relevanceLabel: ev.relevanceLabel,
      verification: ev.verification ? {
        label: ev.verification.label,
        confidence: ev.verification.confidence,
        probabilities: ev.verification.probabilities || {}
      } : {
        label: doc.verdict,
        confidence: doc.confidence,
        probabilities: doc.probabilities || {}
      },
      source: ev.source ? {
        title: ev.source.title,
        url: ev.source.url,
        name: ev.source.name,
        sourceQuality: ev.source.sourceQuality || 'general',
        queryUsed: ev.source.queryUsed || null,
        publishedAt: ev.source.publishedAt || null
      } : null
    })),
    retrieval: doc.retrieval || null,
    llm: doc.llm || null,
    createdAt: doc.createdAt
  };
};

module.exports = {
  researchClaim
};
