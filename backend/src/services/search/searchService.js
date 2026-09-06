const { getSearchProvider, WikipediaProvider } = require('./providers');
const config = require('../../config/env');

const STOP_WORDS = new Set([
  'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
  'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about',
  'into', 'over', 'after', 'beneath', 'under', 'above', 'that', 'this',
  'these', 'those', 'it', 'its', 'as', 'and', 'or', 'but', 'if', 'then',
  'so', 'than', 'too', 'very', 'just', 'now', 'such', 'why', 'what', 'how'
]);

const MODIFIER_WORDS = new Set([
  'approximately', 'approx', 'roughly', 'nearly', 'about', 'estimated',
  'exact', 'exactly', 'typically', 'per', 'second', 'seconds', 'kilometers',
  'degrees', 'degree', 'percent', 'percentage', 'secret', 'named', 'exclusive',
  'exclusively', 'completely', 'totally', 'all', 'entire', 'hours', 'hour',
  'causes', 'cause', 'causing', 'cures', 'cure', 'curing', 'invented', 'invent',
  'created', 'landed', 'lands'
]);

const LOW_QUALITY_DOMAINS = [
  'reddit.com', 'quora.com', 'twitter.com', 'x.com', 'facebook.com',
  'instagram.com', 'pinterest.com', 'tiktok.com', 'buzzfeed.com',
  'dailymail.co.uk', 'dailymail.com', 'thesun.co.uk', 'mirror.co.uk',
  'express.co.uk', 'nypost.com', 'thegatewaypundit.com'
];

/**
 * Extracts essential keywords by removing stop words and punctuation
 */
const extractKeywords = (text) => {
  if (!text) return '';
  const words = text
    .replace(/[.,!?;:"'()\[\]{}]/g, ' ')
    .split(/\s+/)
    .map(w => w.trim())
    .filter(w => w.length > 1 && !STOP_WORDS.has(w.toLowerCase()));
  return words.join(' ');
};

/**
 * Extracts compact entity query for Wikipedia search API
 */
const extractWikiQuery = (claim) => {
  if (!claim) return '';
  const words = claim
    .replace(/[.,!?;:"'()\[\]{}]/g, ' ')
    .split(/\s+/)
    .map(x => x.trim())
    .filter(x => x.length > 1 && !STOP_WORDS.has(x.toLowerCase()) && !MODIFIER_WORDS.has(x.toLowerCase()));
  return words.slice(0, 4).join(' ');
};

/**
 * Normalizes URL into canonical form to ensure strict deduplication
 */
const normalizeCanonicalUrl = (rawUrl) => {
  try {
    const parsed = new URL(rawUrl);
    parsed.protocol = parsed.protocol.toLowerCase();
    parsed.hostname = parsed.hostname.toLowerCase();
    parsed.hash = '';

    const trackingParams = [
      'utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content',
      'fbclid', 'gclid', 'ref', 'source', 'cmp', 'trk'
    ];
    trackingParams.forEach(param => parsed.searchParams.delete(param));

    if (parsed.pathname.length > 1 && parsed.pathname.endsWith('/')) {
      parsed.pathname = parsed.pathname.slice(0, -1);
    }

    return parsed.toString();
  } catch (e) {
    return rawUrl.trim().toLowerCase();
  }
};

/**
 * Determines source authority score (0 to 10) and quality tier
 * Step 4: Search Result Ranking & Source Prioritization
 */
const getSourceQualityTier = (url, sourceName) => {
  if (!url) return { score: 1, tier: 'general' };
  const lowerUrl = url.toLowerCase();
  const lowerSrc = (sourceName || '').toLowerCase();

  for (const bad of LOW_QUALITY_DOMAINS) {
    if (lowerUrl.includes(bad) || lowerSrc.includes(bad)) {
      return { score: 0, tier: 'low_quality' };
    }
  }

  // Tier 1: Academic / Official (.gov, .edu, WHO, CDC, NASA, NIH)
  if (
    lowerUrl.includes('.gov') || lowerUrl.includes('.edu') ||
    lowerUrl.includes('who.int') || lowerUrl.includes('cdc.gov') ||
    lowerUrl.includes('nasa.gov') || lowerUrl.includes('nih.gov') ||
    lowerUrl.includes('un.org')
  ) {
    return { score: 10, tier: 'official' };
  }

  // Tier 2: Scientific journals & medical repositories
  if (
    lowerUrl.includes('nature.com') || lowerUrl.includes('science.org') ||
    lowerUrl.includes('ncbi.nlm.nih.gov') || lowerUrl.includes('pubmed') ||
    lowerUrl.includes('cell.com') || lowerUrl.includes('thelancet.com') ||
    lowerUrl.includes('nejm.org') || lowerUrl.includes('sciencedirect.com')
  ) {
    return { score: 9, tier: 'scientific' };
  }

  // Tier 3: Established encyclopedias & reference
  if (
    lowerUrl.includes('wikipedia.org') || lowerSrc.includes('wikipedia') ||
    lowerUrl.includes('britannica.com') || lowerUrl.includes('merriam-webster.com')
  ) {
    return { score: 9, tier: 'reference' };
  }

  // Tier 4: Major news organizations & fact checkers
  if (
    lowerUrl.includes('reuters.com') || lowerUrl.includes('apnews.com') ||
    lowerUrl.includes('bbc.com') || lowerUrl.includes('bbc.co.uk') ||
    lowerUrl.includes('afp.com') || lowerUrl.includes('snopes.com') ||
    lowerUrl.includes('factcheck.org') || lowerUrl.includes('politifact.com') ||
    lowerUrl.includes('nytimes.com') || lowerUrl.includes('washingtonpost.com') ||
    lowerUrl.includes('theguardian.com') || lowerUrl.includes('npr.org')
  ) {
    return { score: 7, tier: 'news' };
  }

  return { score: 4, tier: 'general' };
};

const getSourceAuthorityScore = (url, sourceName) => {
  return getSourceQualityTier(url, sourceName).score;
};

/**
 * Deterministically generates targeted search queries from user claim
 * Step 3: Multi-query retrieval strategy (Max 3-4 queries)
 */
const generateQueries = (claim) => {
  if (!claim || typeof claim !== 'string') return [];

  const cleaned = claim.trim().replace(/[.,!?;:"'()\[\]{}]/g, ' ').replace(/\s+/g, ' ').trim();
  if (!cleaned) return [];

  const queries = [];

  // Query 1: Exact claim (for direct fact-check hits)
  queries.push(cleaned);

  // Query 2: Core entities + key assertion (stripped of filler)
  const keywords = extractKeywords(cleaned);
  if (keywords && keywords.toLowerCase() !== cleaned.toLowerCase() && keywords.split(' ').length >= 2) {
    queries.push(keywords);
  }

  // Query 3: Factual keyword combination (for encyclopedic / reference sources)
  const wikiQuery = extractWikiQuery(cleaned);
  if (wikiQuery && wikiQuery !== keywords && !queries.includes(wikiQuery)) {
    queries.push(wikiQuery);
  }

  // Query 4: Claim topic + 'scientific evidence' or 'fact check'
  const lower = cleaned.toLowerCase();
  const isMedical = /cancer|vaccin|disease|health|trial|study|medicin|drug|treatment|symptom|virus|infection|smoking/.test(lower);
  const coreTerms = wikiQuery || keywords || cleaned;
  if (isMedical) {
    queries.push(`${coreTerms} scientific evidence`);
  } else {
    queries.push(`${coreTerms} fact check`);
  }

  return Array.from(new Set(queries.map(q => q.trim()))).filter(Boolean).slice(0, 4);
};

/**
 * High-level search orchestration across queries with deduplication & source prioritization
 */
const searchNews = async (claim, options = {}) => {
  const maxArticles = options.maxArticles || config.maxArticles;
  const primaryProvider = getSearchProvider(options.provider, options.apiKey);
  const wikiProvider = new WikipediaProvider(config.searchTimeoutMs);

  const queries = options.customQueries || generateQueries(claim);
  const seenUrls = new Set();
  const collectedArticles = [];

  // Query primary search provider across generated queries concurrently
  const searchPromises = queries.map(async (q) => {
    try {
      const hits = await primaryProvider.search(q, maxArticles);
      return (hits || []).map(h => ({ ...h, queryUsed: q }));
    } catch (e) {
      return [];
    }
  });

  // Query Wikipedia for core entity
  const wikiQuery = extractWikiQuery(claim);
  const wikiPromise = (async () => {
    try {
      const hits = await wikiProvider.search(wikiQuery || queries[0], 4);
      return (hits || []).map(h => ({ ...h, queryUsed: wikiQuery || queries[0] }));
    } catch (e) {
      return [];
    }
  })();

  const [providerResultsArray, wikiHits] = await Promise.all([
    Promise.all(searchPromises),
    wikiPromise
  ]);

  // Combine hits: Wikipedia hits first, then provider hits
  const allHits = [];
  if (Array.isArray(wikiHits)) {
    allHits.push(...wikiHits);
  }
  for (const resList of providerResultsArray) {
    if (Array.isArray(resList)) {
      allHits.push(...resList);
    }
  }

  for (const article of allHits) {
    if (!article || !article.url) continue;

    const canonicalUrl = normalizeCanonicalUrl(article.url);
    if (!seenUrls.has(canonicalUrl)) {
      seenUrls.add(canonicalUrl);
      const { score: authority, tier: qualityTier } = getSourceQualityTier(article.url, article.source);

      // Exclude low-quality / forum / tabloid domains
      if (qualityTier === 'low_quality') {
        continue;
      }

      collectedArticles.push({
        title: article.title,
        url: article.url,
        canonicalUrl: canonicalUrl,
        source: article.source || 'Web',
        sourceQuality: qualityTier,
        authorityScore: authority,
        queryUsed: article.queryUsed || queries[0] || null,
        publishedAt: article.publishedAt || null,
        description: article.description || null,
        retrievedAt: article.retrievedAt || new Date().toISOString()
      });
    }
  }

  // Sort collected articles by authority score descending, keeping maxArticles
  collectedArticles.sort((a, b) => b.authorityScore - a.authorityScore);
  const prioritizedArticles = collectedArticles.slice(0, maxArticles);

  return {
    queries,
    provider: primaryProvider.name,
    articles: prioritizedArticles
  };
};

module.exports = {
  searchNews,
  generateQueries,
  extractKeywords,
  extractWikiQuery,
  getSourceAuthorityScore,
  getSourceQualityTier,
  normalizeCanonicalUrl
};
