/**
 * passageChunker.js - Semantic passage extraction with 1-sentence context overlap
 * Step 6: Passage Chunking & Semantic Integrity
 */

/**
 * Splits raw article text into sentences, protecting decimals, scientific numbers,
 * and common abbreviations from false splits.
 */
const splitIntoSentences = (text) => {
  if (!text) return [];

  // Protect decimals, URLs, and abbreviations
  let safe = text
    .replace(/(\d+)\.(\d+)/g, '$1__DOT__$2')
    .replace(/\b(e\.g|i\.e|dr|mr|mrs|ms|prof|approx|vs|vol|fig|eq|no|sq|mi|km|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec|st|ft|al|ca|c|etc)\./gi, '$1__DOT__')
    .replace(/\[\d+\]/g, '') // strip wikipedia reference markers
    .replace(/\[[a-z]\]/g, '') // strip note markers
    .replace(/\[edit\]/gi, '')
    .replace(/\[update\]/gi, '');

  // Split on sentence-ending punctuation followed by whitespace and capital/quote/digit
  const rawSentences = safe.split(/(?<=[.!?])\s+(?=[A-Z0-9"'“‘])/);

  return rawSentences
    .map(s => s.replace(/__DOT__/g, '.').replace(/\s+/g, ' ').trim())
    .filter(s => s.length >= 12);
};

/**
 * Chunks article text into evidence passages with 1-sentence context overlap
 * Target: 2–5 sentences, 80–800 characters
 */
const chunkArticleIntoPassages = (article, articleIndex = 0, options = {}) => {
  const minSentences = options.minSentences || 2;
  const maxSentences = options.maxSentences || 4;
  const minChars = options.minChars || 80;
  const maxChars = options.maxChars || 800;

  const rawText = article.extractedText || article.text || article.description || '';
  if (!rawText || rawText.length < minChars) {
    if (rawText.length >= 40) {
      return [{
        passageId: `art_${articleIndex}_p0`,
        articleId: `art_${articleIndex}`,
        sourceTitle: article.title || 'Untitled',
        sourceUrl: article.url || null,
        source: article.source || 'Web',
        sourceQuality: article.sourceQuality || 'general',
        passageIndex: 0,
        queryUsed: article.queryUsed || null,
        publishedAt: article.publishedAt || null,
        text: rawText.trim()
      }];
    }
    return [];
  }

  const sentences = splitIntoSentences(rawText);
  if (sentences.length === 0) {
    return [{
      passageId: `art_${articleIndex}_p0`,
      articleId: `art_${articleIndex}`,
      sourceTitle: article.title || 'Untitled',
      sourceUrl: article.url || null,
      source: article.source || 'Web',
      sourceQuality: article.sourceQuality || 'general',
      passageIndex: 0,
      queryUsed: article.queryUsed || null,
      publishedAt: article.publishedAt || null,
      text: rawText.slice(0, maxChars).trim()
    }];
  }

  const passages = [];
  let i = 0;

  while (i < sentences.length) {
    let currentChunk = [];
    let currentLength = 0;
    let j = i;

    while (j < sentences.length) {
      const s = sentences[j];
      const nextLength = currentLength + (currentChunk.length > 0 ? 1 : 0) + s.length;

      // Stop chunk if adding sentence would exceed maxChars
      if (currentChunk.length >= minSentences && nextLength > maxChars) {
        break;
      }

      currentChunk.push(s);
      currentLength = nextLength;
      j++;

      // If reached target sentence count and minChars, close chunk
      if (currentChunk.length >= maxSentences && currentLength >= minChars) {
        break;
      }
    }

    if (currentChunk.length > 0 && currentLength >= minChars) {
      passages.push({
        passageId: `art_${articleIndex}_p${passages.length}`,
        articleId: `art_${articleIndex}`,
        sourceTitle: article.title || 'Untitled',
        sourceUrl: article.url || null,
        source: article.source || 'Web',
        sourceQuality: article.sourceQuality || 'general',
        passageIndex: passages.length,
        queryUsed: article.queryUsed || null,
        publishedAt: article.publishedAt || null,
        text: currentChunk.join(' ').trim()
      });
    }

    // 1-sentence overlap: advance by (sentencesInChunk - 1) if chunk had >= 2 sentences
    const advance = Math.max(1, currentChunk.length - 1);
    i += advance;

    if (j >= sentences.length) {
      break;
    }
  }

  // If no passage met minChars, create single passage from available text
  if (passages.length === 0 && rawText.length >= 35) {
    passages.push({
      passageId: `art_${articleIndex}_p0`,
      articleId: `art_${articleIndex}`,
      sourceTitle: article.title || 'Untitled',
      sourceUrl: article.url || null,
      source: article.source || 'Web',
      sourceQuality: article.sourceQuality || 'general',
      passageIndex: 0,
      queryUsed: article.queryUsed || null,
      publishedAt: article.publishedAt || null,
      text: rawText.slice(0, maxChars).trim()
    });
  }

  return passages;
};

/**
 * Batches passages across multiple articles using round-robin distribution
 * Ensures candidate passages cover diverse sources rather than being monopolized by article 0
 * Caps total candidate passages per claim (Maximum 30–50)
 */
const chunkAllArticles = (articles, maxPassages = 50) => {
  if (!articles || articles.length === 0) return [];

  const articlePassages = articles.map((art, idx) => chunkArticleIntoPassages(art, idx));
  const allPassages = [];
  let addedInRound = true;
  let round = 0;

  while (allPassages.length < maxPassages && addedInRound) {
    addedInRound = false;
    for (let i = 0; i < articlePassages.length; i++) {
      if (allPassages.length >= maxPassages) break;
      if (round < articlePassages[i].length) {
        allPassages.push(articlePassages[i][round]);
        addedInRound = true;
      }
    }
    round++;
  }

  return allPassages;
};

module.exports = {
  splitIntoSentences,
  chunkArticleIntoPassages,
  chunkAllArticles
};
