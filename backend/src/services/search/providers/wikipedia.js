const axios = require('axios');

/**
 * Wikipedia Search Provider
 * Retrieves factual and encyclopedic background articles with high semantic relevance.
 */
class WikipediaProvider {
  constructor(timeoutMs = 12000) {
    this.name = 'wikipedia';
    this.timeoutMs = timeoutMs;
  }

  async search(query, maxResults = 10) {
    try {
      const url = `https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch=${encodeURIComponent(query)}&format=json&utf8=`;
      const response = await axios.get(url, {
        headers: {
          'User-Agent': 'TruthLensFactChecker/1.0 (https://truthlens.ai; contact@truthlens.ai)'
        },
        timeout: this.timeoutMs
      });

      const searchHits = response.data?.query?.search || [];
      const results = [];

      for (const hit of searchHits) {
        if (results.length >= maxResults) break;
        if (!hit.title) continue;

        const cleanTitle = hit.title.trim();
        const articleUrl = `https://en.wikipedia.org/wiki/${encodeURIComponent(cleanTitle.replace(/ /g, '_'))}`;
        const cleanSnippet = (hit.snippet || '')
          .replace(/<span class="searchmatch">/g, '')
          .replace(/<\/span>/g, '')
          .replace(/<[^>]+>/g, ' ')
          .replace(/\s+/g, ' ')
          .trim();

        results.push({
          title: cleanTitle,
          url: articleUrl,
          source: 'Wikipedia',
          publishedAt: hit.timestamp || null,
          description: cleanSnippet || cleanTitle,
          retrievedAt: new Date().toISOString()
        });
      }

      return results;
    } catch (error) {
      console.warn(`[WikipediaProvider] Search failed for query "${query}": ${error.message}`);
      return [];
    }
  }
}

module.exports = WikipediaProvider;
