const axios = require('axios');

/**
 * NewsAPI.org Provider
 */
class NewsApiProvider {
  constructor(apiKey, timeoutMs = 12000) {
    if (!apiKey) {
      throw new Error('NewsAPI requires a valid API key configured via SEARCH_API_KEY.');
    }
    this.name = 'newsapi';
    this.apiKey = apiKey;
    this.timeoutMs = timeoutMs;
  }

  async search(query, maxResults = 10) {
    try {
      const response = await axios.get('https://newsapi.org/v2/everything', {
        params: {
          q: query,
          apiKey: this.apiKey,
          language: 'en',
          sortBy: 'relevancy',
          pageSize: Math.min(maxResults, 20)
        },
        timeout: this.timeoutMs
      });

      const articles = response.data?.articles || [];
      const results = [];

      for (const item of articles) {
        if (!item.url || !item.title) continue;

        results.push({
          title: item.title.trim(),
          url: item.url.trim(),
          source: item.source?.name || 'NewsAPI',
          publishedAt: item.publishedAt || null,
          description: item.description || item.content || null,
          retrievedAt: new Date().toISOString()
        });
      }

      return results;
    } catch (error) {
      const status = error.response?.status;
      const msg = error.response?.data?.message || error.message;
      console.warn(`[NewsApiProvider] Search error (${status}): ${msg}`);
      throw new Error(`NewsAPI request failed: ${msg}`);
    }
  }
}

module.exports = NewsApiProvider;
