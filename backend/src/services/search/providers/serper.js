const axios = require('axios');

/**
 * Serper (Google Search API) Provider
 */
class SerperProvider {
  constructor(apiKey, timeoutMs = 12000) {
    if (!apiKey) {
      throw new Error('Serper provider requires a valid API key configured via SEARCH_API_KEY.');
    }
    this.name = 'serper';
    this.apiKey = apiKey;
    this.timeoutMs = timeoutMs;
  }

  async search(query, maxResults = 10) {
    try {
      const response = await axios.post(
        'https://google.serper.dev/news',
        {
          q: query,
          num: maxResults
        },
        {
          headers: {
            'X-API-KEY': this.apiKey,
            'Content-Type': 'application/json'
          },
          timeout: this.timeoutMs
        }
      );

      const newsItems = response.data?.news || response.data?.organic || [];
      const results = [];

      for (const item of newsItems) {
        if (!item.link || !item.title) continue;

        let domain = 'Google';
        try {
          domain = new URL(item.link).hostname.replace(/^www\./, '');
        } catch (e) {}

        results.push({
          title: item.title.trim(),
          url: item.link.trim(),
          source: item.source || domain,
          publishedAt: item.date || null,
          description: item.snippet || null,
          retrievedAt: new Date().toISOString()
        });
      }

      return results;
    } catch (error) {
      const msg = error.response?.data?.message || error.message;
      console.warn(`[SerperProvider] Search error: ${msg}`);
      throw new Error(`Serper search failed: ${msg}`);
    }
  }
}

module.exports = SerperProvider;
