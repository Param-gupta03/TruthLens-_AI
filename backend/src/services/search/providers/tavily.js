const axios = require('axios');

/**
 * Tavily Search Provider
 * Tailored for AI search & verification tasks
 */
class TavilyProvider {
  constructor(apiKey, timeoutMs = 12000) {
    if (!apiKey) {
      throw new Error('Tavily provider requires a valid API key configured via SEARCH_API_KEY.');
    }
    this.name = 'tavily';
    this.apiKey = apiKey;
    this.timeoutMs = timeoutMs;
  }

  async search(query, maxResults = 10) {
    try {
      const response = await axios.post(
        'https://api.tavily.com/search',
        {
          api_key: this.apiKey,
          query: query,
          search_depth: 'basic',
          include_raw_content: false,
          max_results: maxResults
        },
        {
          headers: { 'Content-Type': 'application/json' },
          timeout: this.timeoutMs
        }
      );

      const results = (response.data?.results || []).map(item => {
        let domain = 'Tavily';
        try {
          domain = new URL(item.url).hostname.replace(/^www\./, '');
        } catch (e) {}

        return {
          title: item.title?.trim() || 'Untitled',
          url: item.url.trim(),
          source: domain,
          publishedAt: item.published_date || null,
          description: item.content?.trim() || null,
          retrievedAt: new Date().toISOString()
        };
      });

      return results;
    } catch (error) {
      const msg = error.response?.data?.message || error.message;
      console.warn(`[TavilyProvider] Search error: ${msg}`);
      throw new Error(`Tavily search failed: ${msg}`);
    }
  }
}

module.exports = TavilyProvider;
