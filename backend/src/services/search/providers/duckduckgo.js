const axios = require('axios');
const cheerio = require('cheerio');
const GoogleNewsProvider = require('./googleNews');
const WikipediaProvider = require('./wikipedia');

/**
 * DuckDuckGo Search Provider
 * Attempts DuckDuckGo Lite search first; if blocked by anti-bot verification or empty,
 * seamlessly falls back to Google News and Wikipedia.
 */
class DuckDuckGoProvider {
  constructor(timeoutMs = 12000) {
    this.name = 'duckduckgo';
    this.timeoutMs = timeoutMs;
    this.googleNews = new GoogleNewsProvider(timeoutMs);
    this.wikipedia = new WikipediaProvider(timeoutMs);
  }

  async search(query, maxResults = 10) {
    let ddgResults = [];

    try {
      const response = await axios.post(
        'https://lite.duckduckgo.com/lite/',
        `q=${encodeURIComponent(query)}`,
        {
          headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml',
            'Accept-Language': 'en-US,en;q=0.9'
          },
          timeout: this.timeoutMs
        }
      );

      // Status 200 means actual search results were returned
      if (response.status === 200 && typeof response.data === 'string') {
        const $ = cheerio.load(response.data);

        $('a.result-link').each((i, el) => {
          if (ddgResults.length >= maxResults) return false;

          const linkEl = $(el);
          let rawHref = linkEl.attr('href') || '';
          let finalUrl = rawHref;

          if (rawHref.includes('uddg=')) {
            try {
              const match = rawHref.match(/uddg=([^&]+)/);
              if (match) {
                finalUrl = decodeURIComponent(match[1]);
              }
            } catch (e) {}
          }

          const title = linkEl.text().trim();
          const tr = linkEl.closest('tr');
          const snippet = tr.next().find('td.result-snippet').text().trim();

          if (finalUrl && (finalUrl.startsWith('http://') || finalUrl.startsWith('https://')) && title) {
            let sourceDomain = 'Web';
            try {
              sourceDomain = new URL(finalUrl).hostname.replace(/^www\./, '');
            } catch (e) {}

            ddgResults.push({
              title,
              url: finalUrl,
              source: sourceDomain,
              publishedAt: null,
              description: snippet || null,
              retrievedAt: new Date().toISOString()
            });
          }
        });
      }
    } catch (error) {
      console.warn(`[DuckDuckGoProvider] DDG endpoint note: ${error.message}`);
    }

    if (ddgResults.length > 0) {
      return ddgResults.slice(0, maxResults);
    }

    // Fallback: Query Google News and Wikipedia concurrently
    try {
      const [wikiHits, newsHits] = await Promise.all([
        this.wikipedia.search(query, Math.ceil(maxResults / 2)),
        this.googleNews.search(query, Math.ceil(maxResults / 2))
      ]);

      const combined = [...wikiHits, ...newsHits];
      return combined.slice(0, maxResults);
    } catch (fallbackError) {
      console.warn(`[DuckDuckGoProvider] Fallback search error: ${fallbackError.message}`);
      return [];
    }
  }
}

module.exports = DuckDuckGoProvider;
