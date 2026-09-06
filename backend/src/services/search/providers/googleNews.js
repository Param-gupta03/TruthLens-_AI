const axios = require('axios');
const cheerio = require('cheerio');

/**
 * Google News RSS Search Provider
 * Retrieves real, up-to-date news articles with headline, source publisher, and publication timestamp.
 */
class GoogleNewsProvider {
  constructor(timeoutMs = 12000) {
    this.name = 'googlenews';
    this.timeoutMs = timeoutMs;
  }

  async search(query, maxResults = 10) {
    try {
      const url = `https://news.google.com/rss/search?q=${encodeURIComponent(query)}&hl=en-US&gl=US&ceid=US:en`;
      const response = await axios.get(url, {
        headers: {
          'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
          'Accept': 'application/rss+xml, application/xml, text/xml'
        },
        timeout: this.timeoutMs
      });

      const $ = cheerio.load(response.data, { xmlMode: true });
      const results = [];

      $('item').each((i, el) => {
        if (results.length >= maxResults) return false;

        const title = $(el).find('title').text().trim();
        const link = $(el).find('link').text().trim();
        const pubDate = $(el).find('pubDate').text().trim();
        const source = $(el).find('source').text().trim() || 'Google News';
        const description = $(el).find('description').text().replace(/<[^>]+>/g, ' ').trim();

        if (title && link) {
          results.push({
            title,
            url: link,
            source,
            publishedAt: pubDate ? new Date(pubDate).toISOString() : null,
            description: description || title,
            retrievedAt: new Date().toISOString()
          });
        }
      });

      return results;
    } catch (error) {
      console.warn(`[GoogleNewsProvider] Search failed for query "${query}": ${error.message}`);
      return [];
    }
  }
}

module.exports = GoogleNewsProvider;
