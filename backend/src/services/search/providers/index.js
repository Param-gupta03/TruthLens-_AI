const DuckDuckGoProvider = require('./duckduckgo');
const NewsApiProvider = require('./newsapi');
const TavilyProvider = require('./tavily');
const SerperProvider = require('./serper');
const GoogleNewsProvider = require('./googleNews');
const WikipediaProvider = require('./wikipedia');
const config = require('../../../config/env');

/**
 * Provider factory function
 */
const getSearchProvider = (providerName = config.searchProvider, apiKey = config.searchApiKey) => {
  const selected = (providerName || 'duckduckgo').toLowerCase().trim();

  switch (selected) {
    case 'newsapi':
      if (!apiKey) {
        throw new Error('NewsAPI provider selected but SEARCH_API_KEY is not configured in .env.');
      }
      return new NewsApiProvider(apiKey, config.searchTimeoutMs);

    case 'tavily':
      if (!apiKey) {
        throw new Error('Tavily provider selected but SEARCH_API_KEY is not configured in .env.');
      }
      return new TavilyProvider(apiKey, config.searchTimeoutMs);

    case 'serper':
    case 'google':
      if (!apiKey) {
        throw new Error('Serper provider selected but SEARCH_API_KEY is not configured in .env.');
      }
      return new SerperProvider(apiKey, config.searchTimeoutMs);

    case 'googlenews':
    case 'news':
      return new GoogleNewsProvider(config.searchTimeoutMs);

    case 'wikipedia':
    case 'wiki':
      return new WikipediaProvider(config.searchTimeoutMs);

    case 'duckduckgo':
    default:
      return new DuckDuckGoProvider(config.searchTimeoutMs);
  }
};

module.exports = {
  getSearchProvider,
  DuckDuckGoProvider,
  GoogleNewsProvider,
  WikipediaProvider,
  NewsApiProvider,
  TavilyProvider,
  SerperProvider
};
