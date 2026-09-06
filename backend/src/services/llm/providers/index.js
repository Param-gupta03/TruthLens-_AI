const OpenAIProvider = require('./openai');
const GeminiProvider = require('./gemini');
const config = require('../../../config/env');

/**
 * Factory for creating the configured LLM provider
 */
const getLLMProvider = (
  providerName = config.llmProvider,
  apiKey = config.llmApiKey,
  model = config.llmModel,
  timeoutMs = config.llmTimeoutMs
) => {
  const selected = (providerName || 'openai').toLowerCase().trim();

  switch (selected) {
    case 'gemini':
    case 'google':
      return new GeminiProvider(apiKey, model || 'gemini-1.5-flash', timeoutMs);

    case 'openai':
    default:
      return new OpenAIProvider(apiKey, model || 'gpt-4o-mini', timeoutMs);
  }
};

module.exports = {
  getLLMProvider,
  OpenAIProvider,
  GeminiProvider
};
