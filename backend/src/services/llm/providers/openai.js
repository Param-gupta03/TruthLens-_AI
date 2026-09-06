const axios = require('axios');

/**
 * OpenAI Chat Completions Provider
 */
class OpenAIProvider {
  constructor(apiKey, model = 'gpt-4o-mini', timeoutMs = 15000) {
    this.name = 'openai';
    this.apiKey = apiKey;
    this.model = model;
    this.timeoutMs = timeoutMs;
  }

  isConfigured() {
    return Boolean(this.apiKey && this.apiKey.trim().length > 0);
  }

  async generate(messages, options = {}) {
    if (!this.isConfigured()) {
      throw new Error('OpenAI API key is not configured. Please set LLM_API_KEY in .env.');
    }

    try {
      const response = await axios.post(
        'https://api.openai.com/v1/chat/completions',
        {
          model: this.model,
          messages: messages,
          temperature: options.temperature !== undefined ? options.temperature : 0.2,
          max_tokens: options.maxTokens || 1200,
          response_format: { type: 'json_object' }
        },
        {
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${this.apiKey}`
          },
          timeout: this.timeoutMs
        }
      );

      const content = response.data?.choices?.[0]?.message?.content;
      if (!content) {
        throw new Error('Empty response received from OpenAI API.');
      }

      return content;
    } catch (error) {
      const status = error.response?.status;
      const apiMessage = error.response?.data?.error?.message || error.message;

      if (status === 401) {
        throw new Error('Invalid OpenAI API key provided in LLM_API_KEY.');
      } else if (status === 429) {
        throw new Error('OpenAI rate limit reached or quota exceeded.');
      } else if (error.code === 'ECONNABORTED' || error.code === 'ETIMEDOUT') {
        throw new Error(`OpenAI request timed out after ${this.timeoutMs}ms.`);
      }

      throw new Error(`OpenAI API error: ${apiMessage}`);
    }
  }
}

module.exports = OpenAIProvider;
