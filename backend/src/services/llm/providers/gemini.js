const axios = require('axios');

/**
 * Google Gemini Provider
 */
class GeminiProvider {
  constructor(apiKey, model = 'gemini-1.5-flash', timeoutMs = 15000) {
    this.name = 'gemini';
    this.apiKey = apiKey;
    this.model = model;
    this.timeoutMs = timeoutMs;
  }

  isConfigured() {
    return Boolean(this.apiKey && this.apiKey.trim().length > 0);
  }

  async generate(messages, options = {}) {
    if (!this.isConfigured()) {
      throw new Error('Gemini API key is not configured. Please set LLM_API_KEY in .env.');
    }

    // Separate system instruction and user contents
    let systemInstruction = null;
    const contents = [];

    for (const msg of messages) {
      if (msg.role === 'system') {
        systemInstruction = {
          parts: [{ text: msg.content }]
        };
      } else {
        contents.push({
          role: msg.role === 'assistant' ? 'model' : 'user',
          parts: [{ text: msg.content }]
        });
      }
    }

    try {
      const url = `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(this.model)}:generateContent?key=${encodeURIComponent(this.apiKey)}`;

      const body = {
        contents: contents,
        generationConfig: {
          temperature: options.temperature !== undefined ? options.temperature : 0.2,
          maxOutputTokens: options.maxTokens || 1200,
          responseMimeType: 'application/json'
        }
      };

      if (systemInstruction) {
        body.systemInstruction = systemInstruction;
      }

      const response = await axios.post(url, body, {
        headers: { 'Content-Type': 'application/json' },
        timeout: this.timeoutMs
      });

      const candidate = response.data?.candidates?.[0];
      const text = candidate?.content?.parts?.[0]?.text;

      if (!text) {
        throw new Error('Empty or blocked response received from Gemini API.');
      }

      return text;
    } catch (error) {
      const status = error.response?.status;
      const apiMessage = error.response?.data?.error?.message || error.message;

      if (status === 400 && apiMessage.toLowerCase().includes('api_key')) {
        throw new Error('Invalid Gemini API key in LLM_API_KEY.');
      } else if (status === 429) {
        throw new Error('Gemini API quota or rate limit exceeded.');
      } else if (error.code === 'ECONNABORTED' || error.code === 'ETIMEDOUT') {
        throw new Error(`Gemini request timed out after ${this.timeoutMs}ms.`);
      }

      throw new Error(`Gemini API error: ${apiMessage}`);
    }
  }
}

module.exports = GeminiProvider;
