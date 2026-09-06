const dotenv = require('dotenv');
const path = require('path');

dotenv.config({ path: path.resolve(__dirname, '../../.env') });

const config = {
  port: parseInt(process.env.PORT, 10) || 5000,
  mongoUri: process.env.MONGO_URI || 'mongodb+srv://<username>:<password>@cluster0.b0eu6.mongodb.net/truthlens?retryWrites=true&w=majority',
  mlServiceUrl: process.env.ML_SERVICE_URL || 'http://127.0.0.1:8000',
  nodeEnv: process.env.NODE_ENV || 'development',
  searchProvider: (process.env.SEARCH_PROVIDER || 'duckduckgo').toLowerCase(),
  searchApiKey: process.env.SEARCH_API_KEY || '',
  maxArticles: parseInt(process.env.MAX_ARTICLES, 10) || 8,
  maxPassages: parseInt(process.env.MAX_PASSAGES, 10) || 50,
  evidenceRelevanceThreshold: process.env.EVIDENCE_RELEVANCE_THRESHOLD ? parseFloat(process.env.EVIDENCE_RELEVANCE_THRESHOLD) : 0.25,
  searchTimeoutMs: parseInt(process.env.SEARCH_TIMEOUT_MS, 10) || 12000,
  extractionTimeoutMs: parseInt(process.env.EXTRACTION_TIMEOUT_MS, 10) || 8000,
  cacheTtlHours: parseInt(process.env.CACHE_TTL_HOURS, 10) || 2,
  llmProvider: (process.env.LLM_PROVIDER || 'openai').toLowerCase().trim(),
  llmModel: process.env.LLM_MODEL || (process.env.LLM_PROVIDER === 'gemini' ? 'gemini-1.5-flash' : 'gpt-4o-mini'),
  llmApiKey: process.env.LLM_API_KEY || process.env.OPENAI_API_KEY || process.env.GEMINI_API_KEY || '',
  llmTimeoutMs: parseInt(process.env.LLM_TIMEOUT_MS, 10) || 15000,
  maxLlmEvidence: parseInt(process.env.MAX_LLM_EVIDENCE, 10) || 5,
  corsOrigin: process.env.CORS_ORIGIN || 'http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173',
  rateLimitWindowMs: parseInt(process.env.RATE_LIMIT_WINDOW_MS, 10) || 15 * 60 * 1000,
  rateLimitMaxResearch: parseInt(process.env.RATE_LIMIT_MAX_RESEARCH, 10) || 30,
  rateLimitMaxGeneral: parseInt(process.env.RATE_LIMIT_MAX_GENERAL, 10) || 120,
  requestTimeoutMs: parseInt(process.env.REQUEST_TIMEOUT_MS, 10) || 60000,
  bodyLimit: process.env.BODY_LIMIT || '1mb'
};

module.exports = config;
