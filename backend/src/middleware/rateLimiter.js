const rateLimit = require('express-rate-limit');
const config = require('../config/env');

/**
 * Strict rate limiter for automated research requests (web crawling + ML + LLM)
 * Protects search providers and scraping pipeline from burst abuse.
 */
const researchRateLimiter = rateLimit({
  windowMs: config.rateLimitWindowMs,
  max: config.rateLimitMaxResearch,
  standardHeaders: true,
  legacyHeaders: false,
  message: {
    success: false,
    message: `Too many research requests from this IP. Please try again after ${Math.ceil(config.rateLimitWindowMs / 60000)} minutes.`
  }
});

/**
 * General rate limiter for standard API endpoints
 */
const generalRateLimiter = rateLimit({
  windowMs: config.rateLimitWindowMs,
  max: config.rateLimitMaxGeneral,
  standardHeaders: true,
  legacyHeaders: false,
  message: {
    success: false,
    message: `Too many requests from this IP. Please try again later.`
  }
});

module.exports = {
  researchRateLimiter,
  generalRateLimiter
};
