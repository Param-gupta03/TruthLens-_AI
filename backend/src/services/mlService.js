const axios = require('axios');
const config = require('../config/env');
const { AppError } = require('../middleware/errorHandler');

const mlClient = axios.create({
  baseURL: config.mlServiceUrl,
  timeout: 30000, // 30s timeout for model inference
  headers: {
    'Content-Type': 'application/json'
  }
});

/**
 * Sends claim and candidate evidence to the Python FastAPI ML service.
 */
const predict = async (claim, evidence, options = {}) => {
  try {
    const payload = {
      claim,
      evidence
    };

    if (typeof options.threshold === 'number') {
      payload.threshold = options.threshold;
    } else if (typeof config.evidenceRelevanceThreshold === 'number') {
      payload.threshold = config.evidenceRelevanceThreshold;
    }

    const response = await mlClient.post('/predict', payload);

    if (!response.data || !response.data.verification || !Array.isArray(response.data.evidenceResults)) {
      throw new AppError('Invalid response structure received from Machine Learning service.', 502);
    }

    return response.data;
  } catch (error) {
    // If it's already an AppError, rethrow
    if (error.isOperational) throw error;
    // Otherwise re-throw Axios error for centralized error handler
    throw error;
  }
};

/**
 * Checks connectivity and health of the Python ML service.
 */
const checkHealth = async () => {
  try {
    const response = await mlClient.get('/health', { timeout: 5000 });
    return {
      healthy: response.status === 200,
      data: response.data
    };
  } catch (error) {
    return {
      healthy: false,
      error: error.message
    };
  }
};

module.exports = {
  predict,
  checkHealth
};
