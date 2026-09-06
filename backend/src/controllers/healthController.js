const mongoose = require('mongoose');
const mlService = require('../services/mlService');

/**
 * Health check for the Node.js API server & Database connectivity
 */
const getHealth = (req, res) => {
  const dbState = mongoose.connection.readyState;
  const dbStatusMap = {
    0: 'disconnected',
    1: 'connected',
    2: 'connecting',
    3: 'disconnecting'
  };

  const mem = process.memoryUsage();

  res.status(200).json({
    success: true,
    service: 'TruthLens API',
    status: 'healthy',
    database: {
      status: dbStatusMap[dbState] || 'unknown',
      connected: dbState === 1
    },
    uptimeSeconds: Math.floor(process.uptime()),
    timestamp: new Date().toISOString(),
    system: {
      nodeVersion: process.version,
      platform: process.platform,
      memoryRssMb: Number((mem.rss / (1024 * 1024)).toFixed(1)),
      heapUsedMb: Number((mem.heapUsed / (1024 * 1024)).toFixed(1))
    }
  });
};

/**
 * Health check for Python ML inference service
 * Pings ML service /health and returns device, CUDA, and model readiness
 */
const getMlHealth = async (req, res, next) => {
  try {
    const result = await mlService.checkHealth();

    if (result.healthy && result.data) {
      return res.status(200).json({
        success: true,
        status: 'healthy',
        mlService: 'healthy',
        cuda: result.data.cuda ?? false,
        device: result.data.device || 'unknown',
        modelA_loaded: result.data.modelA_loaded ?? true,
        modelB_loaded: result.data.modelB_loaded ?? true,
        details: result.data
      });
    } else {
      return res.status(503).json({
        success: false,
        status: 'degraded',
        mlService: 'unreachable',
        message: 'Python ML service is currently unreachable.',
        error: result.error || 'Connection failed'
      });
    }
  } catch (error) {
    next(error);
  }
};

module.exports = {
  getHealth,
  getMlHealth
};
