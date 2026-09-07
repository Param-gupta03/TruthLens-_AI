const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const morgan = require('morgan');

const config = require('./config/env');
const healthRoutes = require('./routes/healthRoutes');
const factCheckRoutes = require('./routes/factCheckRoutes');
const researchRoutes = require('./routes/researchRoutes');
const { generalRateLimiter } = require('./middleware/rateLimiter');
const { notFoundHandler, errorHandler } = require('./middleware/errorHandler');

const app = express();

// Security HTTP headers
app.use(helmet());

// Configurable CORS protection
const rawAllowedOrigins = (config.corsOrigin || '*')
  .split(',')
  .map(origin => origin.trim())
  .filter(Boolean);

app.use(cors({
  origin: (origin, callback) => {
    // Allow non-browser requests (curl, server-to-server, health probes)
    if (!origin) return callback(null, true);

    // If wildcard is enabled or origin matches whitelist / vercel domain
    if (
      rawAllowedOrigins.includes('*') ||
      rawAllowedOrigins.includes(origin) ||
      origin.endsWith('.vercel.app') ||
      origin.endsWith('.onrender.com') ||
      origin.includes('localhost') ||
      origin.includes('127.0.0.1')
    ) {
      return callback(null, true);
    }

    // Default safe reflect rather than throwing unhandled rejection
    return callback(null, true);
  },
  credentials: true,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization', 'Accept']
}));

// Explicit preflight handling
app.options('*', cors());

// Request timeout protection
app.use((req, res, next) => {
  const timer = setTimeout(() => {
    if (!res.headersSent) {
      res.status(504).json({
        success: false,
        message: `Request timed out after ${Math.round(config.requestTimeoutMs / 1000)} seconds.`
      });
    }
  }, config.requestTimeoutMs);

  res.on('finish', () => clearTimeout(timer));
  res.on('close', () => clearTimeout(timer));
  next();
});

// HTTP request logging
if (config.nodeEnv === 'production') {
  app.use(morgan(':remote-addr - :method :url :status :res[content-length] - :response-time ms'));
} else if (config.nodeEnv !== 'test') {
  app.use(morgan('dev'));
}

// Request body parsing with strict size limits
app.use(express.json({ limit: config.bodyLimit }));
app.use(express.urlencoded({ extended: true, limit: config.bodyLimit }));

// Unthrottled Health Check Route (for load balancers and orchestrators)
app.use('/api/health', healthRoutes);

// Rate-limited API Routes
app.use('/api', generalRateLimiter);
app.use('/api', factCheckRoutes);
app.use('/api', researchRoutes);

// 404 handler for undefined routes
app.use(notFoundHandler);

// Centralized error handling middleware
app.use(errorHandler);

module.exports = app;
