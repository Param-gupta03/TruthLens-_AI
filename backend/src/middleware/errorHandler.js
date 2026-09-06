/**
 * Custom application error class
 */
class AppError extends Error {
  constructor(message, statusCode) {
    super(message);
    this.statusCode = statusCode;
    this.isOperational = true;
    Error.captureStackTrace(this, this.constructor);
  }
}

/**
 * 404 Handler for undefined routes
 */
const notFoundHandler = (req, res, next) => {
  res.status(404).json({
    success: false,
    message: `Route not found: ${req.method} ${req.originalUrl}`
  });
};

/**
 * Centralized error handling middleware
 */
const errorHandler = (err, req, res, next) => {
  let statusCode = err.statusCode || 500;
  let message = err.message || 'Internal Server Error';

  // Handle Mongoose CastError (e.g. invalid ObjectId)
  if (err.name === 'CastError') {
    statusCode = 400;
    message = `Invalid format for field: ${err.path}`;
  }

  // Handle Mongoose ValidationError
  if (err.name === 'ValidationError') {
    statusCode = 400;
    const messages = Object.values(err.errors).map(val => val.message);
    message = `Validation Error: ${messages.join(', ')}`;
  }

  // Handle Axios / ML service connection errors
  if (err.isAxiosError) {
    if (err.code === 'ECONNREFUSED') {
      statusCode = 503;
      message = 'Machine Learning inference service is currently unavailable. Please ensure the ML service is running.';
    } else if (err.code === 'ETIMEDOUT' || err.code === 'ECONNABORTED') {
      statusCode = 504;
      message = 'Machine Learning inference service timed out processing the request.';
    } else if (err.response) {
      statusCode = err.response.status >= 500 ? 502 : err.response.status;
      message = err.response.data?.detail || 'Error received from Machine Learning service.';
    } else {
      statusCode = 502;
      message = 'Failed to communicate with Machine Learning service.';
    }
  }

  // Handle CORS errors
  if (err.message && err.message.includes('CORS policy')) {
    statusCode = 403;
  }

  // Handle Payload Too Large
  if (err.type === 'entity.too.large' || err.status === 413) {
    statusCode = 413;
    message = 'Request payload exceeds maximum allowed size.';
  }

  // Log error internally for debugging without exposing secrets or paths in HTTP response
  if (process.env.NODE_ENV !== 'test') {
    console.error(`[Error] [${new Date().toISOString()}] ${req.method} ${req.url} - ${statusCode}: ${err.message}`);
  }

  res.status(statusCode).json({
    success: false,
    message: message
  });
};

module.exports = {
  AppError,
  notFoundHandler,
  errorHandler
};
