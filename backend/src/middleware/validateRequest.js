const mongoose = require('mongoose');

/**
 * Validates the fact-check request body per specifications:
 * - claim: non-empty string, max 2000 chars
 * - evidence: array of non-empty strings, min 1 item, max 20 items, each max 10000 chars
 */
const validateFactCheckRequest = (req, res, next) => {
  const { claim, evidence } = req.body;

  if (!claim || typeof claim !== 'string' || claim.trim().length === 0) {
    return res.status(400).json({
      success: false,
      message: 'Claim must be a non-empty string.'
    });
  }

  if (claim.length > 2000) {
    return res.status(400).json({
      success: false,
      message: 'Claim exceeds maximum allowed length of 2,000 characters.'
    });
  }

  if (!Array.isArray(evidence)) {
    return res.status(400).json({
      success: false,
      message: 'Evidence must be an array of strings.'
    });
  }

  if (evidence.length === 0) {
    return res.status(400).json({
      success: false,
      message: 'Evidence array cannot be empty. At least one evidence passage is required.'
    });
  }

  if (evidence.length > 20) {
    return res.status(400).json({
      success: false,
      message: 'Maximum allowed evidence passages is 20.'
    });
  }

  for (let i = 0; i < evidence.length; i++) {
    const item = evidence[i];
    if (typeof item !== 'string' || item.trim().length === 0) {
      return res.status(400).json({
        success: false,
        message: `Evidence item at index ${i} must be a non-empty string.`
      });
    }

    if (item.length > 10000) {
      return res.status(400).json({
        success: false,
        message: `Evidence item at index ${i} exceeds maximum allowed length of 10,000 characters.`
      });
    }
  }

  next();
};

/**
 * Validates the research request body per Step 20 specifications:
 * - claim: non-empty string, max 2000 chars
 */
const validateResearchRequest = (req, res, next) => {
  const { claim } = req.body;

  if (!claim || typeof claim !== 'string' || claim.trim().length === 0) {
    return res.status(400).json({
      success: false,
      message: 'Claim must be a non-empty string.'
    });
  }

  if (claim.length > 2000) {
    return res.status(400).json({
      success: false,
      message: 'Claim exceeds maximum allowed length of 2,000 characters.'
    });
  }

  next();
};

/**
 * Validates that a route parameter is a valid MongoDB ObjectId
 */
const validateObjectId = (paramName = 'id') => {
  return (req, res, next) => {
    const value = req.params[paramName];
    if (!value || !mongoose.Types.ObjectId.isValid(value)) {
      return res.status(400).json({
        success: false,
        message: `Invalid ID format for parameter: ${paramName}. Must be a 24-character hexadecimal ObjectId.`
      });
    }
    next();
  };
};

/**
 * Validates pagination query parameters (page, limit)
 */
const validatePagination = (req, res, next) => {
  const { page, limit } = req.query;

  if (page !== undefined) {
    const pageNum = Number(page);
    if (!Number.isInteger(pageNum) || pageNum < 1) {
      return res.status(400).json({
        success: false,
        message: "Query parameter 'page' must be a positive integer >= 1."
      });
    }
  }

  if (limit !== undefined) {
    const limitNum = Number(limit);
    if (!Number.isInteger(limitNum) || limitNum < 1 || limitNum > 100) {
      return res.status(400).json({
        success: false,
        message: "Query parameter 'limit' must be an integer between 1 and 100."
      });
    }
  }

  next();
};

module.exports = {
  validateFactCheckRequest,
  validateResearchRequest,
  validateObjectId,
  validatePagination
};
