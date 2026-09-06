const factCheckService = require('../services/factCheckService');

/**
 * Handle POST /api/fact-check
 */
const createFactCheck = async (req, res, next) => {
  try {
    const { claim, evidence } = req.body;
    const result = await factCheckService.createFactCheck({ claim, evidence });

    res.status(201).json({
      success: true,
      data: result
    });
  } catch (error) {
    next(error);
  }
};

/**
 * Handle GET /api/fact-checks
 */
const getFactChecks = async (req, res, next) => {
  try {
    const { page, limit } = req.query;
    const result = await factCheckService.getFactChecks({ page, limit });

    res.status(200).json({
      success: true,
      data: result.records,
      pagination: result.pagination
    });
  } catch (error) {
    next(error);
  }
};

/**
 * Handle GET /api/fact-checks/:id
 */
const getFactCheckById = async (req, res, next) => {
  try {
    const { id } = req.params;
    const result = await factCheckService.getFactCheckById(id);

    res.status(200).json({
      success: true,
      data: result
    });
  } catch (error) {
    next(error);
  }
};

/**
 * Handle DELETE /api/fact-checks/:id
 */
const deleteFactCheck = async (req, res, next) => {
  try {
    const { id } = req.params;
    await factCheckService.deleteFactCheckById(id);

    res.status(200).json({
      success: true,
      message: 'Fact check record deleted successfully.'
    });
  } catch (error) {
    next(error);
  }
};

module.exports = {
  createFactCheck,
  getFactChecks,
  getFactCheckById,
  deleteFactCheck
};
