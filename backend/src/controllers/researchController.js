const retrievalService = require('../services/retrievalService');
const reportService = require('../services/reportService');
const FactCheck = require('../models/FactCheck');
const { AppError } = require('../middleware/errorHandler');
const mongoose = require('mongoose');

/**
 * Handle POST /api/research
 * Automatically discovers news/web evidence, scores relevance with Model A,
 * verifies claim with Model B, generates LLM explanation, and saves record to MongoDB.
 */
const conductResearch = async (req, res, next) => {
  try {
    const { claim, refresh } = req.body;
    const result = await retrievalService.researchClaim(claim, { refresh: refresh === true });

    res.status(200).json({
      success: true,
      data: result
    });
  } catch (error) {
    next(error);
  }
};

/**
 * Handle GET /api/research/:id/report
 * Generates a full research report from an existing fact check record (Step 10)
 */
const getResearchReport = async (req, res, next) => {
  try {
    const { id } = req.params;
    if (!mongoose.Types.ObjectId.isValid(id)) {
      throw new AppError('Invalid research record ID format.', 400);
    }

    const record = await FactCheck.findById(id).lean();
    if (!record) {
      throw new AppError('Research record not found.', 404);
    }

    const report = reportService.buildResearchReport(record);
    res.status(200).json({
      success: true,
      data: report
    });
  } catch (error) {
    next(error);
  }
};

module.exports = {
  conductResearch,
  getResearchReport
};

