const express = require('express');
const router = express.Router();
const researchController = require('../controllers/researchController');
const { validateResearchRequest, validateObjectId } = require('../middleware/validateRequest');
const { researchRateLimiter } = require('../middleware/rateLimiter');

// POST /api/research - Automated news & web evidence research
router.post('/research', researchRateLimiter, validateResearchRequest, researchController.conductResearch);

// GET /api/research/:id/report - Build complete research report (Step 10)
router.get('/research/:id/report', validateObjectId(), researchController.getResearchReport);

module.exports = router;

