const express = require('express');
const router = express.Router();
const factCheckController = require('../controllers/factCheckController');
const { validateFactCheckRequest, validateObjectId, validatePagination } = require('../middleware/validateRequest');

// POST /api/fact-check - Run ML fact check and save result
router.post('/fact-check', validateFactCheckRequest, factCheckController.createFactCheck);

// POST /api/fact-checks - Alias
router.post('/fact-checks', validateFactCheckRequest, factCheckController.createFactCheck);

// GET /api/fact-checks - Get paginated fact check history
router.get('/fact-checks', validatePagination, factCheckController.getFactChecks);

// GET /api/fact-checks/:id - Get single fact check by ID
router.get('/fact-checks/:id', validateObjectId(), factCheckController.getFactCheckById);

// DELETE /api/fact-checks/:id - Delete fact check by ID
router.delete('/fact-checks/:id', validateObjectId(), factCheckController.deleteFactCheck);

module.exports = router;
