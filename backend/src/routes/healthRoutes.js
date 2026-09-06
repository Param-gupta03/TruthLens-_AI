const express = require('express');
const router = express.Router();
const healthController = require('../controllers/healthController');

// GET /api/health
router.get('/', healthController.getHealth);

// GET /api/health/ml
router.get('/ml', healthController.getMlHealth);

module.exports = router;
