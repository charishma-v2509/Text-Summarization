// summaryRoutes.js

const express = require("express");
const router = express.Router();
const { generateSummary } = require("../controllers/summaryController");

// POST /api/summarize
router.post("/summarize", generateSummary);

module.exports = router;
