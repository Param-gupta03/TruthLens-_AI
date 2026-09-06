const { getLLMProvider } = require('./providers');
const {
  SYSTEM_PROMPT,
  buildUserPrompt,
  buildCorrectionPrompt,
  buildConsistencyCorrectionPrompt
} = require('./promptBuilder');
const config = require('../../config/env');

/**
 * Validates that an object conforms to the expected LLM explanation schema
 */
const validateExplanationSchema = (data) => {
  if (!data || typeof data !== 'object') return false;

  if (typeof data.summary !== 'string' || data.summary.trim().length === 0) return false;
  if (typeof data.verdictExplanation !== 'string' || data.verdictExplanation.trim().length === 0) return false;
  if (!Array.isArray(data.keyEvidence)) return false;
  if (!Array.isArray(data.contradictoryEvidence)) return false;
  if (typeof data.uncertainty !== 'string') return false;
  if (!Array.isArray(data.sourceNotes)) return false;

  return true;
};

/**
 * Cleans and parses raw LLM text into JSON
 */
const parseJsonSafely = (rawText) => {
  if (!rawText || typeof rawText !== 'string') return null;

  let cleaned = rawText.trim();
  // Strip markdown code block wrappers if present (e.g. ```json ... ```)
  if (cleaned.startsWith('```')) {
    cleaned = cleaned.replace(/^```(?:json)?\s*/i, '').replace(/```\s*$/, '').trim();
  }

  try {
    return JSON.parse(cleaned);
  } catch (e) {
    return null;
  }
};

/**
 * Verifies that the generated explanation aligns with the fixed ML verdict (Step 18 & 19)
 */
const checkVerdictConsistency = (explanation, expectedVerdict) => {
  const combined = `${explanation.summary} ${explanation.verdictExplanation}`.toLowerCase();

  if (expectedVerdict === 'SUPPORTS') {
    // If verdict is SUPPORTS, the explanation must not claim the statement is completely disproven/false
    if (combined.includes('is completely false') || combined.includes('has been thoroughly debunked') || combined.includes('is entirely refuted')) {
      return false;
    }
  } else if (expectedVerdict === 'REFUTES') {
    // If verdict is REFUTES, the explanation must not state the claim is verified true
    if (combined.includes('is fully proven to be true') || combined.includes('is completely supported by facts') || combined.includes('is factually accurate and true')) {
      return false;
    }
  } else if (expectedVerdict === 'NOT_ENOUGH_INFO') {
    // If verdict is NOT_ENOUGH_INFO, explanation should indicate insufficiency or inconclusiveness
    const signalsInsufficiency =
      combined.includes('insufficient') ||
      combined.includes('not enough') ||
      combined.includes('inconclusive') ||
      combined.includes('limited') ||
      combined.includes('unclear') ||
      combined.includes('no clear');

    if (!signalsInsufficiency && (combined.includes('conclusively proves') || combined.includes('definitively refutes'))) {
      return false;
    }
  }

  return true;
};

/**
 * Generates an objective, structured explanation of the ML fact-check result
 */
const generateExplanation = async (input, options = {}) => {
  const provider = options.providerInstance || getLLMProvider(
    options.provider || config.llmProvider,
    options.apiKey || config.llmApiKey,
    options.model || config.llmModel,
    options.timeoutMs || config.llmTimeoutMs
  );

  // Step 14: Graceful fallback when API key is unconfigured
  if (!provider.isConfigured()) {
    return {
      explanation: null,
      llm: {
        status: 'unavailable',
        reason: 'LLM API key is not configured. Set LLM_API_KEY in .env.',
        provider: provider.name,
        model: provider.model
      }
    };
  }

  // Limit evidence passages sent to the LLM (Step 16: Token Control)
  const maxEvidence = options.maxEvidence || config.maxLlmEvidence;
  const filteredEvidence = (input.evidence || []).slice(0, maxEvidence);

  const userPrompt = buildUserPrompt({
    ...input,
    evidence: filteredEvidence
  });

  const messages = [
    { role: 'system', content: SYSTEM_PROMPT },
    { role: 'user', content: userPrompt }
  ];

  try {
    // Step 1: Initial Generation
    let rawOutput = await provider.generate(messages);
    let parsed = parseJsonSafely(rawOutput);

    // Step 6: Single retry on invalid JSON
    if (!parsed || !validateExplanationSchema(parsed)) {
      console.warn(`[llmService] Malformed JSON from ${provider.name}. Retrying once...`);
      const retryMessages = [
        ...messages,
        { role: 'assistant', content: rawOutput || '' },
        { role: 'user', content: buildCorrectionPrompt(rawOutput, 'Invalid schema or unparseable JSON') }
      ];

      rawOutput = await provider.generate(retryMessages);
      parsed = parseJsonSafely(rawOutput);
    }

    if (!parsed || !validateExplanationSchema(parsed)) {
      console.warn(`[llmService] Output failed schema validation after retry.`);
      return {
        explanation: null,
        llm: {
          status: 'unavailable',
          reason: 'LLM generated invalid response schema after retry.',
          provider: provider.name,
          model: provider.model
        }
      };
    }

    // Step 19: Verdict consistency verification
    let isConsistent = checkVerdictConsistency(parsed, input.verdict);
    if (!isConsistent) {
      console.warn(`[llmService] Explanation contradicted ML verdict "${input.verdict}". Retrying once...`);
      const consistencyMessages = [
        ...messages,
        { role: 'assistant', content: JSON.stringify(parsed) },
        { role: 'user', content: buildConsistencyCorrectionPrompt(input.verdict) }
      ];

      const correctedRaw = await provider.generate(consistencyMessages);
      const correctedParsed = parseJsonSafely(correctedRaw);

      if (correctedParsed && validateExplanationSchema(correctedParsed) && checkVerdictConsistency(correctedParsed, input.verdict)) {
        parsed = correctedParsed;
      } else {
        // Step 18: The ML verdict MUST win. If the LLM insists on contradicting Model B, discard it.
        console.warn(`[llmService] LLM explanation remained inconsistent with ML verdict. Discarding explanation.`);
        return {
          explanation: null,
          llm: {
            status: 'inconsistent_rejected',
            reason: 'Generated explanation was inconsistent with verified ML verdict.',
            provider: provider.name,
            model: provider.model
          }
        };
      }
    }

    // Ensure all required fields exist with clean defaults
    const sanitizedExplanation = {
      summary: parsed.summary.trim(),
      verdictExplanation: parsed.verdictExplanation.trim(),
      keyEvidence: (parsed.keyEvidence || []).map(s => String(s).trim()),
      contradictoryEvidence: (parsed.contradictoryEvidence || []).map(s => String(s).trim()),
      uncertainty: String(parsed.uncertainty || '').trim(),
      sourceNotes: (parsed.sourceNotes || []).map(s => String(s).trim())
    };

    return {
      explanation: sanitizedExplanation,
      llm: {
        status: 'generated',
        provider: provider.name,
        model: provider.model,
        generatedAt: new Date().toISOString()
      }
    };
  } catch (error) {
    // Step 14: Catch-all fallback. Never fail the fact-check if LLM fails.
    console.warn(`[llmService] LLM execution error: ${error.message}`);
    return {
      explanation: null,
      llm: {
        status: 'unavailable',
        reason: error.message,
        provider: provider.name,
        model: provider.model
      }
    };
  }
};

module.exports = {
  generateExplanation,
  validateExplanationSchema,
  parseJsonSafely,
  checkVerdictConsistency
};
