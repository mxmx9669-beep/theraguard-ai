export async function onRequest(context) {
  const { request, env } = context;

  const corsHeaders = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
  };

  if (request.method === "OPTIONS") {
    return new Response(null, { status: 204, headers: corsHeaders });
  }

  if (request.method !== "POST") {
    return json({ ok: false, error: "Method not allowed" }, 405, corsHeaders);
  }

  try {
    const body = await request.json();

    if (!env.OPENAI_API_KEY) {
      return json({ ok: false, error: "OPENAI_API_KEY missing" }, 500, corsHeaders);
    }

    if (!env.VECTOR_STORE_ID) {
      return json({ ok: false, error: "VECTOR_STORE_ID missing" }, 500, corsHeaders);
    }

    const action = String(body.action || "").trim();

    if (action === "get_drugs") {
      const drugs = await getDrugsFromVectorStore(env);
      return json({ ok: true, drugs }, 200, corsHeaders);
    }

    if (action === "get_indications") {
      const drug = String(body.drug || "").trim();
      const patient = body.patient || {};
      if (!drug) return json({ ok: false, error: "Drug is required" }, 400, corsHeaders);
      const result = await getIndicationsForDrug(env, drug, patient);
      return json(result, 200, corsHeaders);
    }

    if (action === "get_subindications") {
      const drug = String(body.drug || "").trim();
      const indication = String(body.indication || "").trim();
      const patient = body.patient || {};
      if (!drug || !indication) return json({ ok: false, error: "Drug and indication are required" }, 400, corsHeaders);
      const result = await getSubIndicationsForDrug(env, drug, indication, patient);
      return json(result, 200, corsHeaders);
    }

    if (action === "calculate_review") {
      const result = await calculateMedicationReview(env, body);
      return json(result, 200, corsHeaders);
    }

    if (action === "drug_question") {
      const result = await answerDrugQuestion(env, body);
      return json(result, 200, corsHeaders);
    }

    return json({ ok: false, error: "Unknown action" }, 400, corsHeaders);
  } catch (error) {
    return json({ ok: false, error: "Server error", details: String(error) }, 500, corsHeaders);
  }
}

/* =========================
   Response helper
========================= */
function json(data, status = 200, corsHeaders = {}) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "Content-Type": "application/json", ...corsHeaders },
  });
}

/* =========================
   OpenAI helper
========================= */
async function openaiFetch(env, path, options = {}) {
  const response = await fetch(`https://api.openai.com/v1${path}`, {
    ...options,
    headers: {
      Authorization: `Bearer ${env.OPENAI_API_KEY}`,
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
  });

  const data = await response.json().catch(() => ({}));

  if (!response.ok) {
    throw new Error(`OpenAI request failed (${response.status}): ${JSON.stringify(data)}`);
  }

  return data;
}

/* =========================
   Local calculations
========================= */
function round1(x) { return Math.round(x * 10) / 10; }

function calcBMI(weight, heightCm) {
  if (!weight || !heightCm) return null;
  const h = heightCm / 100;
  return weight / (h * h);
}

function calcIBW(heightCm, sex) {
  if (!heightCm || !sex) return null;
  const htIn = heightCm / 2.54;
  const base = sex === "M" ? 50 : 45.5;
  return base + 2.3 * (htIn - 60);
}

function calcAdjBW(weight, ibw) {
  if (!weight || !ibw) return null;
  return weight <= ibw ? weight : ibw + 0.4 * (weight - ibw);
}

function calcCrCl(age, sex, weightKg, scrUmol) {
  if (!age || !sex || !weightKg || !scrUmol) return null;
  const scrMgDl = scrUmol / 88.4;
  let crcl = ((140 - age) * weightKg) / (72 * scrMgDl);
  if (sex === "F") crcl *= 0.85;
  return crcl;
}

function buildPatientMetrics(patient = {}) {
  const age = Number(patient.age || 0);
  const sex = patient.sex || "";
  const weight = Number(patient.weight || 0);
  const height = Number(patient.height || 0);
  const scr = Number(patient.creatinine || 0);

  const bmi = calcBMI(weight, height);
  const ibw = calcIBW(height, sex);
  const adjbw = calcAdjBW(weight, ibw);
  const crcl_tbw = calcCrCl(age, sex, weight, scr);
  const crcl_adjbw = adjbw ? calcCrCl(age, sex, adjbw, scr) : null;

  let dosing_weight_type = "TBW";
  let dosing_weight = weight;

  if (bmi && bmi >= 30 && adjbw) {
    dosing_weight_type = "AdjBW";
    dosing_weight = adjbw;
  }

  const crcl_dosing = calcCrCl(age, sex, dosing_weight, scr);

  return {
    bmi: bmi ? round1(bmi) : null,
    ibw: ibw ? round1(ibw) : null,
    adjbw: adjbw ? round1(adjbw) : null,
    crcl_tbw: crcl_tbw ? round1(crcl_tbw) : null,
    crcl_adjbw: crcl_adjbw ? round1(crcl_adjbw) : null,
    dosing_weight_type,
    dosing_weight: dosing_weight ? round1(dosing_weight) : null,
    crcl_dosing: crcl_dosing ? round1(crcl_dosing) : null,
  };
}

/* =========================
   SOURCE TYPE CLASSIFICATION
========================= */
/**
 * Classifies a filename into one of:
 * "monograph" | "protocol" | "guideline" | "antibiogram" | "tdm" | "unknown"
 */
function classifySourceType(filename) {
  if (!filename) return "unknown";
  const lower = filename.toLowerCase();

  if (/antibiogram/.test(lower)) return "antibiogram";
  if (/\btdm\b|therapeutic\s*drug\s*monitor/.test(lower)) return "tdm";
  if (/protocol/.test(lower)) return "protocol";
  if (/guideline|guidance|guide/.test(lower)) return "guideline";
  if (/monograph|drug\s*information|drug\s*info/.test(lower)) return "monograph";

  // If none matched, default to monograph (most likely a drug file)
  return "monograph";
}

/* =========================
   FIX 1 — FULL PAGINATION for vector store files
========================= */
/**
 * Fetches ALL files from the vector store using cursor-based pagination.
 * Previously only fetched the first page — now loops until has_more is false.
 */
async function listVectorStoreFiles(env) {
  const out = [];
  let after = null;
  let pageCount = 0;
  const MAX_PAGES = 50; // safety cap

  while (pageCount < MAX_PAGES) {
    pageCount++;

    const queryParams = new URLSearchParams({ limit: "100" });
    if (after) queryParams.set("after", after);

    const vsFiles = await openaiFetch(
      env,
      `/vector_stores/${env.VECTOR_STORE_ID}/files?${queryParams.toString()}`,
      { method: "GET" }
    );

    const items = Array.isArray(vsFiles.data) ? vsFiles.data : [];

    for (const item of items) {
      const fileId = item.id || item.file_id;
      if (!fileId) continue;

      try {
        const fileObj = await openaiFetch(env, `/files/${fileId}`, { method: "GET" });
        const filename = fileObj?.filename || "";
        const displayName = cleanSourceName(filename);
        const sourceType = classifySourceType(filename);

        out.push({
          file_id: fileId,
          filename,
          drug_name: displayName, // kept for backward-compat
          display_name: displayName,
          source_type: sourceType,
        });
      } catch (_) {
        // Skip files that can't be fetched individually
      }
    }

    // Pagination: continue if there are more pages
    if (vsFiles.has_more && items.length > 0) {
      after = items[items.length - 1].id;
    } else {
      break;
    }
  }

  return out;
}

async function getDrugsFromVectorStore(env) {
  const files = await listVectorStoreFiles(env);

  const names = files
    .map(f => f.display_name || f.drug_name)
    .filter(Boolean);

  return [...new Set(names)].sort((a, b) => a.localeCompare(b));
}

/* =========================
   FIX 1 — IMPROVED NAME CLEANING
   Preserves source-type words: guideline, protocol, antibiogram, tdm
========================= */
function cleanSourceName(filename) {
  if (!filename) return "";

  return filename
    // Remove extension
    .replace(/\.[^/.]+$/, "")
    // Replace underscores/dashes with spaces
    .replace(/[_\-]+/g, " ")
    // Collapse multiple spaces
    .replace(/\s+/g, " ")
    // Remove ONLY truly generic/noisy suffixes — NOT source-type words
    .replace(/\bdrug information\b/gi, "")
    .replace(/\bmonograph\b/gi, "")
    // Clean leading/trailing whitespace
    .trim()
    // Title-case
    .replace(/\b\w/g, c => c.toUpperCase());
}

function normalize(s) {
  return String(s || "").toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
}

async function findDrugFiles(env, drug) {
  const files = await listVectorStoreFiles(env);
  const target = normalize(drug);

  const exact = files.filter(f => normalize(f.display_name || f.drug_name) === target);
  if (exact.length) return exact;

  const loose = files.filter(f => {
    const name = normalize(f.display_name || f.drug_name);
    return name.includes(target) || target.includes(name);
  });
  return loose;
}

/* =========================
   Search helpers
========================= */
async function searchVectorStore(env, query, maxNumResults = 8) {
  const data = await openaiFetch(
    env,
    `/vector_stores/${env.VECTOR_STORE_ID}/search`,
    {
      method: "POST",
      body: JSON.stringify({ query, max_num_results: maxNumResults }),
    }
  );
  return Array.isArray(data.data) ? data.data : [];
}

function filterResultsByFileIds(results, fileIds) {
  if (!Array.isArray(fileIds) || !fileIds.length) return results;
  return results.filter(r => fileIds.includes(r.file_id));
}

function buildFileMap(files) {
  const map = new Map();
  for (const f of files) map.set(f.file_id, f);
  return map;
}

function parsePageFromText(text) {
  const s = String(text || "");
  const match =
    s.match(/\bpage\s*[:\-]?\s*(\d{1,4})\b/i) ||
    s.match(/\bp\.\s*(\d{1,4})\b/i) ||
    s.match(/\bp\s*(\d{1,4})\b/i);
  return match ? Number(match[1]) : null;
}

function mapResultsToEvidence(results, fileMap) {
  const evidence = [];
  for (const item of results) {
    const textChunk = Array.isArray(item.content)
      ? item.content.find(c => c.type === "text" && c.text)
      : null;
    if (!textChunk?.text) continue;
    const fileInfo = fileMap.get(item.file_id);
    evidence.push({
      file_id: item.file_id || null,
      source: fileInfo?.filename || fileInfo?.display_name || "Unknown source",
      source_type: fileInfo?.source_type || "unknown",
      text: textChunk.text,
      page: parsePageFromText(textChunk.text),
    });
  }
  return evidence;
}

/* =========================
   Responses API helper
========================= */
async function runModel(env, systemPrompt, userPrompt) {
  const model = env.MODEL || "gpt-4.1-mini";

  const data = await openaiFetch(env, `/responses`, {
    method: "POST",
    body: JSON.stringify({
      model,
      input: [
        { role: "system", content: [{ type: "input_text", text: systemPrompt }] },
        { role: "user", content: [{ type: "input_text", text: userPrompt }] },
      ],
    }),
  });

  return extractOutputText(data);
}

function extractOutputText(data) {
  let text = "";
  if (!Array.isArray(data.output)) return text;
  for (const item of data.output) {
    if (item.type !== "message" || !Array.isArray(item.content)) continue;
    for (const part of item.content) {
      if (part.type === "output_text" && part.text) text += part.text;
    }
  }
  return text.trim();
}

function parseJsonFromText(text) {
  try { return JSON.parse(text); } catch (_) {}

  const fence = text.match(/```json\s*([\s\S]*?)```/i) || text.match(/```([\s\S]*?)```/i);
  if (fence) { try { return JSON.parse(fence[1].trim()); } catch (_) {} }

  const first = text.indexOf("{");
  const last = text.lastIndexOf("}");
  if (first !== -1 && last !== -1 && last > first) {
    try { return JSON.parse(text.slice(first, last + 1)); } catch (_) {}
  }

  return null;
}

/* =========================
   FIX 2 — VALIDATION FILTERS for indications/sub-pathways
========================= */
const DOSAGE_PATTERN = /\b(\d+\.?\d*\s*(mg|mcg|g|ml|mmol|units?|iu|meq)\b|\bonce\b|\btwice\b|\bthrice\b|\bq\d+h\b|\bbid\b|\btid\b|\bqid\b|\bqd\b|\boad\b|\bprn\b)/i;
const ROUTE_PATTERN = /\b(intravenous|iv\b|intramuscular|im\b|subcutaneous|sc\b|oral|po\b|infusion|bolus|topical|inhaled|nebulized)\b/i;
const MONITORING_PATTERN = /\b(monitor|level|trough|peak|serum|plasma|concentration|renal function|creatinine|electrolyte|culture|sensitivity)\b/i;
const SENTENCE_PATTERN = /[.!?]\s+[A-Z]|^\s*\w.*[,;]\s+\w.*[,;]/;

function isValidIndicationLabel(label) {
  if (!label || typeof label !== "string") return false;

  const s = label.trim();

  // Too long (>60 chars is likely a sentence fragment)
  if (s.length > 60) return false;

  // Too short
  if (s.length < 3) return false;

  // Contains dosage/frequency language
  if (DOSAGE_PATTERN.test(s)) return false;

  // Contains route of administration language
  if (ROUTE_PATTERN.test(s)) return false;

  // Contains monitoring language
  if (MONITORING_PATTERN.test(s)) return false;

  // Looks like a full sentence (contains internal period + capital, or multiple commas)
  if (SENTENCE_PATTERN.test(s)) return false;

  // Starts with a lowercase word indicating a fragment
  if (/^(with|and|or|the|a |an |for |in |of |to |is |are |by )/i.test(s)) return false;

  return true;
}

function deduplicateLabels(items) {
  const seen = new Set();
  return items.filter(item => {
    const key = normalize(item.label || item.value || item);
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

/* =========================
   FIX 2 — SOURCE-AWARE INDICATIONS
========================= */
async function getIndicationsForDrug(env, drug, patient) {
  const metrics = buildPatientMetrics(patient);
  const drugFiles = await findDrugFiles(env, drug);
  const fileIds = drugFiles.map(f => f.file_id);
  const fileMap = buildFileMap(drugFiles);

  // Determine primary source type
  const primarySourceType = drugFiles[0]?.source_type || "monograph";

  // Antibiogram and TDM docs should not produce fake indications
  if (primarySourceType === "antibiogram") {
    return {
      ok: true,
      indications: [],
      source_type: "antibiogram",
      message: "Antibiogram source — indication selection not applicable.",
    };
  }

  let results = await searchVectorStore(
    env,
    `${drug} indications approved uses clinical pathways treatment`,
    10
  );

  results = filterResultsByFileIds(results, fileIds);
  const evidence = mapResultsToEvidence(results, fileMap).slice(0, 8);

  // Source-specific system prompt
  let extractionInstructions = "";
  if (primarySourceType === "monograph") {
    extractionInstructions = `
This is a DRUG MONOGRAPH. Extract only approved clinical indications.
- Each indication must be a short clinical label (e.g., "Community-Acquired Pneumonia", "Urinary Tract Infection")
- Do NOT include dosing, routes, frequencies, or monitoring text as indications
- Do NOT include full sentences
- Maximum 8 indications
- Only include indications that are clearly named in the evidence
`;
  } else if (primarySourceType === "protocol" || primarySourceType === "guideline") {
    extractionInstructions = `
This is a CLINICAL PROTOCOL or GUIDELINE. Extract named treatment pathways or clinical decision branches.
- Each pathway should be a short selectable label (e.g., "Empiric HAP/VAP Therapy", "Step-Down to Oral")
- Do NOT include dosing phrases or monitoring instructions as pathway names
- Do NOT include full sentences
- Maximum 6 pathways
`;
  } else if (primarySourceType === "tdm") {
    extractionInstructions = `
This is a TDM (Therapeutic Drug Monitoring) document. Extract only named clinical scenarios where TDM applies.
- Examples: "Initial Dosing", "Level Adjustment", "Toxicity Management"
- Do NOT include monitoring parameters as indication labels
- Maximum 4 options
`;
  } else {
    extractionInstructions = `
Extract only clearly named clinical indications or usage scenarios.
- Short, selectable labels only
- No dosing or monitoring text
- Maximum 6 options
`;
  }

  const systemPrompt = `
You extract clinical indications from medical source evidence.
Return ONLY valid JSON. No markdown. No extra text.
${extractionInstructions}
STRICT RULES:
- Labels must be 3–60 characters
- No dosage numbers (mg, mcg, g, etc.)
- No frequency words (once daily, BID, TID, q8h, etc.)
- No route words (IV, PO, infusion, bolus, etc.)
- No monitoring words (trough, level, serum, monitor, etc.)
- No full sentences
- No duplicates
If no valid indications found, return an empty array.
`;

  const userPrompt = `
Drug/Source: ${drug}
Source type: ${primarySourceType}

Patient context:
${JSON.stringify(patient, null, 2)}

Calculated metrics:
${JSON.stringify(metrics, null, 2)}

Evidence:
${evidence.map((e, i) => `SOURCE ${i + 1} (${e.source}${e.page ? `, Page ${e.page}` : ""}):\n${e.text}`).join("\n\n")}

Return exactly:
{
  "ok": true,
  "indications": [
    { "value": "...", "label": "..." }
  ]
}
`;

  const text = await runModel(env, systemPrompt, userPrompt);
  const parsed = parseJsonFromText(text);

  if (parsed?.ok && Array.isArray(parsed.indications)) {
    // Post-processing: filter out invalid labels
    const cleaned = deduplicateLabels(
      parsed.indications.filter(ind => isValidIndicationLabel(ind.label || ind.value))
    );
    return { ok: true, indications: cleaned, source_type: primarySourceType };
  }

  return { ok: true, indications: [], source_type: primarySourceType };
}

/* =========================
   FIX 2 — SOURCE-AWARE SUB-PATHWAYS (STRICT)
========================= */
async function getSubIndicationsForDrug(env, drug, indication, patient) {
  const metrics = buildPatientMetrics(patient);
  const drugFiles = await findDrugFiles(env, drug);
  const fileIds = drugFiles.map(f => f.file_id);
  const fileMap = buildFileMap(drugFiles);

  const primarySourceType = drugFiles[0]?.source_type || "monograph";

  // Antibiogram: never generate sub-pathways
  if (primarySourceType === "antibiogram") {
    return { ok: true, subindications: [], source_type: "antibiogram" };
  }

  let results = await searchVectorStore(
    env,
    `${drug} ${indication} subtype subgroup branch severity classification criteria`,
    10
  );

  results = filterResultsByFileIds(results, fileIds);
  const evidence = mapResultsToEvidence(results, fileMap).slice(0, 8);

  const systemPrompt = `
You extract optional sub-pathway branches that exist CLEARLY under a chosen indication.
Return ONLY valid JSON. No markdown.

CRITICAL RULES:
- Return sub-pathways ONLY if the evidence explicitly shows a hierarchy or branch under this indication
- Sub-pathways must be short named categories (e.g., "Mild-Moderate", "Severe", "MRSA Coverage", "With Renal Adjustment")
- Do NOT invent sub-pathways from dosing instructions
- Do NOT include dosage numbers, frequencies, routes, or monitoring as sub-pathway labels
- Do NOT include full sentences
- Labels must be 3–50 characters
- If no clear hierarchical subdivision exists in the evidence, return an empty array
- For TDM sources: only return sub-pathways if there are clearly distinct monitoring scenarios
- For monographs: only return sub-pathways if the indication truly has documented sub-categories
`;

  const userPrompt = `
Drug/Source: ${drug}
Selected indication: ${indication}
Source type: ${primarySourceType}

Patient context:
${JSON.stringify(patient, null, 2)}

Calculated metrics:
${JSON.stringify(metrics, null, 2)}

Evidence:
${evidence.map((e, i) => `SOURCE ${i + 1} (${e.source}${e.page ? `, Page ${e.page}` : ""}):\n${e.text}`).join("\n\n")}

Return exactly:
{
  "ok": true,
  "subindications": [
    { "value": "...", "label": "..." }
  ]
}

If no real sub-branches exist in the evidence, return: { "ok": true, "subindications": [] }
`;

  const text = await runModel(env, systemPrompt, userPrompt);
  const parsed = parseJsonFromText(text);

  if (parsed?.ok && Array.isArray(parsed.subindications)) {
    const cleaned = deduplicateLabels(
      parsed.subindications.filter(sub => isValidIndicationLabel(sub.label || sub.value))
    );
    return { ok: true, subindications: cleaned, source_type: primarySourceType };
  }

  return { ok: true, subindications: [], source_type: primarySourceType };
}

/* =========================
   Regimen generation (unchanged logic, updated findDrugFiles)
========================= */
async function calculateMedicationReview(env, body) {
  const patient = body.patient || {};
  const medications = Array.isArray(body.medications) ? body.medications : [];
  const metrics = buildPatientMetrics(patient);

  if (!medications.length) return { ok: false, error: "At least one medication is required" };

  const review = [];
  const allEvidence = [];

  for (const med of medications) {
    const drug = String(med.drug || "").trim();
    const indication = String(med.indication || "").trim();
    const subIndication = String(med.sub_indication || "").trim();
    if (!drug || !indication) continue;

    const drugFiles = await findDrugFiles(env, drug);
    const fileIds = drugFiles.map(f => f.file_id);
    const fileMap = buildFileMap(drugFiles);

    let results = await searchVectorStore(
      env,
      [drug, indication, subIndication, "dose", "frequency", "duration", "administration",
       "route", "infusion", "bolus", "loading dose", "maintenance", "titration", "monitoring",
       "renal adjustment", "dialysis", patient.krt || "", patient.aki || "",
       patient.route_preference || "", patient.infusion || "", patient.severity || ""]
        .filter(Boolean).join(" "),
      12
    );

    results = filterResultsByFileIds(results, fileIds);
    const evidence = mapResultsToEvidence(results, fileMap).slice(0, 9);
    allEvidence.push(...evidence);

    const systemPrompt = `
You are a clinical dosing assistant.
Use ONLY the supplied evidence and patient context.
Return ONLY valid JSON. No markdown.
Return short structured regimen output.
Do not include patient summary in the output.
Do not invent unsupported recommendations.
`;

    const userPrompt = `
Patient:
${JSON.stringify(patient, null, 2)}

Calculated metrics:
${JSON.stringify(metrics, null, 2)}

Selected medication:
${JSON.stringify({ drug, indication, sub_indication: subIndication }, null, 2)}

Evidence:
${evidence.map((e, i) => `SOURCE ${i + 1} (${e.source}${e.page ? `, Page ${e.page}` : ""}):\n${e.text}`).join("\n\n")}

Return exactly:
{
  "drug": "${drug}",
  "indication": "${indication}",
  "sub_indication": "${subIndication}",
  "recommended_dose": "...",
  "recommended_frequency": "...",
  "recommended_duration": "...",
  "administration": "...",
  "monitoring": ["..."],
  "therapy_strategy": "...",
  "evidence": ["..."]
}
`;

    const text = await runModel(env, systemPrompt, userPrompt);
    const parsed = parseJsonFromText(text);

    review.push(
      parsed?.drug
        ? parsed
        : { drug, indication, sub_indication: subIndication, recommended_dose: "",
            recommended_frequency: "", recommended_duration: "", administration: "",
            monitoring: [], therapy_strategy: "", evidence: [] }
    );
  }

  const ddi = await detectDDI(env, patient, medications, metrics, allEvidence);
  const globalWarnings = await buildGlobalWarnings(env, patient, medications, metrics, allEvidence);

  return { ok: true, medication_review: review, ddi_summary: ddi, global_warnings: globalWarnings };
}

/* =========================
   DDI (unchanged)
========================= */
async function detectDDI(env, patient, medications, metrics, evidence) {
  if (!Array.isArray(medications) || medications.length < 2) {
    return { has_ddi: false, interactions: [] };
  }

  const medNames = medications.map(m => String(m.drug || "").trim()).filter(Boolean);
  const pairs = [];
  for (let i = 0; i < medNames.length; i++) {
    for (let j = i + 1; j < medNames.length; j++) {
      pairs.push(`${medNames[i]} + ${medNames[j]}`);
    }
  }

  const systemPrompt = `
You are a structured drug interaction assistant.
Use ONLY the supplied medication list, evidence, and patient context.
Return ONLY valid JSON. No markdown.
Classify clinically meaningful interactions using rating A, B, C, D, or X.
If no meaningful interaction is supported, return has_ddi false and an empty interactions array.
Keep management short.
`;

  const userPrompt = `
Patient:
${JSON.stringify(patient, null, 2)}

Calculated metrics:
${JSON.stringify(metrics, null, 2)}

Medication list:
${JSON.stringify(medications, null, 2)}

Pairs:
${JSON.stringify(pairs, null, 2)}

Evidence:
${evidence.slice(0, 16).map((e, i) => `SOURCE ${i + 1} (${e.source}${e.page ? `, Page ${e.page}` : ""}):\n${e.text}`).join("\n\n")}

Return exactly:
{
  "has_ddi": true,
  "interactions": [
    { "pair": "...", "rating": "A", "management": "..." }
  ]
}
`;

  const text = await runModel(env, systemPrompt, userPrompt);
  const parsed = parseJsonFromText(text);

  if (typeof parsed?.has_ddi === "boolean" && Array.isArray(parsed?.interactions)) return parsed;
  return { has_ddi: false, interactions: [] };
}

/* =========================
   Global notes (unchanged)
========================= */
async function buildGlobalWarnings(env, patient, medications, metrics, evidence) {
  const systemPrompt = `
You are a medication safety assistant.
Use only supplied evidence and medication list.
Return ONLY valid JSON. No markdown.
Return only short additional notes that are clinically useful.
`;

  const userPrompt = `
Patient:
${JSON.stringify(patient, null, 2)}

Calculated metrics:
${JSON.stringify(metrics, null, 2)}

Medication list:
${JSON.stringify(medications, null, 2)}

Evidence:
${evidence.slice(0, 12).map((e, i) => `SOURCE ${i + 1} (${e.source}${e.page ? `, Page ${e.page}` : ""}):\n${e.text}`).join("\n\n")}

Return exactly:
{ "global_warnings": ["..."] }
`;

  const text = await runModel(env, systemPrompt, userPrompt);
  const parsed = parseJsonFromText(text);
  return Array.isArray(parsed?.global_warnings) ? parsed.global_warnings : [];
}

/* =========================
   Drug Q&A (unchanged)
========================= */
async function answerDrugQuestion(env, body) {
  const drug = String(body.drug || "").trim();
  const question = String(body.question || "").trim();
  const patient = body.patient || {};
  const metrics = buildPatientMetrics(patient);

  if (!drug) return { ok: false, error: "Drug is required" };
  if (!question) return { ok: false, error: "Question is required" };

  const drugFiles = await findDrugFiles(env, drug);
  if (!drugFiles.length) return { ok: false, error: `No vector-store file matched for ${drug}` };

  const fileIds = drugFiles.map(f => f.file_id);
  const fileMap = buildFileMap(drugFiles);

  let results = await searchVectorStore(
    env,
    [drug, question, patient.krt || "", patient.aki || "", patient.severity || "",
     patient.route_preference || "", "renal adjustment", "dialysis", "monitoring"]
      .filter(Boolean).join(" "),
    12
  );

  results = filterResultsByFileIds(results, fileIds);
  const evidence = mapResultsToEvidence(results, fileMap).slice(0, 10);

  const systemPrompt = `
You are a drug monograph Q&A assistant.
Answer ONLY from the supplied evidence for the selected drug.
Return ONLY valid JSON. No markdown.
The answer must be short.
Provide 2 to 3 evidence quotes directly supporting the answer.
Use patient context only to prioritize relevant evidence.
`;

  const userPrompt = `
Selected drug: ${drug}

Patient:
${JSON.stringify(patient, null, 2)}

Calculated metrics:
${JSON.stringify(metrics, null, 2)}

Question:
${question}

Evidence from selected drug file(s):
${evidence.map((e, i) => `SOURCE ${i + 1} (${e.source}${e.page ? `, Page ${e.page}` : ""}):\n${e.text}`).join("\n\n")}

Return exactly:
{
  "ok": true,
  "drug": "${drug}",
  "question": "${question}",
  "answer": "...",
  "evidence": [
    { "quote": "...", "source": "...", "page": null }
  ]
}
`;

  const text = await runModel(env, systemPrompt, userPrompt);
  const parsed = parseJsonFromText(text);

  if (parsed?.ok && Array.isArray(parsed.evidence)) return parsed;

  return {
    ok: true, drug, question,
    answer: "Unable to generate structured answer from the selected drug file.",
    evidence: evidence.slice(0, 3).map(e => ({
      quote: e.text.slice(0, 300), source: e.source, page: e.page || null,
    })),
  };
}
