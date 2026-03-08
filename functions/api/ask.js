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

    const action = (body.action || "").trim();

    if (action === "get_drugs") {
      const drugs = await getDrugsFromVectorStore(env);
      return json({ ok: true, drugs }, 200, corsHeaders);
    }

    if (action === "get_indications") {
      const drug = (body.drug || "").trim();
      const patient = body.patient || {};

      if (!drug) {
        return json({ ok: false, error: "Drug is required" }, 400, corsHeaders);
      }

      const result = await getIndicationsForDrug(env, drug, patient);
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
    return json(
      {
        ok: false,
        error: "Server error",
        details: String(error),
      },
      500,
      corsHeaders
    );
  }
}

/* =========================
   Response helper
========================= */
function json(data, status = 200, corsHeaders = {}) {
  return new Response(JSON.stringify(data), {
    status,
    headers: {
      "Content-Type": "application/json",
      ...corsHeaders,
    },
  });
}

/* =========================
   OpenAI HTTP helper
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
function round1(x) {
  return Math.round(x * 10) / 10;
}

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
   Vector store file catalog
========================= */
async function listVectorStoreFiles(env) {
  const vsFiles = await openaiFetch(
    env,
    `/vector_stores/${env.VECTOR_STORE_ID}/files`,
    { method: "GET" }
  );

  const items = Array.isArray(vsFiles.data) ? vsFiles.data : [];
  const out = [];

  for (const item of items) {
    const fileId = item.id || item.file_id;
    if (!fileId) continue;

    try {
      const fileObj = await openaiFetch(env, `/files/${fileId}`, { method: "GET" });
      const filename = fileObj?.filename || "";
      const drugName = cleanDrugName(filename);

      out.push({
        file_id: fileId,
        filename,
        drug_name: drugName,
      });
    } catch (_) {
      // ignore one-file failures
    }
  }

  return out;
}

async function getDrugsFromVectorStore(env) {
  const files = await listVectorStoreFiles(env);
  const names = files
    .map(f => f.drug_name)
    .filter(Boolean);

  return [...new Set(names)].sort((a, b) => a.localeCompare(b));
}

function cleanDrugName(filename) {
  if (!filename) return "";

  return filename
    .replace(/\.[^/.]+$/, "")
    .replace(/_/g, " ")
    .replace(/\s+/g, " ")
    .replace(/\bdrug information\b/gi, "")
    .replace(/\bmonograph\b/gi, "")
    .replace(/\bguideline\b/gi, "")
    .replace(/\bprotocol\b/gi, "Protocol")
    .trim();
}

function normalize(s) {
  return String(s || "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
}

async function findDrugFiles(env, drug) {
  const files = await listVectorStoreFiles(env);
  const target = normalize(drug);

  const exact = files.filter(f => normalize(f.drug_name) === target);
  if (exact.length) return exact;

  const loose = files.filter(f => normalize(f.drug_name).includes(target) || target.includes(normalize(f.drug_name)));
  return loose;
}

/* =========================
   Vector search helpers
========================= */
async function searchVectorStore(env, query, maxNumResults = 8) {
  const data = await openaiFetch(
    env,
    `/vector_stores/${env.VECTOR_STORE_ID}/search`,
    {
      method: "POST",
      body: JSON.stringify({
        query,
        max_num_results: maxNumResults,
      }),
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

    evidence.push({
      file_id: item.file_id || null,
      source: fileMap.get(item.file_id)?.filename || fileMap.get(item.file_id)?.drug_name || "Unknown source",
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
        {
          role: "system",
          content: [{ type: "input_text", text: systemPrompt }],
        },
        {
          role: "user",
          content: [{ type: "input_text", text: userPrompt }],
        },
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
      if (part.type === "output_text" && part.text) {
        text += part.text;
      }
    }
  }

  return text.trim();
}

function parseJsonFromText(text) {
  try {
    return JSON.parse(text);
  } catch (_) {}

  const fence =
    text.match(/```json\s*([\s\S]*?)```/i) ||
    text.match(/```([\s\S]*?)```/i);

  if (fence) {
    try {
      return JSON.parse(fence[1].trim());
    } catch (_) {}
  }

  const first = text.indexOf("{");
  const last = text.lastIndexOf("}");
  if (first !== -1 && last !== -1 && last > first) {
    try {
      return JSON.parse(text.slice(first, last + 1));
    } catch (_) {}
  }

  return null;
}

/* =========================
   Action: get_indications
========================= */
async function getIndicationsForDrug(env, drug, patient) {
  const metrics = buildPatientMetrics(patient);
  const drugFiles = await findDrugFiles(env, drug);
  const fileIds = drugFiles.map(f => f.file_id);
  const fileMap = buildFileMap(drugFiles);

  let results = await searchVectorStore(
    env,
    `${drug} indications clinical pathways approved uses dosing pathways renal dialysis dosing`
  );

  results = filterResultsByFileIds(results, fileIds);
  const evidence = mapResultsToEvidence(results, fileMap).slice(0, 8);

  const systemPrompt = `
You extract indication/pathway options from drug monograph evidence.
Return ONLY valid JSON.
No markdown.
Split clinically distinct branches into separate selectable options.
If patient age, renal function, obesity, or renal replacement therapy clearly changes the branch, prefer patient-relevant branches.
`;

  const userPrompt = `
Drug: ${drug}

Patient:
${JSON.stringify(patient, null, 2)}

Calculated metrics:
${JSON.stringify(metrics, null, 2)}

Evidence:
${evidence.map((e, i) => `SOURCE ${i + 1} (${e.source}):\n${e.text}`).join("\n\n")}

Return exactly:
{
  "ok": true,
  "indications": ["..."],
  "note": "...",
  "drug_note": "..."
}
`;

  const text = await runModel(env, systemPrompt, userPrompt);
  const parsed = parseJsonFromText(text);

  if (parsed?.ok && Array.isArray(parsed.indications)) {
    return parsed;
  }

  return {
    ok: true,
    indications: [],
    note: "No structured indication options extracted",
    drug_note: `Could not extract pathways for ${drug}`,
  };
}

/* =========================
   Action: calculate_review
========================= */
async function calculateMedicationReview(env, body) {
  const patient = body.patient || {};
  const medications = Array.isArray(body.medications) ? body.medications : [];
  const metrics = buildPatientMetrics(patient);

  if (!medications.length) {
    return { ok: false, error: "At least one medication is required" };
  }

  const review = [];
  const allEvidence = [];
  const allDrugNames = [];

  for (const med of medications) {
    const drug = String(med.drug || "").trim();
    const indication = String(med.indication || "").trim();
    const currentDose = String(med.current_dose || "").trim();
    const frequency = String(med.frequency || "").trim();
    const route = String(med.route || "").trim();

    if (!drug) continue;
    allDrugNames.push(drug);

    const drugFiles = await findDrugFiles(env, drug);
    const fileIds = drugFiles.map(f => f.file_id);
    const fileMap = buildFileMap(drugFiles);

    let results = await searchVectorStore(
      env,
      [
        drug,
        indication,
        currentDose,
        frequency,
        route,
        "dose",
        "loading dose",
        "maintenance dose",
        "duration",
        "therapy strategy",
        "titration",
        "renal adjustment",
        "dialysis",
        patient.krt || "",
        patient.aki || "",
        patient.severity || "",
        patient.route_preference || "",
        patient.infusion || "",
        "monitoring",
        "warnings"
      ].filter(Boolean).join(" "),
      10
    );

    results = filterResultsByFileIds(results, fileIds);
    const evidence = mapResultsToEvidence(results, fileMap).slice(0, 8);
    allEvidence.push(...evidence);

    const systemPrompt = `
You are a clinical pharmacotherapy review assistant.
Use ONLY the supplied evidence and patient metrics.
Return ONLY valid JSON.
No markdown.
Be concise and practical.
Do not invent recommendations outside the evidence.
Separate regimen into structured fields whenever possible.
`;

    const userPrompt = `
Patient:
${JSON.stringify(patient, null, 2)}

Calculated patient metrics:
${JSON.stringify(metrics, null, 2)}

Medication to review:
${JSON.stringify({
  drug,
  indication,
  current_dose: currentDose,
  frequency,
  route,
}, null, 2)}

Evidence:
${evidence.map((e, i) => `SOURCE ${i + 1} (${e.source}${e.page ? `, Page ${e.page}` : ""}):\n${e.text}`).join("\n\n")}

Return exactly:
{
  "drug": "${drug}",
  "assessment": "...",
  "recommended_regimen": "...",
  "recommended_dose": "...",
  "recommended_frequency": "...",
  "recommended_duration": "...",
  "therapy_strategy": "...",
  "monitoring": ["..."],
  "warnings": ["..."],
  "evidence": ["..."],
  "references": ["SOURCE 1", "SOURCE 2"]
}
`;

    const text = await runModel(env, systemPrompt, userPrompt);
    const parsed = parseJsonFromText(text);

    review.push(
      parsed && parsed.drug
        ? parsed
        : {
            drug,
            assessment: "Unable to generate structured review",
            recommended_regimen: "Review source manually",
            recommended_dose: "",
            recommended_frequency: "",
            recommended_duration: "",
            therapy_strategy: "",
            monitoring: [],
            warnings: [],
            evidence: [],
            references: [],
          }
    );
  }

  const globalWarnings = await buildGlobalWarnings(env, patient, medications, metrics, allEvidence);
  const ddi = await detectDDI(env, patient, medications, metrics, allEvidence);

  return {
    ok: true,
    patient_summary: metrics,
    medication_review: review,
    global_warnings: globalWarnings,
    ddi_summary: ddi,
  };
}

async function buildGlobalWarnings(env, patient, medications, metrics, evidence) {
  const systemPrompt = `
You are a medication safety assistant.
Use only supplied evidence and medication list.
Return ONLY valid JSON.
No markdown.
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
{
  "global_warnings": ["..."]
}

Rules:
- Focus on interaction risks, duplicate therapy, additive toxicity, renal safety, bleeding risk, QT risk, nephrotoxicity, overlapping spectrum
- If no strong global warning is supported, return an empty array
`;

  const text = await runModel(env, systemPrompt, userPrompt);
  const parsed = parseJsonFromText(text);

  return Array.isArray(parsed?.global_warnings) ? parsed.global_warnings : [];
}

async function detectDDI(env, patient, medications, metrics, evidence) {
  if (!Array.isArray(medications) || medications.length < 2) {
    return {
      has_ddi: false,
      interactions: []
    };
  }

  const medNames = medications
    .map(m => String(m.drug || "").trim())
    .filter(Boolean);

  const pairText = [];
  for (let i = 0; i < medNames.length; i++) {
    for (let j = i + 1; j < medNames.length; j++) {
      pairText.push(`${medNames[i]} + ${medNames[j]}`);
    }
  }

  const systemPrompt = `
You are a structured drug-drug interaction assistant.
Use ONLY the supplied evidence, medication list, and patient context.
Return ONLY valid JSON.
No markdown.
Only report interactions supported by supplied evidence or strongly inferable from supplied evidence.
`;

  const userPrompt = `
Patient:
${JSON.stringify(patient, null, 2)}

Calculated metrics:
${JSON.stringify(metrics, null, 2)}

Medications:
${JSON.stringify(medications, null, 2)}

Medication pairs:
${JSON.stringify(pairText, null, 2)}

Evidence:
${evidence.slice(0, 16).map((e, i) => `SOURCE ${i + 1} (${e.source}${e.page ? `, Page ${e.page}` : ""}):\n${e.text}`).join("\n\n")}

Return exactly:
{
  "has_ddi": true,
  "interactions": [
    {
      "pair": "...",
      "exists": "Yes",
      "mechanism": "...",
      "clinical_risk": "...",
      "management": "..."
    }
  ]
}

Rules:
- If no clinically meaningful DDI is supported, return:
{
  "has_ddi": false,
  "interactions": []
}
- Keep management short and practical
- Prefer renal toxicity, bleeding, QT, CNS depression, duplicate anticoagulation, serotonin toxicity, electrolyte-related risk, additive hypotension, overlapping spectrum, and TDM-relevant interactions
`;

  const text = await runModel(env, systemPrompt, userPrompt);
  const parsed = parseJsonFromText(text);

  if (typeof parsed?.has_ddi === "boolean" && Array.isArray(parsed?.interactions)) {
    return parsed;
  }

  return {
    has_ddi: false,
    interactions: []
  };
}

/* =========================
   Action: drug_question
========================= */
async function answerDrugQuestion(env, body) {
  const drug = String(body.drug || "").trim();
  const question = String(body.question || "").trim();
  const patient = body.patient || {};
  const metrics = buildPatientMetrics(patient);

  if (!drug) return { ok: false, error: "Drug is required" };
  if (!question) return { ok: false, error: "Question is required" };

  const drugFiles = await findDrugFiles(env, drug);
  if (!drugFiles.length) {
    return { ok: false, error: `No vector-store file matched for ${drug}` };
  }

  const fileIds = drugFiles.map(f => f.file_id);
  const fileMap = buildFileMap(drugFiles);

  let results = await searchVectorStore(
    env,
    [
      drug,
      question,
      patient.krt || "",
      patient.aki || "",
      patient.severity || "",
      patient.route_preference || "",
      "renal adjustment",
      "dialysis",
      "CRRT",
      "HD",
      "PIRRT",
      "monitoring"
    ].filter(Boolean).join(" "),
    12
  );

  results = filterResultsByFileIds(results, fileIds);
  const evidence = mapResultsToEvidence(results, fileMap).slice(0, 10);

  const systemPrompt = `
You are a drug monograph Q&A assistant.
Answer ONLY from the supplied evidence for the selected drug.
Return ONLY valid JSON.
No markdown.
The answer must be short.
Provide 2 to 3 evidence quotes directly supporting the answer.
Use patient context only to prioritize relevant evidence, not to invent unsupported advice.
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
    {
      "quote": "...",
      "source": "...",
      "page": null
    }
  ]
}

Rules:
- answer should be concise
- evidence must contain 2 to 3 items if possible
- quote must be short verbatim or near-verbatim from provided evidence
- if page number is unknown, use null
- do not invent page numbers
`;

  const text = await runModel(env, systemPrompt, userPrompt);
  const parsed = parseJsonFromText(text);

  if (parsed?.ok && Array.isArray(parsed.evidence)) {
    return parsed;
  }

  return {
    ok: true,
    drug,
    question,
    answer: "Unable to generate structured answer from the selected drug file.",
    evidence: evidence.slice(0, 3).map(e => ({
      quote: e.text.slice(0, 300),
      source: e.source,
      page: e.page || null,
    })),
  };
}
