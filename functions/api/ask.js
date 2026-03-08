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
      return json(
        {
          ok: true,
          indications: result.indications || [],
          note: result.note || "",
          drug_note: result.drug_note || ""
        },
        200,
        corsHeaders
      );
    }

    if (action === "calculate") {
      const result = await calculateRecommendation(env, body);
      return json(result, 200, corsHeaders);
    }

    return json({ ok: false, error: "Unknown action" }, 400, corsHeaders);
  } catch (error) {
    return json(
      {
        ok: false,
        error: "Server error",
        details: String(error)
      },
      500,
      corsHeaders
    );
  }
}

function json(data, status = 200, corsHeaders = {}) {
  return new Response(JSON.stringify(data), {
    status,
    headers: {
      "Content-Type": "application/json",
      ...corsHeaders
    }
  });
}

async function openaiFetch(env, path, options = {}) {
  const response = await fetch(`https://api.openai.com/v1${path}`, {
    ...options,
    headers: {
      Authorization: `Bearer ${env.OPENAI_API_KEY}`,
      "Content-Type": "application/json",
      ...(options.headers || {})
    }
  });

  const data = await response.json().catch(() => ({}));

  if (!response.ok) {
    throw new Error(`OpenAI error ${response.status}: ${JSON.stringify(data)}`);
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

function buildPatientMetrics(patient) {
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

  return {
    bmi: bmi ? round1(bmi) : null,
    ibw: ibw ? round1(ibw) : null,
    adjbw: adjbw ? round1(adjbw) : null,
    crcl_tbw: crcl_tbw ? round1(crcl_tbw) : null,
    crcl_adjbw: crcl_adjbw ? round1(crcl_adjbw) : null
  };
}

/* =========================
   Drug list from vector store files
========================= */
async function getDrugsFromVectorStore(env) {
  const vsFiles = await openaiFetch(
    env,
    `/vector_stores/${env.VECTOR_STORE_ID}/files`,
    { method: "GET" }
  );

  const items = Array.isArray(vsFiles.data) ? vsFiles.data : [];
  const names = [];

  for (const item of items) {
    const fileId = item.id || item.file_id;
    if (!fileId) continue;

    try {
      const fileObj = await openaiFetch(env, `/files/${fileId}`, { method: "GET" });
      if (fileObj?.filename) {
        const clean = cleanDrugName(fileObj.filename);
        if (clean) names.push(clean);
      }
    } catch (_) {}
  }

  return [...new Set(names)].sort((a, b) => a.localeCompare(b));
}

function cleanDrugName(filename) {
  return filename
    .replace(/\.[^/.]+$/, "")
    .replace(/_/g, " ")
    .replace(/\s+/g, " ")
    .replace(/\bdrug information\b/gi, "")
    .replace(/\bmonograph\b/gi, "")
    .replace(/\bguideline\b/gi, "")
    .trim();
}

/* =========================
   Vector search
========================= */
async function searchVectorStore(env, query, maxNumResults = 8) {
  const data = await openaiFetch(
    env,
    `/vector_stores/${env.VECTOR_STORE_ID}/search`,
    {
      method: "POST",
      body: JSON.stringify({
        query,
        max_num_results: maxNumResults
      })
    }
  );

  return Array.isArray(data.data) ? data.data : [];
}

function extractSearchTexts(results) {
  const out = [];

  for (const item of results) {
    if (!Array.isArray(item.content)) continue;
    for (const c of item.content) {
      if (c.type === "text" && c.text) out.push(c.text);
    }
  }

  return out;
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
          content: [{ type: "input_text", text: systemPrompt }]
        },
        {
          role: "user",
          content: [{ type: "input_text", text: userPrompt }]
        }
      ]
    })
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

  const match =
    text.match(/```json\s*([\s\S]*?)```/i) ||
    text.match(/```([\s\S]*?)```/i);

  if (match) {
    try {
      return JSON.parse(match[1].trim());
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
   Patient-aware indications
========================= */
async function getIndicationsForDrug(env, drug, patient) {
  const metrics = buildPatientMetrics(patient);

  const searchQuery = [
    drug,
    "indications",
    "clinical pathways",
    "dosing pathways",
    patient.krt || "",
    patient.aki || "",
    patient.severity || ""
  ].filter(Boolean).join(" ");

  const results = await searchVectorStore(env, searchQuery, 10);
  const evidenceTexts = extractSearchTexts(results).slice(0, 10);

  const systemPrompt = `
You extract clinical indication/pathway options from drug protocol evidence.
Return ONLY valid JSON.
No markdown.
`;

  const userPrompt = `
Drug: ${drug}

Patient:
${JSON.stringify(patient, null, 2)}

Calculated metrics:
${JSON.stringify(metrics, null, 2)}

Evidence:
${evidenceTexts.map((t, i) => `SOURCE ${i + 1}:\n${t}`).join("\n\n")}

Task:
Return indication options specific to this drug.
If the drug has multiple pathway branches, split them into separate selectable options.
If age, dialysis, or renal function clearly changes the branch meaningfully, prefer the branch relevant to this patient.
Do not invent unsupported options.

Return exactly:
{
  "indications": [
    "..."
  ],
  "note": "...",
  "drug_note": "..."
}
`;

  const text = await runModel(env, systemPrompt, userPrompt);
  const parsed = parseJsonFromText(text);

  if (parsed && Array.isArray(parsed.indications)) {
    return parsed;
  }

  return {
    indications: [],
    note: "No indication options extracted",
    drug_note: `Could not extract pathways for ${drug}`
  };
}

/* =========================
   Final structured calculation
========================= */
async function calculateRecommendation(env, body) {
  const drug = (body.drug || "").trim();
  const indication = (body.indication || "").trim();
  const patient = body.patient || {};
  const extra = body.extra || {};

  if (!drug) return { ok: false, error: "Drug is required" };
  if (!indication) return { ok: false, error: "Indication is required" };

  const metrics = buildPatientMetrics(patient);

  const query = [
    drug,
    indication,
    "dose",
    "renal adjustment",
    "dialysis",
    "monitoring",
    patient.krt || "",
    patient.aki || "",
    patient.severity || ""
  ].filter(Boolean).join(" ");

  const results = await searchVectorStore(env, query, 10);
  const evidenceTexts = extractSearchTexts(results).slice(0, 10);

  const systemPrompt = `
You are THERAGUARD AI, a clinical dosing assistant.
Use ONLY supplied evidence and calculated patient data.
Return ONLY valid JSON.
No markdown.
Output must be short, practical, and ready to display.
`;

  const userPrompt = `
Patient:
${JSON.stringify(patient, null, 2)}

Calculated patient metrics:
${JSON.stringify(metrics, null, 2)}

Extra:
${JSON.stringify(extra, null, 2)}

Drug:
${drug}

Selected pathway / indication:
${indication}

Evidence:
${evidenceTexts.map((t, i) => `SOURCE ${i + 1}:\n${t}`).join("\n\n")}

Return exactly:
{
  "ok": true,
  "drug": "${drug}",
  "indication": "${indication}",
  "patient_summary": {
    "bmi": null,
    "ibw": null,
    "adjbw": null,
    "crcl_tbw": null,
    "crcl_adjbw": null
  },
  "final_dose": "...",
  "duration": "...",
  "dosing": ["..."],
  "monitoring": ["..."],
  "warnings": ["..."],
  "evidence": ["..."],
  "references": ["SOURCE 1", "SOURCE 2"],
  "safety": {
    "level": "ok",
    "message": "..."
  }
}

Rules:
- final_dose must be one clear final regimen
- duration should be short and specific if supported
- dosing can include supporting practical details
- monitoring must be short bullet items
- warnings must be short bullet items
- patient_summary must reuse the provided calculated metrics exactly if available
- if evidence is insufficient, say so clearly but still keep JSON structure
`;

  const text = await runModel(env, systemPrompt, userPrompt);
  const parsed = parseJsonFromText(text);

  if (parsed && parsed.ok) {
    if (!parsed.patient_summary) parsed.patient_summary = metrics;
    return parsed;
  }

  return {
    ok: true,
    drug,
    indication,
    patient_summary: metrics,
    final_dose: "Unable to generate structured final dose",
    duration: "Not clearly determined",
    dosing: [],
    monitoring: [],
    warnings: [],
    evidence: [],
    references: [],
    safety: {
      level: "warn",
      message: "Model returned unstructured output"
    }
  };
}
