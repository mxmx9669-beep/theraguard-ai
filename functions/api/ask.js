export async function onRequest(context) {
  const { request, env } = context;

  const corsHeaders = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
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
      return json(
        {
          ok: true,
          drugs,
          note: "Drug list loaded from vector store files",
        },
        200,
        corsHeaders
      );
    }

    if (action === "get_indications") {
      const drug = (body.drug || "").trim();
      if (!drug) {
        return json({ ok: false, error: "Drug is required" }, 400, corsHeaders);
      }

      const indicationsResult = await getIndicationsForDrug(env, drug);
      return json(
        {
          ok: true,
          indications: indicationsResult.indications || [],
          note: indicationsResult.note || "Indications loaded",
          drug_note: indicationsResult.drug_note || `Loaded from vector store for ${drug}`,
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
        details: String(error),
      },
      500,
      corsHeaders
    );
  }
}

/* =========================
   Core JSON helper
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
    throw new Error(
      `OpenAI request failed (${response.status}): ${JSON.stringify(data)}`
    );
  }

  return data;
}

/* =========================
   Vector store file → drug list
========================= */
async function getDrugsFromVectorStore(env) {
  // 1) List files attached to the vector store
  const vsFiles = await openaiFetch(
    env,
    `/vector_stores/${env.VECTOR_STORE_ID}/files`,
    { method: "GET" }
  );

  const fileItems = Array.isArray(vsFiles.data) ? vsFiles.data : [];

  // 2) Retrieve each original file object to get filename
  const names = [];
  for (const item of fileItems) {
    const fileId = item.id || item.file_id;
    if (!fileId) continue;

    try {
      const fileObj = await openaiFetch(env, `/files/${fileId}`, { method: "GET" });
      if (fileObj && fileObj.filename) {
        const pretty = filenameToDrugName(fileObj.filename);
        if (pretty) names.push(pretty);
      }
    } catch (_) {
      // ignore individual file failures
    }
  }

  // unique + sorted
  return [...new Set(names)].sort((a, b) => a.localeCompare(b));
}

function filenameToDrugName(filename) {
  if (!filename) return "";

  let name = filename
    .replace(/\.[^/.]+$/, "") // remove extension
    .replace(/_/g, " ")
    .replace(/\s+/g, " ")
    .trim();

  // Clean common protocol suffix noise
  name = name
    .replace(/\bdrug information\b/gi, "")
    .replace(/\bmonograph\b/gi, "")
    .replace(/\bguideline\b/gi, "")
    .replace(/\bprotocol\b/gi, "Protocol")
    .replace(/\(\s*\)/g, "")
    .replace(/\s{2,}/g, " ")
    .trim();

  return name;
}

/* =========================
   Search helper
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

function extractSearchTexts(results) {
  const chunks = [];

  for (const item of results) {
    if (!Array.isArray(item.content)) continue;

    for (const c of item.content) {
      if (c.type === "text" && c.text) {
        chunks.push(c.text);
      }
    }
  }

  return chunks;
}

/* =========================
   Responses helper
========================= */
async function runModel(env, prompt, systemPrompt) {
  const model = env.MODEL || "gpt-4.1-mini";

  const body = {
    model,
    input: [
      {
        role: "system",
        content: [{ type: "input_text", text: systemPrompt }],
      },
      {
        role: "user",
        content: [{ type: "input_text", text: prompt }],
      },
    ],
  };

  const data = await openaiFetch(env, `/responses`, {
    method: "POST",
    body: JSON.stringify(body),
  });

  return {
    model,
    text: extractOutputText(data),
    raw: data,
  };
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
  if (!text) return null;

  // Try direct parse first
  try {
    return JSON.parse(text);
  } catch (_) {}

  // Try fenced code block
  const fenceMatch = text.match(/```json\s*([\s\S]*?)```/i) || text.match(/```([\s\S]*?)```/i);
  if (fenceMatch) {
    try {
      return JSON.parse(fenceMatch[1].trim());
    } catch (_) {}
  }

  // Try first {...} block
  const firstBrace = text.indexOf("{");
  const lastBrace = text.lastIndexOf("}");
  if (firstBrace !== -1 && lastBrace !== -1 && lastBrace > firstBrace) {
    try {
      return JSON.parse(text.slice(firstBrace, lastBrace + 1));
    } catch (_) {}
  }

  return null;
}

/* =========================
   Action: get_indications
========================= */
async function getIndicationsForDrug(env, drug) {
  const results = await searchVectorStore(
    env,
    `${drug} indications approved uses clinical pathway dosing indications list`
  );

  const evidenceTexts = extractSearchTexts(results).slice(0, 8);

  const systemPrompt = `
You are a clinical document extraction assistant.
Return ONLY valid JSON.
Do not add markdown.
`;

  const prompt = `
Extract the indication list for this drug from the supplied evidence.

Drug: ${drug}

Evidence:
${evidenceTexts.map((t, i) => `SOURCE ${i + 1}:\n${t}`).join("\n\n")}

Return exactly this JSON shape:
{
  "indications": ["..."],
  "note": "...",
  "drug_note": "..."
}

Rules:
- indications must be short, clean, human-readable strings
- remove duplicates
- do not invent indications not supported by evidence
- if evidence is weak, return an empty array
`;

  const modelResult = await runModel(env, prompt, systemPrompt);
  const parsed = parseJsonFromText(modelResult.text);

  if (parsed && Array.isArray(parsed.indications)) {
    return parsed;
  }

  return {
    indications: [],
    note: "No structured indications extracted",
    drug_note: `Backend could not confidently extract indications for ${drug}`,
  };
}

/* =========================
   Action: calculate
========================= */
async function calculateRecommendation(env, body) {
  const drug = (body.drug || "").trim();
  const indication = (body.indication || "").trim();
  const patient = body.patient || {};
  const extra = body.extra || {};
  const question =
    (body.question || "").trim() ||
    `Provide patient-specific dosing recommendation for ${drug} for ${indication}.`;

  if (!drug) {
    return { ok: false, error: "Drug is required" };
  }

  if (!indication) {
    return { ok: false, error: "Indication is required" };
  }

  const query = [
    drug,
    indication,
    "dose",
    "renal adjustment",
    "dialysis",
    "monitoring",
    "contraindications",
    patient.krt || "",
    patient.severity || "",
  ]
    .filter(Boolean)
    .join(" ");

  const searchResults = await searchVectorStore(env, query, 10);
  const evidenceTexts = extractSearchTexts(searchResults).slice(0, 10);

  const systemPrompt = `
You are THERAGUARD AI, a clinical medication safety and dosing assistant.
Use ONLY the supplied evidence.
If evidence is insufficient, say so inside the JSON fields rather than inventing facts.
Return ONLY valid JSON.
No markdown.
`;

  const prompt = `
Create a patient-specific structured recommendation.

PATIENT:
${JSON.stringify(patient, null, 2)}

EXTRA:
${JSON.stringify(extra, null, 2)}

DRUG:
${drug}

INDICATION:
${indication}

QUESTION:
${question}

EVIDENCE:
${evidenceTexts.map((t, i) => `SOURCE ${i + 1}:\n${t}`).join("\n\n")}

Return exactly this JSON:
{
  "ok": true,
  "drug": "${drug}",
  "indication": "${indication}",
  "summary": "...",
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
- dosing = concrete actionable dose / interval / special adjustments
- monitoring = labs, troughs, renal function, ECG, bleeding, etc. as supported by evidence
- warnings = major precautions only if supported
- evidence = short evidence bullets, not long paragraphs
- references = use SOURCE numbers only
- safety.level must be one of: ok, warn, bad
- no markdown
- no extra text outside JSON
`;

  const modelResult = await runModel(env, prompt, systemPrompt);
  const parsed = parseJsonFromText(modelResult.text);

  if (parsed && parsed.ok) {
    return parsed;
  }

  return {
    ok: true,
    drug,
    indication,
    summary: "Structured parsing failed, returning model text.",
    dosing: [],
    monitoring: [],
    warnings: [],
    evidence: [],
    references: [],
    safety: {
      level: "warn",
      message: "Backend generated an unstructured response.",
    },
    recommendation: modelResult.text || "No recommendation generated.",
  };
}
