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

    if (action === "get_sources") {
      const sources = await getSourcesFromVectorStore(env);
      return json({ ok: true, sources }, 200, corsHeaders);
    }

    if (action === "source_chat") {
      const result = await answerSourceChat(env, body);
      return json(result, 200, corsHeaders);
    }

    return json({ ok: false, error: "Unknown action" }, 400, corsHeaders);
  } catch (error) {
    console.error("API ERROR:", error);

    return json(
      {
        ok: false,
        error: "Server error",
        details: String(error && error.stack ? error.stack : error),
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
      ...corsHeaders,
    },
  });
}

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
   Vector store sources
========================= */

async function listVectorStoreFiles(env) {
  let allItems = [];
  let after = null;

  while (true) {
    const query = after
      ? `?limit=100&after=${encodeURIComponent(after)}`
      : `?limit=100`;

    const vsFiles = await openaiFetch(
      env,
      `/vector_stores/${env.VECTOR_STORE_ID}/files${query}`,
      { method: "GET" }
    );

    const items = Array.isArray(vsFiles.data) ? vsFiles.data : [];
    allItems.push(...items);

    if (!vsFiles.has_more || !items.length) break;

    after = items[items.length - 1]?.id;
    if (!after) break;
  }

  const out = [];

  for (const item of allItems) {
    const fileId = item.id || item.file_id;
    if (!fileId) continue;

    try {
      const fileObj = await openaiFetch(env, `/files/${fileId}`, { method: "GET" });
      const filename = fileObj?.filename || "";
      const sourceName = cleanSourceName(filename);

      out.push({
        file_id: fileId,
        filename,
        source_name: sourceName,
      });
    } catch (err) {
      console.error("FILE READ ERROR:", fileId, err);
    }
  }

  return out;
}

async function getSourcesFromVectorStore(env) {
  const files = await listVectorStoreFiles(env);

  const names = files
    .map(f => f.source_name)
    .filter(Boolean);

  return [...new Set(names)].sort((a, b) => a.localeCompare(b));
}

function cleanSourceName(filename) {
  if (!filename) return "";

  return filename
    .replace(/\.[^/.]+$/, "")
    .replace(/_/g, " ")
    .replace(/\s+/g, " ")
    .replace(/\bdrug information\b/gi, "")
    .replace(/\bmonograph\b/gi, "")
    .trim();
}

function normalize(s) {
  return String(s || "")
    .toLowerCase()
    .replace(/[^a-z0-9\u0600-\u06FF]+/g, " ")
    .trim();
}

async function findSourceFiles(env, source) {
  const files = await listVectorStoreFiles(env);
  const target = normalize(source);

  const exact = files.filter(f => normalize(f.source_name) === target);
  if (exact.length) return exact;

  const loose = files.filter(f => {
    const src = normalize(f.source_name);
    const name = normalize(f.filename);
    return src.includes(target) || target.includes(src) || name.includes(target);
  });

  return loose;
}

/* =========================
   Search helpers
========================= */

async function searchVectorStore(env, query, maxNumResults = 10) {
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
  for (const f of files) {
    map.set(f.file_id, f);
  }
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
    const parts = Array.isArray(item.content) ? item.content : [];
    const textChunk = parts.find(c => c && c.type === "text" && c.text);

    if (!textChunk || !textChunk.text) continue;

    const mapped = fileMap.get(item.file_id);

    evidence.push({
      file_id: item.file_id || null,
      source: mapped?.filename || mapped?.source_name || "Unknown source",
      text: textChunk.text,
      page: parsePageFromText(textChunk.text),
    });
  }

  return evidence;
}

/* =========================
   Model helpers
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
   Source chat
========================= */

async function answerSourceChat(env, body) {
  const source = String(body.source || "").trim();
  const message = String(body.message || "").trim();
  const language = String(body.language || "auto").trim();
  const evidenceMode = String(body.evidence_mode || "standard").trim();
  const history = Array.isArray(body.history) ? body.history : [];

  if (!source) {
    return { ok: false, error: "Source is required" };
  }

  if (!message) {
    return { ok: false, error: "Message is required" };
  }

  const sourceFiles = await findSourceFiles(env, source);
  if (!sourceFiles.length) {
    return { ok: false, error: `No vector-store file matched for ${source}` };
  }

  const fileIds = sourceFiles.map(f => f.file_id);
  const fileMap = buildFileMap(sourceFiles);

  const compactHistory = history
    .filter(item => item && (item.role === "user" || item.role === "assistant"))
    .slice(-8)
    .map(item => `${item.role.toUpperCase()}: ${String(item.content || "").trim()}`)
    .join("\n");

  const query = [
    source,
    message,
    compactHistory
  ].filter(Boolean).join("\n");

  let results = await searchVectorStore(env, query, evidenceMode === "strict" ? 14 : 10);
  results = filterResultsByFileIds(results, fileIds);

  const evidence = mapResultsToEvidence(results, fileMap).slice(0, evidenceMode === "strict" ? 6 : 4);

  const systemPrompt = `
You are a source-locked clinical knowledge chat assistant.

Rules:
- Answer ONLY from the supplied evidence taken from the selected source.
- Do NOT use outside knowledge.
- Do NOT answer from any other file.
- Support Arabic, English, or mixed language naturally.
- Tolerate minor spelling mistakes and shorthand.
- Use the conversation history to understand follow-up questions.
- Keep the answer clear and conversational.
- If the answer is not clearly supported by evidence, say so.
- Return ONLY valid JSON.
- No markdown.
`;

  const userPrompt = `
Selected source:
${source}

Language mode:
${language}

Evidence mode:
${evidenceMode}

Conversation history:
${compactHistory || "No prior history"}

User message:
${message}

Evidence from selected source only:
${evidence.map((e, i) => `SOURCE ${i + 1} (${e.source}${e.page ? `, Page ${e.page}` : ""}):\n${e.text}`).join("\n\n")}

Return exactly:
{
  "ok": true,
  "answer": "...",
  "citations": [
    {
      "source": "...",
      "page": null,
      "quote": "..."
    }
  ]
}

Rules:
- Answer in the user's apparent language when language mode is auto
- citations should contain 1 to 3 items if possible
- quote must be short and directly supported by supplied evidence
- do not invent page numbers
- if page is unknown, use null
`;

  const text = await runModel(env, systemPrompt, userPrompt);
  const parsed = parseJsonFromText(text);

  if (parsed?.ok) {
    return {
      ok: true,
      answer: String(parsed.answer || "").trim() || "No answer returned.",
      citations: Array.isArray(parsed.citations) ? parsed.citations.slice(0, 3) : [],
    };
  }

  return {
    ok: true,
    answer: evidence.length
      ? "I found relevant text in the selected source, but I could not build a structured response."
      : "No clearly relevant evidence was found in the selected source.",
    citations: evidence.slice(0, 3).map(e => ({
      source: e.source,
      page: e.page || null,
      quote: e.text.slice(0, 280),
    })),
  };
}
