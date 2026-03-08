export async function onRequest(context) {
  const { request, env } = context;

  const corsHeaders = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
  };

  if (request.method === "OPTIONS") {
    return new Response(null, {
      status: 204,
      headers: corsHeaders,
    });
  }

  if (request.method !== "POST") {
    return new Response(
      JSON.stringify({ error: "Method not allowed" }),
      {
        status: 405,
        headers: {
          "Content-Type": "application/json",
          ...corsHeaders,
        },
      }
    );
  }

  try {
    const body = await request.json();

    const {
      question = "",
      patient = {},
      drug = "",
      indication = "",
      extra = {}
    } = body || {};

    if (!env.OPENAI_API_KEY) {
      return new Response(
        JSON.stringify({ error: "OPENAI_API_KEY missing in environment variables" }),
        {
          status: 500,
          headers: {
            "Content-Type": "application/json",
            ...corsHeaders,
          },
        }
      );
    }

    if (!env.VECTOR_STORE_ID) {
      return new Response(
        JSON.stringify({ error: "VECTOR_STORE_ID missing in environment variables" }),
        {
          status: 500,
          headers: {
            "Content-Type": "application/json",
            ...corsHeaders,
          },
        }
      );
    }

    const model = env.MODEL || "gpt-4.1-mini";

    const userPrompt = `
You are THERAGUARD AI, a clinical dosing and medication safety assistant.

Use the vector store as the primary evidence source.
Answer only from retrieved evidence when possible.
Return concise, structured clinical output.

PATIENT DATA:
${JSON.stringify(patient, null, 2)}

DRUG:
${drug || "Not provided"}

INDICATION:
${indication || "Not provided"}

EXTRA PARAMETERS:
${JSON.stringify(extra, null, 2)}

QUESTION:
${question || `Provide dosing recommendation, monitoring, and key precautions for ${drug} in this patient.`}
`;

    const response = await fetch("https://api.openai.com/v1/responses", {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${env.OPENAI_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model,
        input: userPrompt,
        tools: [
          {
            type: "file_search",
            vector_store_ids: [env.VECTOR_STORE_ID],
          }
        ]
      }),
    });

    const data = await response.json();

    if (!response.ok) {
      return new Response(
        JSON.stringify({
          error: "OpenAI API request failed",
          details: data
        }),
        {
          status: 500,
          headers: {
            "Content-Type": "application/json",
            ...corsHeaders,
          },
        }
      );
    }

    let finalText = "";
    if (Array.isArray(data.output)) {
      for (const item of data.output) {
        if (item.type === "message" && Array.isArray(item.content)) {
          for (const part of item.content) {
            if (part.type === "output_text" && part.text) {
              finalText += part.text;
            }
          }
        }
      }
    }

    return new Response(
      JSON.stringify({
        ok: true,
        model,
        recommendation: finalText || "No recommendation generated.",
        raw: data
      }),
      {
        status: 200,
        headers: {
          "Content-Type": "application/json",
          ...corsHeaders,
        },
      }
    );
  } catch (error) {
    return new Response(
      JSON.stringify({
        error: "Server error",
        details: String(error)
      }),
      {
        status: 500,
        headers: {
          "Content-Type": "application/json",
          ...corsHeaders,
        },
      }
    );
  }
}
