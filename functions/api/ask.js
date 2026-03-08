// functions/api/ask.js
export async function onRequest(context) {
  // Handle CORS preflight
  if (context.request.method === "OPTIONS") {
    return new Response(null, {
      headers: {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
      },
    });
  }

  // Only accept POST requests
  if (context.request.method !== "POST") {
    return new Response(JSON.stringify({ error: "Method not allowed" }), {
      status: 405,
      headers: { 
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*"
      },
    });
  }

  try {
    // Parse request body
    const body = await context.request.json();
    
    // Validate required fields
    if (!body.patient || !body.drug) {
      return new Response(JSON.stringify({ 
        success: false, 
        error: "Missing patient or drug data" 
      }), {
        status: 400,
        headers: { 
          "Content-Type": "application/json",
          "Access-Control-Allow-Origin": "*"
        },
      });
    }

    // Extract data
    const { patient, drug, calculated, additional } = body;
    
    // Get environment variables
    const OPENAI_API_KEY = context.env.OPENAI_API_KEY;
    const VECTOR_STORE_ID = context.env.VECTOR_STORE_ID;
    const MODEL = context.env.MODEL || "gpt-4";
    
    if (!OPENAI_API_KEY || !VECTOR_STORE_ID) {
      throw new Error("Missing OpenAI configuration");
    }

    // Step 1: Query Vector Store for relevant clinical content
    const vectorResults = await queryVectorStore(
      OPENAI_API_KEY,
      VECTOR_STORE_ID,
      drug.name,
      drug.indication
    );

    // Step 2: Build prompt with patient data and vector results
    const prompt = buildPrompt(patient, drug, calculated, additional, vectorResults);
    
    // Step 3: Call OpenAI with the prompt
    const aiResponse = await callOpenAI(OPENAI_API_KEY, MODEL, prompt);
    
    // Step 4: Parse and structure the response
    const structuredResponse = parseAIResponse(aiResponse, drug.name, drug.indication);
    
    // Step 5: Return JSON to frontend
    return new Response(JSON.stringify({
      success: true,
      drug: drug.name,
      indication: drug.indication,
      dosing: structuredResponse.dosing || [],
      safety: structuredResponse.safety || { level: "ok", message: "No specific safety concerns" },
      evidence: structuredResponse.evidence || [],
      references: structuredResponse.references || [],
      timestamp: new Date().toISOString()
    }), {
      headers: {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*",
      },
    });

  } catch (error) {
    console.error("Backend error:", error);
    
    return new Response(JSON.stringify({ 
      success: false, 
      error: error.message || "Internal server error"
    }), {
      status: 500,
      headers: { 
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*"
      },
    });
  }
}

/**
 * Query OpenAI Vector Store for relevant clinical content
 */
async function queryVectorStore(apiKey, vectorStoreId, drugName, indication) {
  try {
    const response = await fetch(`https://api.openai.com/v1/vector_stores/${vectorStoreId}/search`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
        'OpenAI-Beta': 'assistants=v2'
      },
      body: JSON.stringify({
        query: `${drugName} ${indication} dosing renal adjustment monitoring`,
        max_num_results: 5
      })
    });

    if (!response.ok) {
      const error = await response.text();
      console.error("Vector store error:", error);
      return [];
    }

    const data = await response.json();
    return data.data || [];
    
  } catch (error) {
    console.error("Vector store query failed:", error);
    return []; // Return empty array on failure, continue with general AI knowledge
  }
}

/**
 * Build prompt for OpenAI
 */
function buildPrompt(patient, drug, calculated, additional, vectorResults) {
  // Format patient data
  const patientInfo = `
Age: ${patient.age} years
Sex: ${patient.sex}
Weight: ${patient.weight} kg
Height: ${patient.height} cm
Serum Creatinine: ${patient.creatinine} µmol/L
AKI: ${patient.aki}
Renal Replacement: ${patient.krt}
Route Preference: ${patient.route_preference}
Severity: ${patient.severity}
Infusion Strategy: ${patient.infusion}
${patient.pregnant ? `Pregnant: ${patient.pregnant}` : ''}
${patient.lactating ? `Lactating: ${patient.lactating}` : ''}
  `;

  // Format calculated values
  const calculatedInfo = `
BMI: ${calculated.bmi?.toFixed(1) || 'N/A'}
CrCl (TBW): ${calculated.crcl_tbw?.toFixed(1) || 'N/A'} mL/min
IBW: ${calculated.ibw?.toFixed(1) || 'N/A'} kg
AdjBW: ${calculated.adjbw?.toFixed(1) || 'N/A'} kg
  `;

  // Format additional parameters
  const additionalInfo = `
Vancomycin Trough: ${additional.vanco_trough || 'N/A'} mg/L
Augmented Renal Clearance: ${additional.arc}
ARC Band: ${additional.arc_band || 'N/A'}
PIRRT Residual UO: ${additional.pirrt_uo}
Procainamide Rate: ${additional.proc_rate || 'N/A'} mg/min
Child-Pugh Class: ${additional.child_pugh}
  `;

  // Format vector store results if available
  let vectorContext = "";
  if (vectorResults && vectorResults.length > 0) {
    vectorContext = "\nRelevant clinical guidelines from vector store:\n";
    vectorResults.forEach((result, i) => {
      vectorContext += `\n[Source ${i+1}]: ${result.content || JSON.stringify(result)}`;
    });
  }

  // Build the complete prompt
  return `You are a clinical dosing expert. Provide specific dosing recommendations based on:

PATIENT DATA:
${patientInfo}

CALCULATED METRICS:
${calculatedInfo}

ADDITIONAL PARAMETERS:
${additionalInfo}

DRUG: ${drug.name}
INDICATION: ${drug.indication}
${vectorContext}

Return a structured JSON response with:
1. dosing: array of specific dosing instructions (include loading dose, maintenance dose, interval, route)
2. safety: object with level ("ok", "warn", "bad") and message
3. evidence: array of supporting statements from guidelines
4. references: array of citations

Format your response as valid JSON only, no other text.`;
}

/**
 * Call OpenAI API
 */
async function callOpenAI(apiKey, model, prompt) {
  const response = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${apiKey}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      model: model,
      messages: [
        {
          role: "system",
          content: "You are a clinical dosing expert. Return only valid JSON, no other text."
        },
        {
          role: "user",
          content: prompt
        }
      ],
      temperature: 0.3,
      response_format: { type: "json_object" }
    })
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`OpenAI API error: ${error}`);
  }

  const data = await response.json();
  return data.choices[0].message.content;
}

/**
 * Parse and validate AI response
 */
function parseAIResponse(aiResponse, drugName, indication) {
  try {
    // Parse JSON from AI response
    const parsed = JSON.parse(aiResponse);
    
    // Ensure required structure
    return {
      dosing: Array.isArray(parsed.dosing) ? parsed.dosing : [parsed.dosing || "See recommendation below"],
      safety: parsed.safety || { level: "ok", message: "No specific safety concerns" },
      evidence: Array.isArray(parsed.evidence) ? parsed.evidence : [],
      references: Array.isArray(parsed.references) ? parsed.references : []
    };
    
  } catch (error) {
    console.error("Failed to parse AI response:", error);
    
    // Fallback response if parsing fails
    return {
      dosing: [aiResponse.substring(0, 200) + "..."],
      safety: { level: "warn", message: "AI response parsing failed, showing raw output" },
      evidence: [],
      references: []
    };
  }
}
