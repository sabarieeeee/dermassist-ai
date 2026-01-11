import { GoogleGenAI, Type } from "@google/genai";
import { SkinAnalysis } from "../types";

const ai = new GoogleGenAI({ apiKey: process.env.API_KEY || '' });

const ANALYSIS_SCHEMA = {
  type: Type.OBJECT,
  properties: {
    isSkin: { type: Type.BOOLEAN, description: "Whether the image is clearly human skin." },
    isHealthy: { type: Type.BOOLEAN, description: "Whether the skin appears healthy without notable rashes or lesions." },
    diseaseName: { type: Type.STRING, description: "Likely name of the skin condition." },
    description: { type: Type.STRING, description: "A detailed medical overview of the condition." },
    treatments: { type: Type.ARRAY, items: { type: Type.STRING }, description: "Recommended treatment approaches." },
    medicines: { type: Type.ARRAY, items: { type: Type.STRING }, description: "Common over-the-counter or clinical medicines often used." },
    symptoms: { type: Type.ARRAY, items: { type: Type.STRING }, description: "Key visual or sensory symptoms." },
    reasons: { type: Type.ARRAY, items: { type: Type.STRING }, description: "Common causes or triggers for this condition." },
    healingPeriod: { type: Type.STRING, description: "Typical duration for recovery." },
    precautions: { type: Type.ARRAY, items: { type: Type.STRING }, description: "Immediate precautions to take." },
    prevention: { type: Type.ARRAY, items: { type: Type.STRING }, description: "Long-term prevention strategies." },
  },
  required: ["isSkin", "isHealthy"]
};

const safeParseJson = (text: string) => {
  try {
    const cleaned = text.replace(/```json/g, '').replace(/```/g, '').trim();
    return JSON.parse(cleaned);
  } catch (e) {
    console.error("Failed to parse Gemini JSON:", e);
    return {};
  }
};

export const analyzeSkinImage = async (base64Image: string): Promise<SkinAnalysis> => {
  const response = await ai.models.generateContent({
    model: 'gemini-3-flash-preview',
    contents: {
      parts: [
        {
          text: `Act as a professional dermatological assistant. 
          Analyze this skin image using visual features recognized in the HAM10000 dataset benchmarks (e.g., pigment networks, streaks, dots/globules).
          1. Verify if it is human skin. If not, set isSkin to false.
          2. If skin, check for abnormalities. If healthy, set isHealthy to true.
          3. Provide comprehensive details: diseaseName, description, symptoms, reasons, treatments, medicines, healingPeriod, precautions, and prevention.
          4. ALWAYS prioritize educational guidance and state this is not a clinical diagnosis.`
        },
        {
          inlineData: {
            mimeType: "image/jpeg",
            data: base64Image.split(',')[1]
          }
        }
      ]
    },
    config: {
      responseMimeType: "application/json",
      responseSchema: ANALYSIS_SCHEMA
    }
  });

  return safeParseJson(response.text || '{}') as SkinAnalysis;
};

export const compareProgression = async (img1: string, img2: string): Promise<string> => {
  const response = await ai.models.generateContent({
    model: 'gemini-3-flash-preview',
    contents: {
      parts: [
        {
          text: `Compare these two images of the same skin condition taken at different times (Baseline vs Current). 
          Assess healing progress based on visual resolution of rashes, pigment changes, and texture. 
          Provide a concise progress report.`
        },
        {
          inlineData: { mimeType: "image/jpeg", data: img1.split(',')[1] }
        },
        {
          inlineData: { mimeType: "image/jpeg", data: img2.split(',')[1] }
        }
      ]
    }
  });

  return response.text || "Unable to generate comparison report.";
};