from flask import Flask, request, jsonify
import requests
import json 
import os 
# --- START: Local Development Setup (Necessary for local testing) ---
# NOTE: load_dotenv() is included for local testing only. 
# It is ignored when deploying to a platform like Render.
from dotenv import load_dotenv
load_dotenv()
# --- END: Local Development Setup ---

app = Flask(__name__)

# === CRITICAL FOR DEPLOYMENT: Read API Key from Environment Variable ===
# In a secure deployment environment (like Render), you MUST set 
# a variable named GEMINI_API_KEY in the service settings.
# The code securely reads the key from the environment.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_IS_MISSING")

# Note: Changing model to match the flash model used in the ESP32 code
GEMINI_API_MODEL = "gemini-2.5-flash-preview-09-2025" 
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_API_MODEL}:generateContent?key={GEMINI_API_KEY}"

@app.route("/query", methods=["POST"])
def query_gemini():
    """
    Endpoint to receive a prompt and send it to the Gemini API.
    Expected JSON input: 
    {
        "prompt": "Your query text", 
        "system_instruction": "Optional persona instruction",
        "use_search": true/false (optional),
        "max_tokens": 50 (optional, limits response length for speed)
    }
    """
    data = request.json
    prompt = data.get("prompt", "")
    system_instruction = data.get("system_instruction") 
    use_search = data.get("use_search", False)
    
    # Setting a generous default of 400 tokens to prevent incomplete responses.
    # This addresses the original problem of truncated sentences.
    max_tokens = data.get("max_tokens", 400) 
    
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    # This check ensures the API key is actually set on the server.
    if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_IS_MISSING":
        return jsonify({"error": "API Key is not set in the hosting service's environment variables. Please set GEMINI_API_KEY."}), 500

    # Initialize the payload with contents
    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    
    config = {}

    # Add system instruction if provided
    if system_instruction:
        payload["systemInstruction"] = {
            "parts": [{"text": system_instruction}]
        }
    
    # Add Google Search grounding if requested
    if use_search:
        payload["tools"] = [{ "google_search": {} }]

    # Add maxOutputTokens to generationConfig
    if max_tokens is not None:
        try:
            config["maxOutputTokens"] = int(max_tokens)
        except ValueError:
            return jsonify({"error": "max_tokens must be an integer."}), 400

    if config:
        payload["generationConfig"] = config

    print("--- Sending Payload ---")
    print(json.dumps(payload, indent=2))
    print("-----------------------")
        
    try:
        response = requests.post(GEMINI_API_URL, json=payload)
        response.raise_for_status() 
        
        print(f"API Status: {response.status_code}")
        
        result = response.json()
        
        # --- Ultra-Safe Parsing ---
        candidates = result.get("candidates", [{}])
        if candidates:
            first_candidate = candidates[0]
            content = first_candidate.get("content", {})
            parts = content.get("parts", [{}])
            
            if parts and parts[0].get("text"):
                answer = parts[0].get("text")
                return jsonify({"answer": answer})
        
        # Check for blocked prompt feedback
        prompt_feedback = result.get("promptFeedback")
        if prompt_feedback:
            reason = prompt_feedback.get("blockReason", "Unknown reason.")
            return jsonify({"error": f"Generation blocked by Gemini safety filters or content policy. Reason: {reason}"}), 500
        
        # Fallback for unparseable but successful API responses
        print("\n!!! RAW API RESPONSE ON PARSING FAILURE !!!")
        print(json.dumps(result, indent=2))
        print("------------------------------------------\n")
        return jsonify({"error": "Failed to extract text from API response. Check console logs for raw output."}), 500
        
    except requests.exceptions.RequestException as e:
        error_details = getattr(e.response, 'text', str(e))
        print(f"API Request Error Details: {error_details}")
        return jsonify({"error": f"API Request Error (HTTP Failure): {e}. Check if your GEMINI_API_KEY is valid."}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # NOTE: app.run() is ONLY for local development!
    app.run(host="0.0.0.0", port=5000)
