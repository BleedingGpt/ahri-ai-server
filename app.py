import os
import requests
import json
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# Load environment variables (though Render uses its own system, this is good practice)
load_dotenv()

app = Flask(__name__)

# --- Configuration ---
# Render securely provides this as an environment variable
API_KEY = os.environ.get("GEMINI_API_KEY")

# Check API key presence immediately upon app load
if not API_KEY:
    # IMPORTANT: Do not print the key, just confirm it is missing.
    print("FATAL ERROR: GEMINI_API_KEY environment variable is not set. Check Render Environment settings.")
    # Raise an error to prevent gunicorn from starting up successfully.
    raise RuntimeError("GEMINI_API_KEY environment variable is not set.")

API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent"
HEADERS = {
    "Content-Type": "application/json"
}

# --- Utility Function ---

def get_gemini_content(data):
    """Handles the request to the Gemini API with necessary error checks."""
    try:
        # Add the API key to the request URL
        full_url = f"{API_URL}?key={API_KEY}"
        
        # Determine if grounding (Google Search) is requested
        use_search = data.pop('use_search', False)
        
        # Build the initial payload structure
        payload = {
            "contents": [{
                "parts": [{
                    "text": data.get("prompt", "")
                }]
            }],
            "systemInstruction": {
                "parts": [{
                    "text": data.get("system_instruction", "You are a helpful and concise assistant.")
                }]
            },
            "generationConfig": {
                # Ensure the model cannot stop prematurely (e.g., in the middle of a sentence)
                "maxOutputTokens": 2048
            }
        }
        
        if use_search:
            payload['tools'] = [{"google_search": {}}]

        # Send the request to the Gemini API
        response = requests.post(full_url, headers=HEADERS, data=json.dumps(payload))
        response.raise_for_status() # Raise exception for bad status codes (4xx or 5xx)

        response_json = response.json()
        
        # Robustly extract the generated text
        candidate = response_json.get('candidates', [{}])[0]
        content = candidate.get('content', {})
        parts = content.get('parts', [{}])
        
        # This is the critical line that needed the robust check
        generated_text = parts[0].get('text', "")

        if not generated_text:
            # Check for specific failure reasons from the model
            finish_reason = candidate.get('finishReason', 'UNKNOWN')
            if finish_reason == 'MAX_TOKENS':
                 return "Error: Model stopped prematurely due to max tokens. Try simplifying the prompt.", 500
            elif finish_reason == 'SAFETY':
                return "Error: Request or response blocked due to safety settings.", 400
            
            # Catch all other missing text errors
            return "Error: Gemini API returned a valid response but without generated text content.", 500


        return generated_text, 200

    except requests.exceptions.RequestException as e:
        # Log the failure to connect or get a proper HTTP response
        print(f"API Request Failure: {e}")
        return f"Error: Failed to connect to Gemini API. Check network or key validity. Details: {e}", 503
    except Exception as e:
        # Catch any unexpected errors during processing
        print(f"Unexpected Processing Error: {e}")
        return f"Error: An unexpected error occurred while processing the request. Details: {e}", 500

# --- Flask Routes ---

@app.route('/query', methods=['POST'])
def query_gemini():
    """Endpoint for the ESP32 to query the Gemini API."""
    if not request.is_json:
        return jsonify({"error": "Missing JSON in request"}), 400

    data = request.get_json()

    # Call the content handler
    result_text, status_code = get_gemini_content(data)
    
    # Check if the result is an error message
    if status_code != 200:
        # Return the error message directly with the appropriate status code
        return jsonify({"error": result_text}), status_code

    # If successful, return the cleaned answer
    return jsonify({"answer": result_text})

@app.route('/', methods=['GET'])
def home():
    """Simple endpoint to confirm the server is running."""
    return "Gemini API Proxy Server is running and ready for POST requests to /query."

# --- Run Application ---

# This is typically only used for local testing, Gunicorn handles production
if __name__ == '__main__':
    app.run(debug=True)
