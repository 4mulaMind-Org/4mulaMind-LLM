import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
4mulaMind NLP Server
Wraps IntentPredictor + regex parameter extraction to produce
the exact JSON format expected by 4mulaQuery's NLPQueryService.java
"""

import re
from flask import Flask, request, jsonify
from src.inference import IntentPredictor

app = Flask(__name__)

# Load model once at startup
print("Loading 4mulaMind-LLM...")
predictor = IntentPredictor(
    checkpoint_path="checkpoints/4mulamind_4mulaquery_intent_sft.pt",
    tokenizer_path="tokenizers/4mulamind_bpe.json",
)
print("Model ready.")


def extract_params(instruction):
    """Extract id, username, email from natural language."""
    params = {"id": None, "username": None, "email": None}

    # Email extraction
    email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', instruction)
    if email_match:
        params["email"] = email_match.group(0)

    # ID extraction: "id 135", "id=135", "user 135", "user number 135"
    id_match = re.search(
        r'\bid\s*[=:]?\s*(\d+)|\buser\s+(?:number\s+)?(\d+)',
        instruction,
        re.IGNORECASE,
    )
    if id_match:
        params["id"] = int(id_match.group(1) or id_match.group(2))

    # Username: "named X", "username X", "name X"
    name_match = re.search(
        r'\b(?:named|username|name)\s+([a-zA-Z][a-zA-Z0-9_]{0,31})'
        r'|\buser\s+(?![0-9]|named\b|with\b|email\b|id\b)([a-zA-Z][a-zA-Z0-9_]{0,31})',
        instruction,
        re.IGNORECASE,
    )
    if name_match:
        params["username"] = name_match.group(1) or name_match.group(2)

    return params


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "4mulaMind-LLM"})


@app.route("/translate", methods=["POST"])
def translate():
    """Accepts {"query": "..."} and returns JSON command for Java."""
    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()

    if not query:
        return jsonify({
            "command": "UNSUPPORTED",
            "id": None, "username": None, "email": None,
            "reason": "Empty query",
        }), 400

    try:
        intent = predictor.predict(query).strip().upper()
    except Exception as e:
        return jsonify({
            "command": "UNSUPPORTED",
            "id": None, "username": None, "email": None,
            "reason": f"Model error: {e}",
        }), 500

    # Validate intent
    if intent not in ("SEARCH", "INSERT", "DELETE", "ALL"):
        return jsonify({
            "command": "UNSUPPORTED",
            "id": None, "username": None, "email": None,
            "reason": f"Model returned unknown intent: {intent}",
        }), 200

    params = extract_params(query)

    return jsonify({
        "command": intent,
        "id": params["id"],
        "username": params["username"],
        "email": params["email"],
        "reason": None,
    }), 200


if __name__ == "__main__":
    print("Starting NLP server on http://0.0.0.0:5001")
    app.run(host="0.0.0.0", port=5001, debug=False)
