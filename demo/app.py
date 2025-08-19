# app.py
import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
import google.genai as genai
from google.genai.types import Content, Part, GenerateContentConfig
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
   raise RuntimeError("Missing GEMINI_API_KEY in .env")
client = genai.Client(api_key=API_KEY)
model_name = "gemini-1.5-flash"  # rẻ/nhanh cho chatbot
app = Flask(__name__)
def build_contents(history, user_message):
   """
   history: [(user, bot), ...]
   user_message: str
   -> List[Content] theo định dạng SDK mới
   """
   contents = []
   if history:
       for u, b in history:
           if u:
               contents.append(Content(role="user",  parts=[Part.from_text(u)]))
           if b:
               contents.append(Content(role="model", parts=[Part.from_text(b)]))
   contents.append(Content(role="user", parts=[Part.from_text(user_message)]))
   return contents
@app.route("/chat", methods=["POST"])
def chat():
   data = request.get_json(silent=True) or {}
   message = data.get("message", "").strip()
   history = data.get("history", [])
   if not message:
       return jsonify({"error": "Missing 'message'"}), 400
   try:
       resp = client.models.generate_content(
           model=model_name,
           contents=build_contents(history, message),
           config=GenerateContentConfig(
               temperature=0.7,
               max_output_tokens=512,
           ),
           system_instruction=Content(
               role="system",
               parts=[Part.from_text(
                   "You are a friendly, concise Vietnamese assistant. "
                   "Answer clearly and helpfully."
               )]
           ),
       )
       return jsonify({"response": resp.text})
   except Exception as e:
       return jsonify({"error": str(e)}), 500
if __name__ == "__main__":
   # chạy:  python app.py
   app.run(host="0.0.0.0", port=5000, debug=True)