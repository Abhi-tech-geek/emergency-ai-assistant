from fastapi import FastAPI
from pydantic import BaseModel
from groq import Groq
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#  groq api key here
client = Groq(api_key=API_KEY)

class EmergencyRequest(BaseModel):
    text: str

# fallback safety system
def fallback(text):
    text = text.lower()

    if "bleeding" in text or "blood" in text:
        return "Severe Bleeding:\n1. Apply pressure\n2. Use clean cloth\n3. Elevate wound\n4. Call ambulance"

    if "burn" in text:
        return "Burn Injury:\n1. Cool under water\n2. Do NOT apply ice\n3. Cover loosely\n4. Seek medical help"

    if "chest pain" in text or "heart" in text:
        return "Possible Heart Attack:\n1. Make person sit\n2. Loosen clothes\n3. Call emergency\n4. Start CPR if unconscious"

    return "Unknown emergency. Call emergency services immediately."

@app.post("/analyze")
def analyze(req: EmergencyRequest):
    try:
        prompt = f"""
You are an emergency medical assistant.

1. Identify the emergency.
2. Give clear first aid steps.
3. Keep instructions short and actionable.

Emergency: {req.text}
        """

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "user", "content": prompt}]
        )

        return {"result": response.choices[0].message.content}

    except Exception as e:
        # if internet fails, fallback runs
        return {"result": fallback(req.text)}
