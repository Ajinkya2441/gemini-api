from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from datetime import datetime, timedelta
import google.generativeai as genai
import os

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize FastAPI app
app = FastAPI()

# Request schema
class QuestionRequest(BaseModel):
    question: str


@app.post("/ask")
async def ask_question(query: QuestionRequest):
    question = query.question.strip().lower()

    # Handle local logic for date/time
    if "yesterday" in question:
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%A, %B %d, %Y')
        return {"answer": f"Yesterday was {yesterday}."}
    elif "today" in question or "date" in question:
        today = datetime.now().strftime('%A, %B %d, %Y')
        return {"answer": f"Today's date is {today}."}
    elif "time" in question:
        current_time = datetime.now().strftime('%I:%M %p')
        return {"answer": f"The current time is {current_time}."}

    # Use Gemini Pro with simpler prompt
    try:
        prompt = f"""
Please provide a concise answer to the following question. 
Answer should be direct and to the point, without sections or emojis.
If the question is about a programming concept, include one short code example if relevant.

Question: {question}
"""
        model = genai.GenerativeModel("models/gemma-3-4b-it")
        chat = model.start_chat(history=[])
        response = chat.send_message(prompt)
        return {"answer": response.text.strip()}

    except Exception as e:
        return JSONResponse(status_code=500, content={
            "answer": f"Error: {str(e)}. Check your Gemini API key or usage quota."
        })