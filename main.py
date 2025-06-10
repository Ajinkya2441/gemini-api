from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
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

def is_programming_related(question: str) -> bool:
    programming_keywords = {
        'python', 'javascript', 'java', 'c++', 'code', 'programming', 'function', 
        'class', 'variable', 'loop', 'array', 'list', 'dictionary', 'api', 
        'database', 'sql', 'framework', 'library', 'debug', 'error', 'exception',
        'algorithm', 'data structure', 'git', 'html', 'css', 'react', 'node',
        'django', 'flask', 'fastapi', 'express', 'compiler', 'interpreter',
        'string', 'integer', 'boolean', 'syntax', 'backend', 'frontend',
        'development', 'coding', 'software', 'developer', 'web', 'server'
    }
    
    question_words = set(question.lower().split())
    return any(keyword in question.lower() for keyword in programming_keywords)

@app.post("/ask")
async def ask_question(query: QuestionRequest):
    question = query.question.strip()
    
    # Check if the question is programming-related
    if not is_programming_related(question):
        return JSONResponse(
            status_code=400,
            content={
                "answer": "I can only help with programming-related questions. Please ask something about coding, programming languages, software development, or related technical topics."
            }
        )

    # Use Gemini Pro with programming-focused prompt
    try:
        prompt = f"""
Please provide a concise, technical answer to the following programming question.
Focus only on programming concepts, implementation details, and code examples.
If the question is not clearly about programming, respond that you can only help with programming questions.

Question: {question}
"""
        model = genai.GenerativeModel("models/gemma-3-4b-it")
        chat = model.start_chat(history=[])
        response = chat.send_message(prompt)
        return {"answer": response.text.strip()}

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "answer": f"Error: {str(e)}. Check your Gemini API key or usage quota."
            }
        )