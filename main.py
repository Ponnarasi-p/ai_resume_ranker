from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

model = pipeline("sentiment-analysis")

class ResumeInput(BaseModel):
    text: str

@app.post("/rank-resume/")
def rank_resume(resume: ResumeInput):
    try:
        analysis = model(resume.text[:500])
        label = analysis[0]['label']
        score = analysis[0]['score']

        if label == "POSITIVE":
            rank_score = int(score * 100)
            feedback = "Great skills! You seem confident. Mention more backend tools for better impact."
        else:
            rank_score = int((1 - score) * 100)
            feedback = "Try improving technical keywords and highlight your experience clearly."

        return {
            "score": rank_score,
            "feedback": feedback
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
