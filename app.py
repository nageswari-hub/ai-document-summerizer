
from fastapi import FastAPI, UploadFile, File
from transformers import pipeline
import PyPDF2
import io

app = FastAPI()

summarizer = pipeline(
    task="text-generation",
    model="gpt2"
)
@app.get("/")
def home():
    return {"message": "AI Document Summarizer API"}

@app.post("/summarize")
async def summarize_pdf(file: UploadFile = File(...)):
    content = await file.read()
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))

    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""

    text = text[:3000]

    summary = summarizer(text, max_length=150, min_length=50, do_sample=False)

    return {
        "summary": summary[0]["summary_text"]
    }
