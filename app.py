from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib

# Load model + vectorizer
model = joblib.load("ai_text_detector.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "prediction": None})

@app.post("/predict", response_class=HTMLResponse)
def predict(request: Request, text: str = Form(...)):
    X = tfidf.transform([text])
    prediction = model.predict(X)[0]
    return templates.TemplateResponse("index.html", {"request": request, "prediction": prediction, "text": text})
