from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import warnings
import os
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from src.model import inference

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    row, label = inference.predict_random_record()
    return templates.TemplateResponse("index.html", {"request": request, "row": row, "label": label})

@app.get("/refresh", response_class=HTMLResponse)
def refresh_row_and_label(request: Request):
    row, label = inference.predict_random_record()
    return templates.TemplateResponse("index.html", {"request": request, "row": row, "label": label})