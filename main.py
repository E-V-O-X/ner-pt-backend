from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import spacy

MODEL_NAME = "pt_core_news_md"   # mais leve
nlp = None                       # carrega na 1ª chamada

app = FastAPI(title="NER-PT")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InText(BaseModel):
    text: str

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/ner")
def ner(payload: InText):
    global nlp
    if nlp is None:
        nlp = spacy.load(MODEL_NAME)
    doc = nlp(payload.text)
    keep = {"PER", "LOC", "ORG", "GPE"}
    names = set()
    for ent in doc.ents:
        if ent.label_ in keep:
            item = " ".join(str(ent.text).split())
            if item.isupper():
                names.add(item)
            else:
                parts = []
                for p in item.split(" "):
                    if "-" in p:
                        parts.append("-".join(s[:1].upper() + s[1:].lower() for s in p.split("-")))
                    else:
                        parts.append(p[:1].upper() + p[1:].lower())
                names.add(" ".join(parts))
    return {"proper_names": sorted(names)}
