from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import spacy

# Carrega o modelo PT (baixe no deploy)
# Sugestão: pt_core_news_lg (melhor qualidade) ou pt_core_news_md (mais leve)
MODEL_NAME = "pt_core_news_lg"
nlp = spacy.load(MODEL_NAME)

app = FastAPI(title="NER-PT")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # em produção, troque pelo domínio do seu front
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InText(BaseModel):
    text: str

@app.post("/ner")
def ner(payload: InText):
    doc = nlp(payload.text)
    # Pegamos entidades de tipos comuns para nomes próprios
    keep_labels = {"PER", "LOC", "ORG", "GPE"}  # pessoas, lugares, organizações
    names = set()
    for ent in doc.ents:
        if ent.label_ in keep_labels:
            # normaliza espaços, remove quebras, capitaliza título
            item = " ".join(str(ent.text).split())
            # não grude siglas: se for 2+ maiúsculas, mantém
            if item.isupper():
                names.add(item)
            else:
                # Title case sensível a apóstrofos/hífens simples
                parts = []
                for p in item.split(" "):
                    if "-" in p:
                        parts.append("-".join(s[:1].upper() + s[1:].lower() for s in p.split("-")))
                    else:
                        parts.append(p[:1].upper() + p[1:].lower())
                names.add(" ".join(parts))
    return {"proper_names": sorted(names)}
