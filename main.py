from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from typing import Optional
import numpy as np

app = FastAPI(title="Banka Kredi Risk Tahmin API")

# Static klasörünü (HTML ve CSS için) bağlıyoruz
app.mount("/static", StaticFiles(directory="static"), name="static")

# Kaydettiğimiz Pipeline ve Optimum Eşik Değerini yüklüyoruz
artifact = joblib.load("credit_risk_pipeline.joblib")
pipeline = artifact["pipeline"]
optimal_threshold = artifact["threshold"]
model_name = artifact.get("model_name", "ML Model")

# Kullanıcıdan gelecek 11 verinin şeması ve doğrulama kuralları
# 1. CreditApplication sınıfında eksik olabilen 2 sütunu Optional yapıyoruz:
class CreditApplication(BaseModel):
    person_age: int = Field(..., ge=18, le=90)
    person_income: float = Field(..., gt=0)
    person_home_ownership: str
    person_emp_length: Optional[float] = Field(None, ge=0, le=60)  # Boş bırakılabilir
    loan_intent: str
    loan_grade: str
    loan_amnt: float = Field(..., gt=0)
    loan_int_rate: Optional[float] = Field(None, gt=0, le=40)      # Boş bırakılabilir
    loan_percent_income: float = Field(..., ge=0, le=1.5)
    cb_person_default_on_file: str
    cb_person_cred_hist_length: int = Field(..., ge=0, le=60)

# Ana sayfaya girildiğinde index.html ekranını aç
@app.get("/")
def read_root():
    return FileResponse("static/index.html")

# Formdan gelen veriyi tahmin eden endpoint
@app.post("/predict")
def predict_credit_risk(data: CreditApplication):
    input_df = pd.DataFrame([data.model_dump()])
    
    # Boş (None) gelen değerleri Pipeline'daki SimpleImputer'ın tanıması için np.nan'a çeviriyoruz
    input_df[["person_emp_length", "loan_int_rate"]] = input_df[["person_emp_length", "loan_int_rate"]].astype(float)
    
    prob_default = float(pipeline.predict_proba(input_df)[0, 1])
    is_default = int(prob_default >= optimal_threshold)
    
    return {
        "model_name": model_name,
        "probability": round(prob_default * 100, 2),
        "threshold": round(optimal_threshold * 100, 2),
        "prediction": is_default,
        "status": "KREDİ REDDEDİLDİ (Yüksek Temerrüt Riski)" if is_default == 1 else "KREDİ ONAYLANDI (Düşük Risk)"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)