"""
Telco Churn Prediction Service
===============================
REST API for real-time churn predictions using MLflow production model

Kullanım:
    uvicorn api.predict_service:app --reload --host 0.0.0.0 --port 8000
    
Endpoints:
    GET  /health       - Health check
    GET  /model-info   - Current production model info
    POST /predict      - Get churn prediction
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
import logging
from datetime import datetime

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Telco Churn Prediction API",
    description="MLOps-powered churn prediction service",
    version="1.0.0"
)

# CORS (frontend için gerekli olabilir)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
MODEL = None
MODEL_VERSION = None
MODEL_INFO = {}


class PredictionInput(BaseModel):
    """
    Prediction request schema - Real Telco Customer Data
    Matches the cleaned Kaggle telco churn dataset
    """
    # Customer ID
    customer_id: str = Field(..., description="Customer ID")
    
    # Demographics
    gender: str = Field(..., description="Gender (Male/Female)")
    SeniorCitizen: int = Field(..., description="Senior citizen (0/1)")
    Partner: str = Field(..., description="Has partner (Yes/No)")
    Dependents: str = Field(..., description="Has dependents (Yes/No)")
    
    # Account info
    tenure: int = Field(..., description="Months as customer", ge=0)
    MonthlyCharges: float = Field(..., description="Monthly charges ($)", ge=0)
    TotalCharges: float = Field(..., description="Total charges ($)", ge=0)
    
    # Services
    PhoneService: str = Field(..., description="Phone service (Yes/No)")
    MultipleLines: str = Field(..., description="Multiple lines")
    InternetService: str = Field(..., description="Internet service type")
    OnlineSecurity: str = Field(..., description="Online security")
    OnlineBackup: str = Field(..., description="Online backup")
    DeviceProtection: str = Field(..., description="Device protection")
    TechSupport: str = Field(..., description="Tech support")
    StreamingTV: str = Field(..., description="Streaming TV")
    StreamingMovies: str = Field(..., description="Streaming movies")
    
    # Contract
    contract_type: str = Field(..., description="Contract type")
    PaperlessBilling: str = Field(..., description="Paperless billing (Yes/No)")
    payment_method: str = Field(..., description="Payment method")
    
    # High-cardinality features (engineered)
    service_combo_id: str = Field(..., description="Service combination ID")
    geo_code: str = Field(..., description="Geographic code (G1-G50)")
    
    class Config:
        schema_extra = {
            "example": {
                "customer_id": "TEST-CUST-001",
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "contract_type": "Month-to-month",
                "PaperlessBilling": "Yes",
                "payment_method": "Electronic check",
                "MonthlyCharges": 89.85,
                "TotalCharges": 1078.20,
                "service_combo_id": "Fiber optic_Yes_Yes_Yes",
                "geo_code": "G23"
            }
        }


class PredictionOutput(BaseModel):
    """Prediction response schema"""
    customer_id: Optional[str]
    churn_probability: float = Field(..., ge=0.0, le=1.0)
    churn_prediction: int = Field(..., ge=0, le=1)
    risk_level: str
    model_version: str
    timestamp: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_version: Optional[str]
    timestamp: str


class ModelInfoResponse(BaseModel):
    """Model information response"""
    model_name: str
    model_version: str
    model_stage: str
    model_uri: str
    loaded_at: str
    metrics: Optional[Dict[str, float]]


def load_production_model():
    """
    MLflow'dan production modelini load et
    """
    global MODEL, MODEL_VERSION, MODEL_INFO
    
    try:
        logger.info("🔄 Loading production model from MLflow...")
        
        # MLflow client
        client = MlflowClient()
        model_name = "telco_churn_classifier_cleaned"  # Real Kaggle data model
        
        # Production modelini getir
        try:
            prod_versions = client.get_latest_versions(model_name, stages=["Production"])
            
            if not prod_versions:
                logger.warning("⚠️ No production model found, trying Staging...")
                prod_versions = client.get_latest_versions(model_name, stages=["Staging"])
                
                if not prod_versions:
                    logger.error("❌ No model in Production or Staging!")
                    return False
            
            model_version = prod_versions[0]
            MODEL_VERSION = model_version.version
            
            # Model URI
            model_uri = f"models:/{model_name}/{MODEL_VERSION}"
            
            # Model'i yükle - sklearn.load_model kullan (predict_proba için)
            MODEL = mlflow.sklearn.load_model(model_uri)
            
            # Model bilgilerini sakla
            run = client.get_run(model_version.run_id)
            MODEL_INFO = {
                "model_name": model_name,
                "model_version": str(MODEL_VERSION),  # Convert to string
                "model_stage": model_version.current_stage,
                "model_uri": model_uri,
                "loaded_at": datetime.now().isoformat(),
                "metrics": run.data.metrics
            }
            
            logger.info(f"✅ Model loaded successfully: v{str(MODEL_VERSION)} ({model_version.current_stage})")
            
            # Try to get metrics (both old and new naming)
            roc_auc = (
                run.data.metrics.get('test_roc_auc') or 
                run.data.metrics.get('roc_auc') or 
                run.data.metrics.get('val_roc_auc')
            )
            
            if roc_auc:
                logger.info(f"   ROC-AUC: {roc_auc:.4f}")
            else:
                logger.info(f"   ROC-AUC: N/A")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            return False
            
    except Exception as e:
        logger.error(f"❌ MLflow connection error: {e}")
        return False


@app.on_event("startup")
async def startup_event():
    """App başlangıcında model yükle"""
    logger.info("🚀 Starting Churn Prediction API...")
    
    success = load_production_model()
    
    if not success:
        logger.warning("⚠️ Model could not be loaded at startup!")
        logger.warning("   API will return 503 for predictions until model is loaded")
    else:
        logger.info("✅ API ready to serve predictions")


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Telco Churn Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "model_info": "/model-info",
            "predict": "/predict (POST)"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint
    """
    return HealthResponse(
        status="healthy" if MODEL is not None else "unhealthy",
        model_loaded=MODEL is not None,
        model_version=str(MODEL_VERSION) if MODEL_VERSION else None,  # Convert to string
        timestamp=datetime.now().isoformat()
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info():
    """
    Production model bilgileri
    """
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check MLflow connection."
        )
    
    return ModelInfoResponse(**MODEL_INFO)


@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
async def predict_churn(input_data: PredictionInput):
    """
    Churn tahmini yap
    
    Returns:
        - churn_probability: [0, 1] arası olasılık
        - churn_prediction: 0 (no churn) veya 1 (churn)
        - risk_level: LOW, MEDIUM, HIGH
    """
    
    # Model yüklü mü kontrol et
    if MODEL is None:
        # Tekrar yüklemeyi dene
        if not load_production_model():
            raise HTTPException(
                status_code=503,
                detail="Model not available. Please contact system administrator."
            )
    
    try:
        # Input dataframes'e çevir (model pipeline bunu bekliyor)
        # Pydantic model'den dict al
        customer_data = input_data.dict()
        
        # DataFrame oluştur (tek satır)
        input_df = pd.DataFrame([customer_data])
        
        # FEATURE CROSSES OLUŞTUR (training'deki gibi!)
        # Bu cross'lar model tarafından bekleniyor
        input_df["cross_contract_payment"] = (
            input_df["contract_type"].astype(str) + "__x__" + 
            input_df["payment_method"].astype(str)
        )
        input_df["cross_service_contract"] = (
            input_df["service_combo_id"].astype(str) + "__x__" + 
            input_df["contract_type"].astype(str)
        )
        input_df["cross_geo_contract"] = (
            input_df["geo_code"].astype(str) + "__x__" + 
            input_df["contract_type"].astype(str)
        )
        
        # Model'e gönder (pipeline preprocessing yapacak)
        prediction_proba = MODEL.predict_proba(input_df)
        
        # Churn probability (class 1)
        churn_probability = float(prediction_proba[0][1])
        
        # Binary prediction (threshold = 0.5)
        churn_prediction = 1 if churn_probability >= 0.5 else 0
        
        # Risk level
        if churn_probability < 0.3:
            risk_level = "LOW"
        elif churn_probability < 0.7:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"
        
        # Confidence (얼마나emin?)
        # 0.5'e yakınsa düşük confidence, 0 veya 1'e yakınsa yüksek
        confidence = abs(churn_probability - 0.5) * 2
        
        # Log prediction (monitoring için)
        logger.info(f"Prediction: customer_id={input_data.customer_id}, "
                   f"churn_prob={churn_probability:.4f}, risk={risk_level}")
        
        return PredictionOutput(
            customer_id=input_data.customer_id,
            churn_probability=round(churn_probability, 4),
            churn_prediction=churn_prediction,
            risk_level=risk_level,
            model_version=str(MODEL_VERSION),  # Convert to string
            timestamp=datetime.now().isoformat(),
            confidence=round(confidence, 4)
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/reload-model", tags=["Admin"])
async def reload_model():
    """
    Modeli yeniden yükle (yeni production model deploy edildiğinde)
    """
    logger.info("🔄 Reloading model...")
    
    success = load_production_model()
    
    if success:
        return {
            "status": "success",
            "message": f"Model v{MODEL_VERSION} reloaded",
            "model_info": {
                **MODEL_INFO,
                "model_version": str(MODEL_VERSION)  # Ensure string
            }
        }
    else:
        raise HTTPException(
            status_code=500,
            detail="Model reload failed"
        )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "predict_service:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )