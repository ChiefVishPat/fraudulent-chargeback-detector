"""
FastAPI application for fraud detection ML service.
Provides endpoints for training, prediction, metrics, and model management.
"""
import os
import json
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import structlog

from .data_preprocessing import FraudDataProcessor
from .models import ModelEnsemble

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Initialize FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="ML-powered fraud detection system with RandomForest, LightGBM, and XGBoost",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
processor = FraudDataProcessor()
ensemble = ModelEnsemble()
current_run_id = None
training_status = {"status": "idle", "progress": 0, "message": ""}

# Pydantic models for API
class TransactionData(BaseModel):
    TransactionID: str
    CustomerID: str
    Amount: float = Field(..., gt=0, description="Transaction amount must be positive")
    MerchantCategory: str
    TransactionTime: str = Field(..., description="ISO format datetime string")
    Location: str
    PaymentMethod: str
    AccountAge: int = Field(..., ge=0, description="Account age in days")
    PreviousTransactions: int = Field(..., ge=0, description="Number of previous transactions")

class PredictionRequest(BaseModel):
    transactions: List[TransactionData]
    threshold: Optional[float] = Field(None, ge=0, le=1, description="Custom threshold (0-1)")

class PredictionResponse(BaseModel):
    predictions: List[Dict[str, Any]]
    model_used: str
    threshold: float
    timestamp: str

class TrainingRequest(BaseModel):
    train_size: float = Field(0.6, ge=0.5, le=0.7, description="Training set proportion (50-70%)")
    val_size: float = Field(0.2, ge=0.15, le=0.25, description="Validation set proportion")
    test_size: float = Field(0.2, ge=0.15, le=0.25, description="Test set proportion")
    use_smote: bool = Field(True, description="Use SMOTE resampling for imbalanced data")
    smote_strategy: str = Field('auto', description="SMOTE sampling strategy")
    random_state: int = Field(42, description="Random seed for reproducibility")

class TrainingResponse(BaseModel):
    run_id: str
    status: str
    message: str
    best_model: Optional[str] = None
    metrics_summary: Optional[Dict] = None

class ModelInfo(BaseModel):
    available_models: List[str]
    best_model: Optional[str]
    last_trained: Optional[str]
    training_status: Dict
    feature_count: Optional[int]

# Mount static files for frontend
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/", response_class=FileResponse)
async def root():
    """Serve the frontend HTML page."""
    return FileResponse("frontend/index.html")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/train", response_model=TrainingResponse)
async def train_models(
    request: TrainingRequest,
    background_tasks: BackgroundTasks
):
    """Train fraud detection models."""
    global current_run_id, training_status
    
    try:
        # Check if training is already in progress
        if training_status["status"] == "training":
            raise HTTPException(
                status_code=409, 
                detail="Training already in progress"
            )
        
        # Generate run ID
        current_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Validate split sizes
        total_size = request.train_size + request.val_size + request.test_size
        if abs(total_size - 1.0) > 1e-6:
            raise HTTPException(
                status_code=400,
                detail=f"Split sizes must sum to 1.0, got {total_size}"
            )
        
        # Start training in background
        background_tasks.add_task(
            _train_models_background,
            current_run_id,
            request.train_size,
            request.val_size,
            request.test_size,
            request.use_smote,
            request.smote_strategy,
            request.random_state
        )
        
        logger.info("Training started", run_id=current_run_id)
        
        return TrainingResponse(
            run_id=current_run_id,
            status="training",
            message="Model training started in background"
        )
        
    except Exception as e:
        logger.error("Training request failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

async def _train_models_background(run_id: str, train_size: float, val_size: float, 
                                  test_size: float, use_smote: bool, smote_strategy: str,
                                  random_state: int):
    """Background task for model training with imbalanced data handling."""
    global training_status, ensemble, processor
    
    try:
        training_status = {"status": "training", "progress": 10, "message": "Loading and processing data..."}
        
        # Process data with three-way split and SMOTE
        data = processor.process_pipeline(
            file_path="Fraud.csv",
            train_size=train_size,
            val_size=val_size, 
            test_size=test_size,
            random_state=random_state,
            use_smote=use_smote,
            smote_strategy=smote_strategy
        )
        
        training_status = {"status": "training", "progress": 30, 
                          "message": f"Data processed with {'SMOTE' if use_smote else 'original'} sampling, training models..."}
        
        # Data is already split into train/val/test with stratification
        X_train = data['X_train']
        X_val = data['X_val'] 
        X_test = data['X_test']
        y_train = data['y_train']
        y_val = data['y_val']
        y_test = data['y_test']
        
        # SMOTE-resampled data for training (if enabled)
        X_train_resampled = data.get('X_train_resampled', X_train)
        y_train_resampled = data.get('y_train_resampled', y_train)
        
        training_status = {"status": "training", "progress": 50, "message": "Training ensemble with imbalanced data techniques..."}
        
        # Train ensemble with proper imbalanced data handling
        results = ensemble.train_all(
            X_train, y_train,
            X_val, y_val,
            X_test, y_test,
            use_resampled_data=use_smote,
            X_train_resampled=X_train_resampled,
            y_train_resampled=y_train_resampled
        )
        
        training_status = {"status": "training", "progress": 80, "message": "Saving models and results..."}
        
        # Save results
        ensemble.save_results("results", run_id)
        model_paths = ensemble.save_all_models("results")
        
        # Generate plots for all models
        for name, result in results.items():
            result['model'].plot_curves(data['X_test'], data['y_test'], 
                                      f"results/validation/{run_id}")
        
        # Create summary
        metrics_summary = {name: result['metrics'] for name, result in results.items()}
        best_model = ensemble.best_model_name
        
        # Enhanced training completion message
        best_metrics = results[best_model]['metrics']
        training_status = {
            "status": "completed",
            "progress": 100,
            "message": f"Training completed with imbalanced data techniques. Best model: {best_model} (PR AUC: {best_metrics['pr_auc']:.4f}, MCC: {best_metrics['mcc']:.4f})",
            "run_id": run_id,
            "best_model": best_model,
            "metrics_summary": metrics_summary,
            "training_details": {
                "used_smote": use_smote,
                "smote_strategy": smote_strategy if use_smote else None,
                "train_size": train_size,
                "val_size": val_size,
                "test_size": test_size
            }
        }
        
        logger.info("Training completed successfully", 
                   run_id=run_id, best_model=best_model)
        
    except Exception as e:
        training_status = {
            "status": "failed",
            "progress": 0,
            "message": f"Training failed: {str(e)}"
        }
        logger.error("Training failed", run_id=run_id, error=str(e))

@app.get("/training-status")
async def get_training_status():
    """Get current training status."""
    return training_status

@app.post("/predict", response_model=PredictionResponse)
async def predict_fraud(request: PredictionRequest):
    """Predict fraud for transactions."""
    try:
        # Check if models are trained
        if ensemble.best_model_name is None:
            raise HTTPException(
                status_code=400,
                detail="No trained models available. Please train models first."
            )
        
        predictions = []
        best_model = ensemble.get_best_model()
        threshold = request.threshold or best_model.best_threshold
        
        for transaction in request.transactions:
            # Convert to dict and process
            transaction_dict = transaction.dict()
            
            try:
                # Process single transaction
                X = processor.process_single_transaction(transaction_dict)
                
                # Get prediction
                probability = best_model.predict_proba(X)[0]
                prediction = int(probability >= threshold)
                
                predictions.append({
                    "transaction_id": transaction.TransactionID,
                    "fraud_probability": float(probability),
                    "is_fraud": bool(prediction),
                    "threshold_used": float(threshold),
                    "risk_level": _get_risk_level(probability)
                })
                
            except Exception as e:
                logger.error("Prediction failed for transaction", 
                           transaction_id=transaction.TransactionID, error=str(e))
                predictions.append({
                    "transaction_id": transaction.TransactionID,
                    "error": str(e),
                    "is_fraud": False,
                    "fraud_probability": 0.0,
                    "risk_level": "unknown"
                })
        
        return PredictionResponse(
            predictions=predictions,
            model_used=ensemble.best_model_name,
            threshold=threshold,
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Prediction request failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

def _get_risk_level(probability: float) -> str:
    """Convert probability to risk level."""
    if probability >= 0.8:
        return "very_high"
    elif probability >= 0.6:
        return "high"
    elif probability >= 0.4:
        return "medium"
    elif probability >= 0.2:
        return "low"
    else:
        return "very_low"

@app.get("/models/info", response_model=ModelInfo)
async def get_model_info():
    """Get information about available models."""
    try:
        available_models = list(ensemble.models.keys())
        best_model = ensemble.best_model_name
        feature_count = len(processor.feature_columns) if processor.feature_columns else None
        
        # Check for last training timestamp
        last_trained = None
        if current_run_id:
            last_trained = current_run_id
        
        return ModelInfo(
            available_models=available_models,
            best_model=best_model,
            last_trained=last_trained,
            training_status=training_status,
            feature_count=feature_count
        )
        
    except Exception as e:
        logger.error("Failed to get model info", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/metrics")
async def get_model_metrics(run_id: Optional[str] = None):
    """Get detailed model metrics."""
    try:
        target_run_id = run_id or current_run_id
        
        if not target_run_id:
            raise HTTPException(status_code=404, detail="No training runs found")
        
        metrics_path = f"results/validation/{target_run_id}/metrics.json"
        
        if not os.path.exists(metrics_path):
            raise HTTPException(status_code=404, detail=f"Metrics not found for run {target_run_id}")
        
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        return {
            "run_id": target_run_id,
            "metrics": metrics,
            "best_model": ensemble.best_model_name
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get metrics", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/feature-importance")
async def get_feature_importance(run_id: Optional[str] = None):
    """Get feature importance for models."""
    try:
        target_run_id = run_id or current_run_id
        
        if not target_run_id:
            raise HTTPException(status_code=404, detail="No training runs found")
        
        importance_path = f"results/validation/{target_run_id}/feature_importances.json"
        
        if not os.path.exists(importance_path):
            raise HTTPException(status_code=404, detail=f"Feature importance not found for run {target_run_id}")
        
        with open(importance_path, 'r') as f:
            importance_data = json.load(f)
        
        return {
            "run_id": target_run_id,
            "feature_importance": importance_data,
            "best_model": ensemble.best_model_name
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get feature importance", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/plots/{run_id}/{model_name}")
async def get_model_plots(run_id: str, model_name: str):
    """Get model performance plots."""
    try:
        plot_path = f"results/validation/{run_id}/{model_name.lower()}_curves.png"
        
        if not os.path.exists(plot_path):
            raise HTTPException(status_code=404, detail="Plot not found")
        
        return FileResponse(plot_path, media_type="image/png")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get plot", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/model/{model_name}")
async def download_model(model_name: str, run_id: Optional[str] = None):
    """Download trained model artifacts."""
    try:
        target_run_id = run_id or current_run_id
        
        if not target_run_id:
            raise HTTPException(status_code=404, detail="No training runs found")
        
        model_files = []
        models_dir = "results/models"
        
        for filename in os.listdir(models_dir):
            if filename.startswith(model_name.lower()) and target_run_id in filename:
                model_files.append(os.path.join(models_dir, filename))
        
        if not model_files:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found for run {target_run_id}")
        
        # Return the most recent model file
        model_file = max(model_files, key=os.path.getctime)
        
        return FileResponse(
            model_file,
            media_type="application/octet-stream",
            filename=os.path.basename(model_file)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to download model", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/runs")
async def get_training_runs():
    """Get list of all training runs."""
    try:
        runs = []
        validation_dir = "results/validation"
        
        if os.path.exists(validation_dir):
            for run_dir in os.listdir(validation_dir):
                run_path = os.path.join(validation_dir, run_dir)
                if os.path.isdir(run_path):
                    metrics_path = os.path.join(run_path, "metrics.json")
                    if os.path.exists(metrics_path):
                        try:
                            with open(metrics_path, 'r') as f:
                                metrics = json.load(f)
                            
                            # Get best model by PR AUC
                            best_model = max(metrics.keys(), key=lambda x: metrics[x].get('pr_auc', 0))
                            
                            runs.append({
                                "run_id": run_dir,
                                "timestamp": run_dir,
                                "best_model": best_model,
                                "best_pr_auc": metrics[best_model].get('pr_auc', 0),
                                "models_count": len(metrics)
                            })
                        except:
                            continue
        
        # Sort by timestamp (newest first)
        runs.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return {"runs": runs, "total": len(runs)}
        
    except Exception as e:
        logger.error("Failed to get training runs", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)