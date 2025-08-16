"""
FastAPI application for fraud detection ML service.
Provides endpoints for training, prediction, metrics, and model management.
"""
import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
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

def load_latest_models():
    """Load the latest trained models if available."""
    global current_run_id, ensemble, processor, training_status
    
    try:
        runs_dir = "results/runs"
        if not os.path.exists(runs_dir):
            logger.info("No runs directory found, starting fresh")
            return False
            
        # Get all run directories
        run_dirs = [d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]
        if not run_dirs:
            logger.info("No training runs found")
            return False
            
        # Sort by directory name (which includes timestamp) to get latest
        latest_run = sorted(run_dirs)[-1]
        run_path = os.path.join(runs_dir, latest_run)
        
        # Check if models exist in this run
        models_dir = os.path.join(run_path, "models")
        if not os.path.exists(models_dir):
            logger.info(f"No models found in latest run {latest_run}")
            return False
            
        # Load the models
        logger.info(f"Loading models from run {latest_run}")
        success = ensemble.load_models(models_dir)
        
        if success:
            current_run_id = latest_run
            
            # Initialize processor with dummy data to set up feature columns
            try:
                # We need to set up the processor's feature columns
                # The easiest way is to process some dummy data to initialize the feature engineering
                from .data_preprocessing import FraudDataProcessor
                processor = FraudDataProcessor()
                
                # Load training summary to get feature info if available
                summary_path = os.path.join(run_path, "training_summary.json")
                if os.path.exists(summary_path):
                    with open(summary_path, 'r') as f:
                        summary = json.load(f)
                
                # Create minimal dummy data to initialize processor feature columns
                dummy_data = pd.DataFrame([{
                    "trans_date_trans_time": "2020-01-01 12:00:00",
                    "cc_num": 1234567890123456,
                    "merchant": "Test Store",
                    "category": "grocery_pos",
                    "amt": 50.0,
                    "first": "John",
                    "last": "Doe",
                    "gender": "M",
                    "street": "123 Test St",
                    "city": "Test City",
                    "state": "CA",
                    "zip": 12345,
                    "lat": 40.0,
                    "long": -120.0,
                    "city_pop": 100000,
                    "job": "Engineer",
                    "dob": "1990-01-01",
                    "trans_num": "test123",
                    "unix_time": 1577808000,
                    "merch_lat": 40.1,
                    "merch_long": -120.1,
                    "is_fraud": 0
                }])
                
                # Try to load the processor state from the training run
                processor_state_path = os.path.join(run_path, 'processor_state.joblib')
                if os.path.exists(processor_state_path):
                    import joblib
                    processor_state = joblib.load(processor_state_path)
                    
                    # Restore the processor state
                    processor.scaler = processor_state['scaler']
                    processor.feature_columns = processor_state['feature_columns']
                    processor.label_encoders = processor_state['label_encoders']
                    
                    logger.info(f"Loaded processor state with {len(processor.feature_columns)} features")
                else:
                    # Fallback: Initialize processor with dummy data (but don't fit scaler)
                    logger.warning("No processor state found, using fallback initialization")
                    
                    processed_dummy = processor.engineer_features(dummy_data)
                    X_dummy, y_dummy = processor.prepare_features(processed_dummy, fit_scaler=True)
                    
                    logger.warning("Using newly fitted scaler - predictions may be unreliable")
                
                logger.info(f"Processor initialized with {len(processor.feature_columns)} features")
                    
                training_status = {
                    "status": "completed", 
                    "progress": 100, 
                    "message": f"Loaded existing models from {latest_run}",
                    "run_id": latest_run,
                    "best_model": ensemble.best_model_name
                }
                return True
                    
            except Exception as e:
                logger.warning(f"Could not initialize processor: {e}")
                
            logger.info(f"Models loaded from {latest_run}, best model: {ensemble.best_model_name}")
            return True
        else:
            logger.warning(f"Failed to load models from {latest_run}")
            return False
            
    except Exception as e:
        logger.error(f"Error loading latest models: {e}")
        return False

# Pydantic models for API - Updated for credit card fraud dataset
class TransactionData(BaseModel):
    trans_date_trans_time: str = Field(..., description="Transaction datetime string")
    cc_num: int = Field(..., description="Credit card number")
    merchant: str = Field(..., description="Merchant name")
    category: str = Field(..., description="Merchant category")
    amt: float = Field(..., gt=0, description="Transaction amount must be positive")
    first: str = Field(..., description="Customer first name")
    last: str = Field(..., description="Customer last name")
    gender: str = Field(..., description="Customer gender (M/F)")
    street: str = Field(..., description="Customer street address")
    city: str = Field(..., description="Customer city")
    state: str = Field(..., description="Customer state")
    zip: int = Field(..., description="Customer zip code")
    lat: float = Field(..., description="Customer latitude")
    long: float = Field(..., description="Customer longitude")
    city_pop: int = Field(..., description="City population")
    job: str = Field(..., description="Customer job title")
    dob: str = Field(..., description="Customer date of birth")
    trans_num: str = Field(..., description="Transaction number")
    unix_time: int = Field(..., description="Unix timestamp")
    merch_lat: float = Field(..., description="Merchant latitude")
    merch_long: float = Field(..., description="Merchant longitude")

class PredictionRequest(BaseModel):
    transactions: List[TransactionData]
    threshold: Optional[float] = Field(None, ge=0, le=1, description="Custom threshold (0-1)")

class PredictionResponse(BaseModel):
    predictions: List[Dict[str, Any]]
    model_used: str
    threshold: float
    timestamp: str

class TrainingRequest(BaseModel):
    train_size: float = Field(0.6, ge=0.4, le=0.8, description="Training set proportion (40-80%)")
    val_size: float = Field(0.2, ge=0.1, le=0.4, description="Validation set proportion")
    test_size: float = Field(0.2, ge=0.1, le=0.4, description="Test set proportion")
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

# Startup event to load existing models
@app.on_event("startup")
async def startup_event():
    """Load existing models on startup if available."""
    logger.info("Starting up, checking for existing models...")
    success = load_latest_models()
    if success:
        logger.info("Successfully loaded existing models on startup")
    else:
        logger.info("No existing models found, starting fresh")

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
        
        # Process data with three-way split and SMOTE using new dataset
        data = processor.process_pipeline(
            train_path="fraud-detection/fraudTrain.csv",
            test_path="fraud-detection/fraudTest.csv",
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
        
        # Save the processor state for later loading
        processor_state = {
            'scaler': processor.scaler,
            'feature_columns': processor.feature_columns,
            'label_encoders': processor.label_encoders
        }
        run_dir = f"results/runs/{run_id}"
        # Ensure the run directory exists before saving processor state
        os.makedirs(run_dir, exist_ok=True)
        processor_path = os.path.join(run_dir, 'processor_state.joblib')
        import joblib
        joblib.dump(processor_state, processor_path)
        logger.info(f"Saved processor state to {processor_path}")
        
        # Save results in organized structure
        ensemble.save_results("results", run_id)
        model_paths = ensemble.save_all_models(run_dir)
        
        # Create run directory and generate plots for all models
        run_dir = f"results/runs/{run_id}"
        for name, result in results.items():
            result['model'].plot_curves(data['X_test'], data['y_test'], run_dir)
        
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

@app.post("/load-models")
async def load_models_endpoint():
    """Manually load the latest trained models."""
    try:
        logger.info("Manual model loading requested")
        success = load_latest_models()
        
        if success:
            return {
                "success": True,
                "message": f"Models loaded successfully from run {current_run_id}",
                "best_model": ensemble.best_model_name,
                "run_id": current_run_id
            }
        else:
            return {
                "success": False,
                "message": "No trained models found to load",
                "best_model": None,
                "run_id": None
            }
    except Exception as e:
        logger.error("Failed to load models", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to load models: {str(e)}")

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
                
                # Handle NaN probabilities
                if pd.isna(probability):
                    probability = 0.0
                    
                prediction = int(probability >= threshold)
                
                predictions.append({
                    "transaction_id": transaction.trans_num,
                    "fraud_probability": float(probability),
                    "is_fraud": bool(prediction),
                    "threshold_used": float(threshold),
                    "risk_level": _get_risk_level(probability)
                })
                
            except Exception as e:
                logger.error("Prediction failed for transaction", 
                           transaction_id=transaction.trans_num, error=str(e))
                predictions.append({
                    "transaction_id": transaction.trans_num,
                    "error": str(e),
                    "is_fraud": False,
                    "fraud_probability": 0.0,
                    "threshold_used": float(threshold),
                    "risk_level": "unknown"
                })
        
        return PredictionResponse(
            predictions=predictions,
            model_used=f"Best Model: {ensemble.best_model_name}",
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
    """Get detailed model metrics from organized structure."""
    try:
        target_run_id = run_id or current_run_id
        
        if not target_run_id:
            raise HTTPException(status_code=404, detail="No training runs found")
        
        metrics_path = f"results/runs/{target_run_id}/metrics/metrics.json"
        
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
    """Get feature importance for models from organized structure."""
    try:
        target_run_id = run_id or current_run_id
        
        if not target_run_id:
            raise HTTPException(status_code=404, detail="No training runs found")
        
        importance_path = f"results/runs/{target_run_id}/metrics/feature_importances.json"
        
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
    """Get model performance plots from organized structure."""
    try:
        plot_path = f"results/runs/{run_id}/plots/{model_name.lower()}_curves.png"
        
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
    """Download trained model artifacts from organized structure."""
    try:
        target_run_id = run_id or current_run_id
        
        if not target_run_id:
            raise HTTPException(status_code=404, detail="No training runs found")
        
        model_file = f"results/runs/{target_run_id}/models/{model_name.lower()}.joblib"
        
        if not os.path.exists(model_file):
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found for run {target_run_id}")
        
        return FileResponse(
            model_file,
            media_type="application/octet-stream",
            filename=f"{model_name.lower()}_{target_run_id}.joblib"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to download model", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/runs")
async def get_training_runs():
    """Get list of all training runs from organized structure."""
    try:
        runs = []
        runs_dir = "results/runs"
        
        if os.path.exists(runs_dir):
            for run_dir in os.listdir(runs_dir):
                run_path = os.path.join(runs_dir, run_dir)
                if os.path.isdir(run_path):
                    # Check for training summary first
                    summary_path = os.path.join(run_path, "training_summary.json")
                    if os.path.exists(summary_path):
                        try:
                            with open(summary_path, 'r') as f:
                                summary = json.load(f)
                            
                            runs.append({
                                "run_id": run_dir,
                                "timestamp": summary.get('timestamp', run_dir),
                                "best_model": summary.get('best_model'),
                                "best_pr_auc": summary.get('best_model_metrics', {}).get('pr_auc', 0),
                                "models_count": len(summary.get('models_trained', [])),
                                "training_completed": True
                            })
                        except:
                            # Fallback to old method if summary doesn't exist
                            metrics_path = os.path.join(run_path, "metrics", "metrics.json")
                            if os.path.exists(metrics_path):
                                try:
                                    with open(metrics_path, 'r') as f:
                                        metrics = json.load(f)
                                    
                                    best_model = max(metrics.keys(), key=lambda x: metrics[x].get('pr_auc', 0))
                                    
                                    runs.append({
                                        "run_id": run_dir,
                                        "timestamp": run_dir,
                                        "best_model": best_model,
                                        "best_pr_auc": metrics[best_model].get('pr_auc', 0),
                                        "models_count": len(metrics),
                                        "training_completed": True
                                    })
                                except:
                                    continue
        
        # Sort by timestamp (newest first)
        runs.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return {"runs": runs, "total": len(runs)}
        
    except Exception as e:
        logger.error("Failed to get training runs", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/test-data/samples")
async def get_test_data_samples(
    fraud_only: Optional[bool] = None,
    limit: int = Query(50, ge=1, le=500, description="Number of samples to return (1-500)"),
    offset: int = Query(0, ge=0, description="Number of samples to skip"),
    include_predictions: bool = Query(False, description="Include model predictions for each transaction")
):
    """Get sample transactions from test dataset for prediction testing."""
    try:
        # Load the cleaned test dataset
        test_data_path = "fraud-detection/fraudTest_cleaned.csv"
        
        if not os.path.exists(test_data_path):
            raise HTTPException(status_code=404, detail="Test dataset not found")
        
        # Read the CSV file
        df = pd.read_csv(test_data_path)
        
        # Filter by fraud status if requested
        if fraud_only is not None:
            df = df[df['is_fraud'] == (1 if fraud_only else 0)]
        
        # Apply pagination
        total_count = len(df)
        df_paginated = df.iloc[offset:offset+limit].copy()
        
        # Convert to list of dictionaries
        transactions = []
        for idx, row in df_paginated.iterrows():
            transaction = {
                "index": int(idx),
                "trans_date_trans_time": str(row['trans_date_trans_time']),
                "cc_num": int(row['cc_num']),
                "merchant": str(row['merchant']),
                "category": str(row['category']),
                "amt": float(row['amt']),
                "first": str(row['first']),
                "last": str(row['last']),
                "gender": str(row['gender']),
                "street": str(row['street']),
                "city": str(row['city']),
                "state": str(row['state']),
                "zip": int(row['zip']),
                "lat": float(row['lat']),
                "long": float(row['long']),
                "city_pop": int(row['city_pop']),
                "job": str(row['job']),
                "dob": str(row['dob']),
                "trans_num": str(row['trans_num']),
                "unix_time": int(row['unix_time']),
                "merch_lat": float(row['merch_lat']),
                "merch_long": float(row['merch_long']),
                "actual_fraud": bool(row['is_fraud']),
                "risk_indicators": _get_transaction_risk_indicators(row)
            }
            
            # Add model prediction if requested and models are available
            if include_predictions and ensemble.best_model_name is not None:
                try:
                    # Create a copy without the actual_fraud and risk_indicators for prediction
                    pred_data = {k: v for k, v in transaction.items() 
                               if k not in ['actual_fraud', 'risk_indicators', 'index']}
                    
                    X = processor.process_single_transaction(pred_data)
                    best_model = ensemble.get_best_model()
                    probability = best_model.predict_proba(X)[0]
                    threshold = best_model.best_threshold
                    
                    if pd.isna(probability):
                        probability = 0.0
                        
                    transaction["model_prediction"] = {
                        "fraud_probability": float(probability),
                        "is_fraud": bool(probability >= threshold),
                        "risk_level": _get_risk_level(probability),
                        "threshold_used": float(threshold),
                        "model_used": ensemble.best_model_name
                    }
                except Exception as e:
                    logger.warning(f"Failed to get prediction for transaction {transaction['trans_num']}: {e}")
                    transaction["model_prediction"] = None
            
            transactions.append(transaction)
        
        # Calculate statistics
        fraud_count = len(df[df['is_fraud'] == 1]) if fraud_only is None else (len(df_paginated) if fraud_only else 0)
        legitimate_count = len(df[df['is_fraud'] == 0]) if fraud_only is None else (len(df_paginated) if not fraud_only else 0)
        
        return {
            "transactions": transactions,
            "pagination": {
                "total_count": total_count,
                "returned_count": len(transactions),
                "offset": offset,
                "limit": limit,
                "has_more": offset + limit < total_count
            },
            "statistics": {
                "total_transactions": int(total_count),
                "fraud_transactions": int(fraud_count),
                "legitimate_transactions": int(legitimate_count),
                "fraud_rate": float(fraud_count / total_count) if total_count > 0 else 0
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get test data samples", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to load test data: {str(e)}")

def _get_transaction_risk_indicators(row) -> List[str]:
    """Get risk indicators for a transaction based on fraud patterns."""
    indicators = []
    
    # Amount-based indicators
    if row['amt'] > 400:
        indicators.append(f"High amount: ${row['amt']:.2f}")
    if row['amt'] > 900:
        indicators.append("Very high amount (>$900)")
    
    # Time-based indicators
    try:
        from datetime import datetime
        dt = datetime.strptime(str(row['trans_date_trans_time']), '%Y-%m-%d %H:%M:%S')
        hour = dt.hour
        
        if hour >= 22 or hour <= 3:
            indicators.append(f"Late night transaction ({hour:02d}:00)")
        if hour >= 23 or hour <= 2:
            indicators.append("Peak fraud hours (23:00-02:00)")
    except:
        pass
    
    # Category-based indicators
    high_risk_categories = ['shopping_net', 'misc_net', 'grocery_pos']
    if row['category'] in high_risk_categories:
        indicators.append(f"High-risk category: {row['category']}")
    
    # Distance-based indicators (simplified calculation)
    try:
        from math import radians, cos, sin, asin, sqrt
        lat1, lon1 = radians(row['lat']), radians(row['long'])
        lat2, lon2 = radians(row['merch_lat']), radians(row['merch_long'])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        distance_km = 2 * asin(sqrt(a)) * 6371  # Earth's radius in km
        
        if distance_km > 100:
            indicators.append(f"Distant merchant: {distance_km:.0f}km away")
        if distance_km > 500:
            indicators.append("Very distant merchant (>500km)")
    except:
        pass
    
    # Age-based indicators
    try:
        from datetime import datetime
        dob = datetime.strptime(str(row['dob']), '%Y-%m-%d')
        trans_date = datetime.strptime(str(row['trans_date_trans_time']), '%Y-%m-%d %H:%M:%S')
        age = (trans_date - dob).days // 365
        
        if age < 25:
            indicators.append(f"Young customer: {age} years old")
    except:
        pass
    
    if not indicators:
        indicators.append("Low risk profile")
    
    return indicators

# Main execution is now handled by main.py in project root