"""
FastAPI Application
===================
REST API endpoints for the Marketing Mix RL system.
Designed for agentic orchestration via n8n or other workflow tools.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from pathlib import Path
import yaml
import json
import uvicorn

from data_processor import DataProcessor
from mmm_model import MMMModel
from rl_engine import RLEngine, QLearningAgent, UCBAgent, TrainingHistory


# =============================================================================
# PYDANTIC MODELS (Request/Response Schemas)
# =============================================================================

class ProcessDataRequest(BaseModel):
    """Request to process data."""
    reload: bool = Field(default=False, description="Force reload from CSV")

class ProcessDataResponse(BaseModel):
    """Response after processing data."""
    success: bool
    n_rows: int
    date_range: Dict[str, str]
    channels: List[str]
    n_states: int
    state_dimensions: List[str]
    n_actions: int
    action_names: List[str]

class TrainMMMRequest(BaseModel):
    """Request to train MMM."""
    retrain: bool = Field(default=False, description="Force retrain even if model exists")

class TrainMMMResponse(BaseModel):
    """Response after training MMM."""
    success: bool
    metrics: Dict[str, float]
    channel_contributions: Dict[str, float]
    channel_rois: Dict[str, float]

class TrainRLRequest(BaseModel):
    """Request to train RL agents."""
    algorithm: str = Field(default="both", description="'q', 'ucb', or 'both'")
    episodes: Optional[int] = Field(default=None, description="Override episode count")

class TrainRLResponse(BaseModel):
    """Response after training RL."""
    success: bool
    q_learning: Optional[Dict[str, Any]] = None
    ucb: Optional[Dict[str, Any]] = None

class RecommendRequest(BaseModel):
    """Request for budget recommendation."""
    budget: float = Field(..., description="Total budget to allocate")
    month: int = Field(..., ge=1, le=12, description="Month (1-12)")
    algorithm: str = Field(default="compare", description="'q', 'ucb', or 'compare'")
    nps: float = Field(default=50.0, description="NPS score")
    has_promotion: int = Field(default=0, description="1 if promotion month, 0 otherwise")
    constraints: Optional[Dict[str, Dict[str, float]]] = Field(
        default=None, 
        description="Constraints like {'TV': {'min': 1000, 'max': 5000}}"
    )

class RecommendResponse(BaseModel):
    """Response with recommendation."""
    success: bool
    recommendation: Dict[str, Any]

class WhatIfRequest(BaseModel):
    """Request for what-if analysis."""
    current_allocation: Dict[str, float] = Field(..., description="Current spend by channel")
    from_channel: str = Field(..., description="Channel to take budget from")
    to_channel: str = Field(..., description="Channel to give budget to")
    amount: float = Field(..., description="Amount to move")
    context: Optional[Dict[str, float]] = Field(default=None, description="Context (nps, month, etc)")

class WhatIfResponse(BaseModel):
    """Response with what-if analysis."""
    success: bool
    analysis: Dict[str, Any]

class MarginalROIRequest(BaseModel):
    """Request for marginal ROI analysis."""
    allocation: Dict[str, float] = Field(..., description="Current spend by channel")
    context: Optional[Dict[str, float]] = Field(default=None)

class MarginalROIResponse(BaseModel):
    """Response with marginal ROIs."""
    success: bool
    marginal_rois: Dict[str, float]
    ranking: List[str]

class PredictRequest(BaseModel):
    """Request for revenue prediction."""
    allocation: Dict[str, float] = Field(..., description="Spend by channel")
    context: Optional[Dict[str, float]] = Field(default=None)

class PredictResponse(BaseModel):
    """Response with prediction."""
    success: bool
    predicted_revenue: float
    predicted_roas: float


# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(
    title="Marketing Mix RL API",
    description="API for Marketing Mix Optimization with Reinforcement Learning",
    version="1.0.0"
)

# CORS middleware for n8n and other clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
state = {
    "processor": None,
    "mmm": None,
    "rl_engine": None,
    "initialized": False
}


# =============================================================================
# INITIALIZATION
# =============================================================================

def get_processor() -> DataProcessor:
    """Get or create DataProcessor."""
    if state["processor"] is None:
        state["processor"] = DataProcessor("config.yaml")
    return state["processor"]

def get_mmm() -> MMMModel:
    """Get or create MMMModel."""
    if state["mmm"] is None:
        state["mmm"] = MMMModel("config.yaml")
    return state["mmm"]

def get_rl_engine() -> RLEngine:
    """Get or create RLEngine."""
    if state["rl_engine"] is None:
        state["rl_engine"] = RLEngine("config.yaml")
    return state["rl_engine"]

def ensure_initialized():
    """Ensure all components are initialized."""
    if not state["initialized"]:
        raise HTTPException(
            status_code=400, 
            detail="System not initialized. Call /data/process first."
        )

def initialize_on_startup():
    """Initialize all components on server startup."""
    print("\n" + "="*60)
    print("INITIALIZING MARKETING MIX RL SYSTEM")
    print("="*60)
    
    models_path = Path("models")
    
    try:
        # Step 1: Load/Process Data
        print("\n[1/4] Loading Data Processor...")
        processor = get_processor()
        
        if (models_path / "data_processor.joblib").exists():
            processor.load("data_processor.joblib")
            processor.load_data()
            processor.engineer_features()
            print("  ✓ Loaded from saved file")
        else:
            print("  → Processing raw data...")
            processor.load_data()
            processor.load_special_sales()
            processor.engineer_features()
            processor.build_state_space()
            processor.build_action_space()
            processor.save()
            print("  ✓ Processed and saved")
        
        print(f"  → States: {processor.n_states}, Actions: {processor.n_actions}")
        
        # Step 2: Load/Train MMM
        print("\n[2/4] Loading MMM Model...")
        mmm = get_mmm()
        
        if (models_path / "mmm_model.joblib").exists():
            mmm.load("mmm_model.joblib")
            print("  ✓ Loaded from saved file")
        else:
            print("  → Training MMM...")
            mmm.train(processor.processed_data)
            processor.add_roi_proportional_action(mmm.channel_rois)
            processor.save()
            mmm.save()
            print("  ✓ Trained and saved")
        
        print(f"  → R²: {mmm.metrics.get('r2', 0):.4f}")
        
        # Step 3: Load/Train RL Agents
        print("\n[3/4] Loading RL Agents...")
        rl = get_rl_engine()
        rl.set_environment(processor, mmm)
        
        q_exists = (models_path / "q_agent.joblib").exists()
        ucb_exists = (models_path / "ucb_agent.joblib").exists()
        
        if q_exists and ucb_exists:
            rl.load_agents()
            print("  ✓ Loaded Q-Learning and UCB agents")
        else:
            print("  → Training RL agents (this may take a minute)...")
            if not q_exists:
                print("    Training Q-Learning...")
                rl.train_q_learning(n_episodes=30000, verbose=False)
            if not ucb_exists:
                print("    Training UCB...")
                rl.train_ucb(n_episodes=30000, verbose=False)
            rl.load_agents()
            print("  ✓ Trained and saved")
        
        # Step 4: Mark as initialized
        state["initialized"] = True
        
        print("\n[4/4] System Ready!")
        print("="*60)
        print("✓ All components loaded and ready")
        print("="*60 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# Startup event
@app.on_event("startup")
async def startup_event():
    """Run initialization when server starts."""
    initialize_on_startup()


# =============================================================================
# ENDPOINTS: DATA
# =============================================================================

@app.post("/data/process", response_model=ProcessDataResponse)
async def process_data(request: ProcessDataRequest):
    """
    Process raw data and build state/action spaces.
    
    This is the first endpoint to call. It:
    1. Loads SecondFile.csv and SpecialSale.csv
    2. Engineers features (temporal, budget, performance)
    3. Builds state space from config
    4. Discovers action space from historical allocations
    """
    try:
        processor = get_processor()
        
        # Check if already processed and not forcing reload
        models_path = Path("models")
        processor_file = models_path / "data_processor.joblib"
        
        if processor_file.exists() and not request.reload:
            processor.load("data_processor.joblib")
            processor.load_data()
            processor.load_special_sales()
            processor.engineer_features()
        else:
            processor.load_data()
            processor.load_special_sales()
            processor.engineer_features()
            processor.build_state_space()
            processor.build_action_space()
            processor.save()
        
        state["initialized"] = True
        
        summary = processor.get_summary()
        
        return ProcessDataResponse(
            success=True,
            n_rows=summary['n_rows'],
            date_range=summary['date_range'],
            channels=summary['channels'],
            n_states=summary['n_states'],
            state_dimensions=summary['state_dimensions'],
            n_actions=summary['n_actions'],
            action_names=summary['action_names']
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/data/summary")
async def get_data_summary():
    """Get summary of processed data."""
    ensure_initialized()
    processor = get_processor()
    return {"success": True, "summary": processor.get_summary()}


@app.get("/data/actions")
async def get_actions():
    """Get list of all available actions."""
    ensure_initialized()
    processor = get_processor()
    return {"success": True, "actions": processor.actions}


@app.get("/data/states")
async def get_state_config():
    """Get state space configuration."""
    ensure_initialized()
    processor = get_processor()
    return {"success": True, "state_config": processor.state_config}


# =============================================================================
# ENDPOINTS: MMM
# =============================================================================

@app.post("/mmm/train", response_model=TrainMMMResponse)
async def train_mmm(request: TrainMMMRequest):
    """
    Train the Marketing Mix Model.
    
    The MMM learns:
    - Channel effectiveness (coefficients)
    - Adstock decay rates (carryover effects)
    - Saturation curves (diminishing returns)
    """
    ensure_initialized()
    
    try:
        processor = get_processor()
        mmm = get_mmm()
        
        models_path = Path("models")
        mmm_file = models_path / "mmm_model.joblib"
        
        if mmm_file.exists() and not request.retrain:
            mmm.load("mmm_model.joblib")
        else:
            mmm.train(processor.processed_data)
            mmm.save()
            
            # Add ROI-proportional action
            if processor.n_actions > 0:
                processor.add_roi_proportional_action(mmm.channel_rois)
                processor.save()
        
        return TrainMMMResponse(
            success=True,
            metrics=mmm.metrics,
            channel_contributions=mmm.channel_contributions,
            channel_rois=mmm.channel_rois
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/mmm/summary")
async def get_mmm_summary():
    """Get MMM model summary."""
    mmm = get_mmm()
    
    if mmm.model is None:
        raise HTTPException(status_code=400, detail="MMM not trained. Call /mmm/train first.")
    
    return {"success": True, "summary": mmm.get_summary()}


@app.post("/mmm/predict", response_model=PredictResponse)
async def predict_revenue(request: PredictRequest):
    """
    Predict revenue for a given allocation.
    """
    mmm = get_mmm()
    
    if mmm.model is None:
        raise HTTPException(status_code=400, detail="MMM not trained. Call /mmm/train first.")
    
    try:
        predicted = mmm.predict(request.allocation, request.context)
        total_spend = sum(request.allocation.values())
        roas = (predicted - total_spend) / total_spend if total_spend > 0 else 0
        
        return PredictResponse(
            success=True,
            predicted_revenue=float(predicted),
            predicted_roas=float(roas)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/mmm/marginal-roi", response_model=MarginalROIResponse)
async def get_marginal_roi(request: MarginalROIRequest):
    """
    Get marginal ROI for each channel.
    
    Shows which channel would give the best return for the next dollar spent.
    """
    mmm = get_mmm()
    
    if mmm.model is None:
        raise HTTPException(status_code=400, detail="MMM not trained. Call /mmm/train first.")
    
    try:
        marginal = mmm.get_marginal_roi(request.allocation, request.context)
        ranking = sorted(marginal.keys(), key=lambda x: marginal[x], reverse=True)
        
        return MarginalROIResponse(
            success=True,
            marginal_rois=marginal,
            ranking=ranking
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/mmm/what-if", response_model=WhatIfResponse)
async def what_if_analysis(request: WhatIfRequest):
    """
    Simulate moving budget from one channel to another.
    """
    mmm = get_mmm()
    
    if mmm.model is None:
        raise HTTPException(status_code=400, detail="MMM not trained. Call /mmm/train first.")
    
    try:
        analysis = mmm.simulate_reallocation(
            current_spend=request.current_allocation,
            from_channel=request.from_channel,
            to_channel=request.to_channel,
            amount=request.amount,
            context=request.context
        )
        
        return WhatIfResponse(success=True, analysis=analysis)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# ENDPOINTS: RL
# =============================================================================

@app.post("/rl/train", response_model=TrainRLResponse)
async def train_rl(request: TrainRLRequest):
    """
    Train RL agents (Q-Learning and/or UCB).
    
    - Q-Learning: Value-based RL with epsilon-greedy exploration
    - UCB: Contextual bandit with Upper Confidence Bound exploration
    """
    ensure_initialized()
    
    try:
        processor = get_processor()
        mmm = get_mmm()
        rl = get_rl_engine()
        
        # Ensure MMM is trained
        if mmm.model is None:
            mmm_file = Path("models") / "mmm_model.joblib"
            if mmm_file.exists():
                mmm.load("mmm_model.joblib")
            else:
                raise HTTPException(
                    status_code=400, 
                    detail="MMM not trained. Call /mmm/train first."
                )
        
        rl.set_environment(processor, mmm)
        
        response = TrainRLResponse(success=True)
        
        if request.algorithm in ['q', 'both']:
            q_summary = rl.train_q_learning(
                n_episodes=request.episodes,
                verbose=True
            )
            response.q_learning = q_summary
        
        if request.algorithm in ['ucb', 'both']:
            ucb_summary = rl.train_ucb(
                n_episodes=request.episodes,
                verbose=True
            )
            response.ucb = ucb_summary
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/rl/policy/{algorithm}")
async def get_policy(algorithm: str):
    """Get learned policy summary for Q-Learning or UCB."""
    rl = get_rl_engine()
    
    try:
        # Try to load agents if not in memory
        processor = get_processor()
        mmm = get_mmm()
        
        if processor.n_actions == 0:
            processor.load("data_processor.joblib")
        
        rl.set_environment(processor, mmm)
        rl.load_agents()
        
        summary = rl.get_policy_summary(algorithm)
        
        if 'error' in summary:
            raise HTTPException(status_code=400, detail=summary['error'])
        
        return {"success": True, "policy": summary}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# ENDPOINTS: RECOMMEND
# =============================================================================

@app.post("/recommend", response_model=RecommendResponse)
async def get_recommendation(request: RecommendRequest):
    """
    Get budget allocation recommendation.
    
    Uses trained RL policies to recommend optimal channel allocation
    based on budget, month, NPS, and promotion status.
    """
    try:
        processor = get_processor()
        mmm = get_mmm()
        rl = get_rl_engine()
        
        # Load components if needed
        if processor.n_actions == 0:
            processor.load("data_processor.joblib")
            processor.load_data()
            processor.engineer_features()
        
        if mmm.model is None:
            mmm.load("mmm_model.joblib")
        
        rl.set_environment(processor, mmm)
        rl.load_agents()
        
        recommendation = rl.recommend(
            budget=request.budget,
            month=request.month,
            algorithm=request.algorithm,
            nps=request.nps,
            has_promotion=request.has_promotion,
            constraints=request.constraints
        )
        
        return RecommendResponse(success=True, recommendation=recommendation)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recommend/compare")
async def compare_recommendations(request: RecommendRequest):
    """
    Compare Q-Learning vs UCB recommendations side by side.
    """
    request.algorithm = "compare"
    return await get_recommendation(request)


# =============================================================================
# ENDPOINTS: PIPELINE (Full Workflow)
# =============================================================================

@app.post("/pipeline/run")
async def run_full_pipeline(
    retrain_mmm: bool = False,
    retrain_rl: bool = False,
    rl_episodes: Optional[int] = None
):
    """
    Run the full pipeline: Data → MMM → RL.
    
    This is a convenience endpoint that runs all steps in sequence.
    """
    results = {"steps": []}
    
    try:
        # Step 1: Process Data
        data_result = await process_data(ProcessDataRequest(reload=False))
        results["steps"].append({
            "step": "data_processing",
            "success": True,
            "n_rows": data_result.n_rows,
            "n_states": data_result.n_states,
            "n_actions": data_result.n_actions
        })
        
        # Step 2: Train MMM
        mmm_result = await train_mmm(TrainMMMRequest(retrain=retrain_mmm))
        results["steps"].append({
            "step": "mmm_training",
            "success": True,
            "r2": mmm_result.metrics.get('r2', 0)
        })
        
        # Step 3: Train RL
        rl_result = await train_rl(TrainRLRequest(
            algorithm="both",
            episodes=rl_episodes
        ))
        results["steps"].append({
            "step": "rl_training",
            "success": True,
            "q_learning_mean_reward": rl_result.q_learning.get('mean_reward') if rl_result.q_learning else None,
            "ucb_mean_reward": rl_result.ucb.get('mean_reward') if rl_result.ucb else None
        })
        
        results["success"] = True
        results["message"] = "Pipeline completed successfully"
        
        return results
    
    except Exception as e:
        results["success"] = False
        results["error"] = str(e)
        return results


@app.get("/pipeline/status")
async def get_pipeline_status():
    """Get status of all pipeline components."""
    models_path = Path("models")
    
    status = {
        "data_processor": {
            "file_exists": (models_path / "data_processor.joblib").exists(),
            "loaded": state["processor"] is not None and state["processor"].n_actions > 0
        },
        "mmm": {
            "file_exists": (models_path / "mmm_model.joblib").exists(),
            "loaded": state["mmm"] is not None and state["mmm"].model is not None
        },
        "q_agent": {
            "file_exists": (models_path / "q_agent.joblib").exists(),
            "loaded": state["rl_engine"] is not None and state["rl_engine"].q_agent is not None
        },
        "ucb_agent": {
            "file_exists": (models_path / "ucb_agent.joblib").exists(),
            "loaded": state["rl_engine"] is not None and state["rl_engine"].ucb_agent is not None
        },
        "initialized": state["initialized"]
    }
    
    return {"success": True, "status": status}


# =============================================================================
# ENDPOINTS: HEALTH & INFO
# =============================================================================

@app.get("/")
async def root():
    """API root - returns basic info."""
    return {
        "name": "Marketing Mix RL API",
        "version": "1.0.0",
        "endpoints": {
            "data": ["/data/process", "/data/summary", "/data/actions", "/data/states"],
            "mmm": ["/mmm/train", "/mmm/summary", "/mmm/predict", "/mmm/marginal-roi", "/mmm/what-if"],
            "rl": ["/rl/train", "/rl/policy/{algorithm}"],
            "recommend": ["/recommend", "/recommend/compare"],
            "pipeline": ["/pipeline/run", "/pipeline/status"],
            "health": ["/health"]
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "initialized": state["initialized"]}


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    api_config = config.get('api', {})
    
    print("="*60)
    print("Starting Marketing Mix RL API")
    print("="*60)
    print(f"Host: {api_config.get('host', '0.0.0.0')}")
    print(f"Port: {api_config.get('port', 8000)}")
    print(f"Docs: http://localhost:{api_config.get('port', 8000)}/docs")
    print("="*60)
    
    uvicorn.run(
        "api:app",
        host=api_config.get('host', '0.0.0.0'),
        port=api_config.get('port', 8000),
        reload=api_config.get('reload', True)
    )