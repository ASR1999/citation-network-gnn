from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import uvicorn
import sys
import os

# --- Import the Recommender class ---
# Ensure recommender.py and model.py are in the same directory or accessible via PYTHONPATH
try:
    from recommender import Recommender
except ImportError as e:
    print(f"Error: Could not import Recommender class. Make sure recommender.py is accessible.")
    print(f"Details: {e}")
    sys.exit(1)

# --- Define API Models ---
class RecommendationRequest(BaseModel):
    title: str = Field(..., example="Graph Attention Networks", description="Title of the query paper")
    abstract: str = Field(..., example="We present graph attention networks...", description="Abstract of the query paper")
    k: int = Field(10, gt=0, le=100, description="Number of recommendations to return (1-100)")

class RecommendationItem(BaseModel):
    node_idx: int
    title: str
    year: float # Pandas might load year as float if there are NaNs, handle appropriately
    similarity_score: float

class RecommendationResponse(BaseModel):
    recommendations: list[RecommendationItem]

# --- Initialize FastAPI App ---
app = FastAPI(
    title="Scholarly Recommender API",
    description="API to get citation recommendations based on paper title and abstract using SciBERT content similarity.",
    version="1.0.0"
)

# --- Global Recommender Instance ---
# This loads the models and data ONCE when the API starts.
recommender_instance: Recommender | None = None

@app.on_event("startup")
async def startup_event():
    """Load the Recommender model during startup."""
    global recommender_instance
    print("API Startup: Initializing Recommender...")
    try:
        recommender_instance = Recommender()
        print("API Startup: Recommender initialized successfully.")
    except Exception as e:
        print(f"API Startup Error: Failed to initialize Recommender: {e}")
        # Depending on the error, you might want the API to fail startup
        # For now, we'll allow it to start but the endpoint will fail.
        recommender_instance = None # Ensure it's None if init fails

# --- API Endpoint ---
@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """
    Accepts a paper title and abstract, returns top-k citation recommendations.
    """
    global recommender_instance
    if recommender_instance is None:
        raise HTTPException(status_code=503, detail="Recommender service is not available due to initialization failure.")

    print(f"Received recommendation request for title: '{request.title[:50]}...' with k={request.k}")

    try:
        # Call the recommend method
        recommendations_df = recommender_instance.recommend(
            query_title=request.title,
            query_abstract=request.abstract,
            k=request.k
        )

        # Check if recommendations were generated
        if recommendations_df.empty:
            print("No recommendations generated.")
            return RecommendationResponse(recommendations=[])

        # Convert DataFrame to list of dicts suitable for Pydantic model
        recommendations_list = recommendations_df.to_dict(orient='records')

        # Validate and return the response
        return RecommendationResponse(recommendations=recommendations_list)

    except Exception as e:
        print(f"Error during recommendation: {e}")
        # Log the full traceback for debugging if needed
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error during recommendation: {e}")

# --- Run the API Server ---
if __name__ == "__main__":
    print("Starting FastAPI server...")
    print("Access the API docs at http://127.0.0.1:8080/docs")
    # Use uvicorn to run the app.
    # host="0.0.0.0" makes it accessible on your network.
    # reload=True automatically restarts the server when code changes (for development).
    uvicorn.run("api:app", host="0.0.0.0", port=8080, reload=True)