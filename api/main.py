from fastapi import FastAPI

app = FastAPI(
    title="Wine Quality Prediction API",
    description="An end-to-end API for predicting wine quality from chemical characteristics.",
    version="1.0.0"
)

@app.get("/")
def read_root():
    """Root endpoint welcoming the user."""
    return {
        "message": "Welcome to the Wine Quality Prediction API",
        "docs_url": "/docs",
        "status": "running"
    }

@app.get("/health")
def health_check():
    """Health check endpoint for monitoring."""
    return {"status": "healthy"}
