import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.router import router

app = FastAPI(
    title="AI for Smart Home — Energy Forecasting API",
    description="API to predict next-day energy consumption using NASA POWER climate data.",
    version="1.0.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the routes from router.py
app.include_router(router)

@app.get("/")
async def root():
    return {"message": "Welcome to the AI Energy Forecasting API. Use /docs for API documentation."}

if __name__ == "__main__":
    # Run the API using Uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
