# Free Deployment Guide for AI Smart Home Energy Forecasting

This guide explains how to deploy your Smart Home Energy Forecasting project for free using various platforms.

## 📊 Project Components to Deploy
1. **FastAPI Backend** (`api/main.py`) - Serves predictions via REST API
2. **Streamlit Frontend** (`streamlit_app.py`) - User interface 
3. **Model Files** (`models/best_model.pkl`, `models/scaler.pkl`) - Trained ML/DL model
4. **Processed Data** (`data/processed/processed_energy_data.csv`) - For feature alignment
5. **NASA POWER API** - Free external data source (already integrated)

## 🆓 Best Free Deployment Options

### Option 1: Hugging Face Spaces (Recommended)
**Why**: Specifically designed for ML apps, supports both Streamlit and Gradio, free tier with persistent storage, easy model hosting.

#### Steps:
1. Create account at [huggingface.co](https://huggingface.co/join)
2. Create new Space: https://huggingface.co/spaces/new
   - Name: `your-username/smart-home-energy-forecast`
   - SDK: **Streamlit** (or Gradio if you prefer)
   - Visibility: Public (required for free tier)
3. Upload your files:
   - `streamlit_app.py` (main app)
   - `api/` folder (FastAPI code - we'll adapt it)
   - `src/` folder (core logic)
   - `models/` folder (best_model.pkl, scaler.pkl)
   - `data/processed/processed_energy_data.csv`
   - `config.py`
   - `requirements.txt`
4. Modify `streamlit_app.py` to work without separate API calls (optional)
5. Hugging Face will automatically detect requirements.txt and deploy

#### Alternative: Use Gradio on HF Spaces
- Wrap your FastAPI logic in a Gradio interface
- Single file deployment
- Example structure:
```python
import gradio as gr
from src.predict import predict_energy

def predict_wrapper(latitude, longitude, temperature, humidity, wind_speed, pressure, 
                   cloud_amount, precipitation, lag_1, lag_7, rolling_mean_7, 
                   day_of_week, month, season):
    input_data = {
        "latitude": latitude, "longitude": longitude, "temperature": temperature,
        "humidity": humidity, "wind_speed": wind_speed, "pressure": pressure,
        "cloud_amount": cloud_amount, "precipitation": precipitation,
        "lag_1": lag_1, "lag_7": lag_7, "rolling_mean_7": rolling_mean_7,
        "day_of_week": day_of_week, "month": month, "season": season
    }
    result = predict_energy(input_data)
    return f"{result['predicted_energy_consumption']:.2f} {result['unit']}\nConfidence: {result['confidence']}\nModel: {result['model_used']}"

# Create interface
iface = gr.Interface(
    fn=predict_wrapper,
    inputs=[
        gr.Number(label="Latitude", value=40.7128),
        gr.Number(label="Longitude", value=-74.0060),
        gr.Number(label="Temperature (°C)", value=25.0),
        gr.Number(label="Humidity (%)", value=60.0),
        gr.Number(label="Wind Speed (m/s)", value=4.1),
        gr.Number(label="Pressure (hPa)", value=1012.5),
        gr.Number(label="Cloud Amount (%)", value=15.0),
        gr.Number(label="Precipitation (mm/day)", value=0.0),
        gr.Number(label="Yesterday's Solar Radiation", value=14.2),
        gr.Number(label="7 Days Ago Solar Radiation", value=12.5),
        gr.Number(label="7-Day Avg Solar Radiation", value=13.1),
        gr.Dropdown(label="Day of Week", choices=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"], value="Thursday"),
        gr.Dropdown(label="Month", choices=list(range(1,13)), value=6),
        gr.Radio(label="Season", choices=["Winter","Spring","Summer","Fall"], value="Summer")
    ],
    outputs=[
        gr.Textbox(label="Prediction Result"),
        gr.Textbox(label="Confidence"),
        gr.Textbox(label="Model Used")
    ],
    title="AI Smart Home Energy Forecast",
    description="Predict daily energy consumption using NASA climate data"
)

if __name__ == "__main__":
    iface.launch()
```

### Option 2: Streamlit Community Cloud + Render.com (Separate Services)
**Why**: Streamlit Cloud is free and easy for frontend, Render.com has free tier for backend APIs.

#### Backend Deployment (Render.com):
1. Create account at [render.com](https://render.com/signup)
2. New → Web Service
3. Connect your GitHub repo (or manually deploy)
4. Build Command: `pip install -r requirements.txt`
5. Start Command: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
6. Free tier includes: 750 hours/month, 512 MB RAM

#### Frontend Deployment (Streamlit Community Cloud):
1. Create account at [streamlit.io/cloud](https://streamlit.io/cloud)
2. Connect your GitHub repo
3. Main file path: `streamlit_app.py`
4. Deploy!
5. Update `streamlit_app.py` to use your Render.com API URL:
   ```python
   API_BASE_URL = "https://your-service.onrender.com"  # From Render deployment
   ```

#### Pros:
- Truly separate services (microservices architecture)
- Each can scale independently
- Streamlit Cloud is optimized for Streamlit apps

#### Cons:
- Need to manage two services
- Free tiers may sleep after inactivity
- CORS configuration needed

### Option 3: Railway.app (All-in-One)
**Why**: Simple deployment, supports both web services and databases, generous free tier.

#### Steps:
1. Create account at [railway.app](https://railway.app/)
2. New Project → Deploy from GitHub repo
3. Railway will detect it's a Python project
4. Add two services:
   - **Web Service** for FastAPI: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
   - **Web Service** for Streamlit: `streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0`
5. Set environment variables if needed
6. Free tier includes: 500 hours/month shared across services

### Option 4: Fly.io (Advanced but Powerful)
**Why**: Great performance, free tier includes 3 small VMs, good for production-like deployments.

#### Steps:
1. Install Flyctl: `curl -L https://fly.io/install.sh | sh`
2. Sign up: `fly auth signup`
3. Launch app: `fly launch` (in project directory)
4. Choose region, configure services
5. Modify Dockerfile if needed (Fly.io uses Docker)
6. Free tier includes: 3x shared-cpu-1x 256MB VMs, 3GB persistent storage

### Option 5: Vercel + Serverless Functions (For API Only)
**Why**: Excellent frontend hosting, serverless functions for API.

#### Limitation: 
Streamlit doesn't work well on Vercel (it's not designed for serverless). Better for:
- Frontend: Simple HTML/JS or React app that calls your API
- API: Vercel serverless functions (Python)

#### Better Alternative:
Use Vercel for a simple frontend and deploy API elsewhere, or use Next.js API routes.

## 🔧 Implementation Details for Free Deployment

### 1. Handling Model Files
- **Small models (<100MB)**: Include directly in repo
- **Large models**: Use Hugging Face Model Hub (free) or cloud storage links
- In your code, download from URL if not present locally:
```python
import os
import urllib.request

def download_model_if_needed(model_url, local_path):
    if not os.path.exists(local_path):
        print(f"Downloading model from {model_url}")
        urllib.request.urlretrieve(model_url, local_path)

# Usage
BEST_MODEL_URL = "https://huggingface.co/your-username/your-model/resolve/main/best_model.pkl"
download_model_if_needed(BEST_MODEL_URL, BEST_MODEL_PATH)
```

### 2. NASA POWER API Considerations
- The API is free and doesn't require authentication
- Consider adding caching to avoid rate limits:
```python
import requests
from functools import lru_cache
import time

@lru_cache(maxsize=100)
def fetch_nasa_data(lat, lon, start_date, end_date):
    # Your existing NASA API call logic
    # Cache results for 1 hour
    time.sleep(1)  # Be respectful to the API
    return response.json()
```

### 3. Requirements.txt Adjustments
Ensure your requirements.txt includes all necessary packages:
```
pandas
numpy
scikit-learn
xgboost
lightgbm
tensorflow
fastapi
uvicorn
pydantic
requests
matplotlib
seaborn
joblib
python-dotenv
streamlit  # If deploying Streamlit
```

### 4. Port Binding for Cloud Platforms
Most platforms provide PORT environment variable:
```python
# In api/main.py
import os
port = int(os.environ.get("PORT", 8000))
uvicorn.run("api.main:app", host="0.0.0.0", port=port, reload=False)
```

### 5. Handling File Paths
Use relative paths or config-based paths:
```python
from config import BEST_MODEL_PATH, SCALER_PATH, PROCESSED_DATA_DIR
# These will work as long as files are in the repo
```

## 💰 Cost Analysis of Free Tiers

| Platform | Free Tier Limits | Sleep/Restart | Best For |
|----------|------------------|---------------|----------|
| **Hugging Face Spaces** | Unlimited hours, 5GB storage | No sleep (but may restart) | ML apps, Streamlit/Gradio |
| **Streamlit Community Cloud** | Unlimited hours | Sleeps after 1-2 days inactivity | Pure Streamlit apps |
| **Render.com** | 750 hours/month | Sleeps after 15 min inactivity | Backend APIs, web services |
| **Railway.app** | 500 hours/month shared | Sleeps after 1 hour inactivity | Full stack apps |
| **Fly.io** | 3x small VMs | Always on (but limited resources) | Production-like apps |
| **PythonAnywhere** | 1 always-on free app | Limited CPU, must renew every 3 months | Simple Python apps |

## 🚀 Recommended Deployment Path

### For Beginners: Hugging Face Spaces with Streamlit
1. Fastest to get started
2. Single deployment
3. Optimized for ML apps
4. Good community support
5. Easy model sharing

### For Learning Microservices: Separate Services
1. Backend on Render.com (FastAPI)
2. Frontend on Streamlit Community Cloud
3. Teaches API communication, CORS, service separation
4. More realistic production architecture

### For Maximum Control: Fly.io or Railway.app
1. More configuration but better performance
2. Ability to add databases, cron jobs, etc.
3. Closer to what you'd use in a professional setting

## 📝 Step-by-Step: Hugging Face Spaces Deployment

### Step 1: Prepare Your Repo
Ensure your GitHub repo has:
- All necessary source code
- `requirements.txt` with all dependencies
- Model files (if small enough) or plan to download them
- `streamlit_app.py` as the main entry point

### Step 2: Create HF Space
1. Go to https://huggingface.co/spaces/new
2. Select "Streamlit" as SDK
3. Name your space (e.g., `smart-home-energy-forecast`)
4. Click "Create Space"

### Step 3: Configure Space Settings
In your Space settings:
- Under "Files and versions", upload your files or connect to GitHub
- Ensure `streamlit_app.py` is in the root directory
- Add any necessary environment variables under "Secrets"

### Step 4: Modify for HF Spaces (if needed)
Add this to the top of your `streamlit_app.py` if deploying on HF Spaces:
```python
import os
# HF Spaces sets this environment variable
IS_HF_SPACE = os.getenv("SPACE_ID") is not None
```

### Step 5: Wait for Deployment
HF Spaces will:
1. Clone your repo (if connected to GitHub)
2. Install dependencies from requirements.txt
3. Run `streamlit run streamlit_app.py`
4. Provide you with a shareable URL

### Step 6: Test and Share
- Test all functionalities
- Share your Space URL: `https://huggingface.co/spaces/your-username/your-space-name`
- Optionally embed in other sites or share on social media

## ⚠️ Important Considerations for Free Tiers

### 1. Sleeping/Idle Time
- Most free services sleep after inactivity (5-15 minutes)
- First request after sleep may be slow ("cold start")
- Consider this in your user experience design

### 2. Limited Resources
- Free tiers typically have:
  - 256-512 MB RAM
  - Shared CPU
  - Limited storage (1-5GB)
- Your model and data should fit within these limits

### 3. Bandwidth Limits
- Some services have monthly bandwidth limits
- NASA API calls from your deployed app will count toward this
- Consider caching frequently requested locations

### 4. Data Persistence
- Filesystem may be ephemeral on restart
- Use `/tmp` for temporary files only
- For persistent data, consider external storage or include in repo

### 5. Security
- Never commit API keys or secrets to public repo
- Use environment variables/secrets provided by the platform
- NASA POWER API doesn't require auth, so you're safe there

## 🔧 Troubleshooting Free Deployments

### Common Issues:
1. **ModuleNotFoundError**: Check requirements.txt includes all imports
2. **Port already bound**: Use `os.environ.get("PORT", 8000)` 
3. **File not found**: Use relative paths or `__file__` to locate files
4. **Slow first request**: Normal for free tier cold starts
5. **Application error**: Check logs provided by the platform

### Debugging Tips:
- Most platforms provide logs in their dashboard
- Test locally with same commands as production
- Start simple, then add complexity
- Use `try/except` blocks and display errors in Streamlit for debugging

## 🎯 Sample Working Configuration

For Hugging Face Spaces with Streamlit, your repo structure should look like:
```
smart-home-energy-forecast/
├── streamlit_app.py
├── requirements.txt
├── config.py
├── src/
│   ├── data_ingestion.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── api/
│   ├── main.py
│   ├── schemas.py
│   └── router.py
├── models/
│   ├── best_model.pkl
│   └── scaler.pkl
├── data/
│   └── processed/
│       └── processed_energy_data.csv
└── README.md
```

## 📈 Monitoring and Maintenance

Even on free tiers, consider:
1. **Monthly check-in**: Ensure your app hasn't exceeded free limits
2. **Performance monitoring**: Note cold start times
3. **Usage tracking**: If your platform provides analytics
4. **Updates**: Push changes to GitHub if connected, or manually upload
5. **Community engagement**: Respond to comments if your Space is public

## 💡 Pro Tips for Free Deployment Success

1. **Optimize model size**: Use joblib compression or quantize if needed
2. **Cache aggressively**: Cache NASA API responses, model loading, etc.
3. **Minimize dependencies**: Only include what you actually use
4. **Graceful degradation**: If NASA API fails, show cached/default values
5. **Clear error messages**: Help users understand if it's a temporary issue
6. **Loading states**: Show spinners during API calls and model loading
7. **Mobile responsive**: Test your Streamlit app on different screen sizes
8. **Documentation**: Add a README to your deployed Space explaining usage

## 🚨 When to Consider Paid Options

Consider upgrading when:
1. You need >750 hours/month (Render) or >500 hours (Railway)
2. Your model/data exceeds free storage limits
3. You need custom domains or SSL certificates
4. You require GPU acceleration for inference
5. You want guaranteed uptime/no sleeping
6. You're building a commercial product

## ✅ Conclusion

Yes, you can absolutely deploy this project for free! The **Hugging Face Spaces + Streamlit** combination is particularly well-suited for your ML application and offers the best balance of ease-of-use, performance, and cost (free).

Your project is deployment-ready with:
- Modular, clean code structure
- Proper separation of concerns
- External API integration (NASA POWER)
- Saved model artifacts
- Requirements file
- README documentation

Start with one free platform, get it working, then explore others to expand your DevOps skills. The experience you gain deploying this project will be valuable for future ML engineering roles!

Remember: Free tiers are meant for learning, prototyping, and low-traffic applications. If your project gains traction or you need guaranteed performance, consider the paid tiers of these platforms or traditional cloud providers.