# Multimodal Image Generation Studio

A production-grade Python and Streamlit-based image generation workspace implementing a robust **6-stage pipeline blueprint**. The app offers a seamless web interface for creating visually striking images and supports both **Stability AI** (premium API) and **Pollinations.ai** (free, keyless alternative) as generation backends.

---

## 🚀 The 6-Stage Pipeline Blueprint

This application is built around a rigorous 6-stage architecture to ensure maximum performance, security, and quality:

1. **Stage 1: Prompt Payload Formulation**
   - Standardizes user aspect ratios (e.g. `16:9`, `1:1`, `21:9`) and dynamically maps them to optimal pixel dimensions for modern diffusion models (targeting ~1 megapixel).
2. **Stage 2: Network Gateway**
   - Applies a split-timeout policy (`connect = 3.05s` to fail fast on dead servers, `read = 60s` to allow slow inference loops to complete).
   - Utilizes exponential backoff with random jitter for connection drops and rate limits.
3. **Stage 3: Security & Moderation Gates**
   - Handles pre-generation blockages (input content policy checks) and post-generation blocks (output filters).
   - Cleanly propagates specific error messages (e.g. out of credits, blocked content) directly to the UI instead of crash looping.
4. **Stage 4: Transport Protocol**
   - Memory-safe binary chunked streaming writes directly to the disk, avoiding RAM spikes even under high-load concurrency.
5. **Stage 5: Integrity Verification**
   - Performs a complete, pixel-level PIL decode loop to check for silent download truncation instead of relying purely on fast header/CRC checks.
6. **Stage 6: Automated QA Hook**
   - Loads a **CLIP ViT-L/14 model** to score the generated assets automatically.
   - **Lens 1 (Aesthetic)**: Evaluates image aesthetics using a LAION-trained regressor (threshold: `5.0/10`).
   - **Lens 2 (Semantic)**: Verifies text-image alignment using CLIP cosine similarity (threshold: `20.0/100`).

---

## 🛠️ Installation & Setup

### Prerequisites
Make sure you have **Python 3.10+** installed on your system.

### 1. Clone & Install Dependencies
Navigate to the project root and install the required libraries:
```bash
pip install -r requirements.txt
```

### 2. Configure Environment (`.env`)
Create or edit the `.env` file in the project root:
```env
# Switch between 'pollinations' (free, keyless) and 'stability' (premium)
IMAGE_PROVIDER=pollinations

# Required only if using 'stability'
IMAGE_API_KEY=your_stability_api_key_here
```

---

## 🏃 Running the Application

### Launching the Frontend
Start the Streamlit web dashboard:
```bash
streamlit run app.py
```

### Running Moderation Tests
You can test policy/moderation gate handling by running:
```bash
python test_moderation.py
```

---

## 📂 Project Structure

- `app.py`: Streamlit-based web user interface.
- `studio.py`: Core pipeline backend implementing the 6-stage image pipeline.
- `test_moderation.py`: Diagnostic script testing moderation triggers and API response logic.
- `requirements.txt`: Project package dependencies.
- `.env`: Environment variables and configurations.
