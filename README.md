# ✈️ Runway Sentinel AI
> **Autonomous Multi-Agent Runway FOD Detection & Hazard Response System**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Groq](https://img.shields.io/badge/Powered%20by-Groq%20Vision-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

Runway Sentinel AI bridges the gap between raw computer vision and autonomous airfield safety operations. By combining **Groq-powered vision inference**, a **deterministic risk engine**, **native LLM tool-calling**, and an **Aviation Command Center UI**, the system detects Foreign Object Debris (FOD) and autonomously executes protective ground actions in seconds.

---

## 🌟 Key Features

| Feature | Details |
|---|---|
| **Real-time Vision** | Frame-by-frame FOD detection via Groq Vision API (Llama-4 Scout) |
| **Deterministic Risk Engine** | Rules-based risk matrix — no black-box LLM for safety-critical scoring |
| **Agentic Tool Execution** | Native Groq tool-calling: `halt_traffic()`, `dispatch_sweep_team()`, `alert_ground_crew()` |
| **Incident Reporting** | Structured, audit-ready reports at temperature=0 |
| **Demo Mode** | `RUNWAYGUARD_MOCK=true` — runs full UI offline, no API key required |
| **Command Center UI** | Streamlit dark-mode interface with glowing action panels and live history log |

---

## 🏗 Repository Structure

```
runway-sentinel-ai/
├── app.py                        # Streamlit entrypoint — Command Center UI
├── requirements.txt              # Pinned Python dependencies
├── .env.example                  # API key template
├── .gitignore
├── .streamlit/
│   └── config.toml               # Dark theme enforcement
├── src/
│   ├── __init__.py
│   ├── config.py                 # Fail-fast config, reads .env + st.secrets
│   ├── detection.py              # Groq vision inference & response parser
│   ├── risk.py                   # Deterministic FOD risk scoring
│   ├── agent.py                  # Groq tool-calling agentic layer
│   └── reporting.py              # LLM incident report generation (temp=0)
├── tests/
│   ├── conftest.py               # Pre-sets env vars for offline CI
│   └── test_pipeline.py          # 12 fully mocked unit tests
├── LIMITATIONS.md
└── README.md
```

---

## 🚀 Deploy in 3 Steps — Streamlit Community Cloud (Free)

### Prerequisites
- GitHub account
- Groq API key from [console.groq.com](https://console.groq.com) (free tier available)

### Step 1 — Fork this repository

Click **Fork** on the top right of this page to fork `runway-sentinel-ai` to your GitHub account.

### Step 2 — Deploy to Streamlit Cloud

1. Go to **[share.streamlit.io](https://share.streamlit.io)**
2. Click **"New app"**
3. Select your forked repo, branch `main`, and set **Main file path** to `app.py`
4. Click **Advanced settings** → **Secrets** and add:

```toml
GROQ_API_KEY = "gsk_your_key_here"
```

5. Click **Deploy!** — your app will be live in ~60 seconds.

### Step 3 — Demo Mode (No API key)

To run without a Groq key (e.g., for offline demos):

```bash
# Streamlit Cloud: set this in Secrets
RUNWAYGUARD_MOCK = "true"

# Local:
RUNWAYGUARD_MOCK=true streamlit run app.py
```

---

## 💻 Local Development

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/runway-sentinel-ai.git
cd runway-sentinel-ai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure API key
cp .env.example .env
# Edit .env: GROQ_API_KEY=gsk_your_key_here

# 4. Run
streamlit run app.py

# Or in mock mode (no API key needed)
RUNWAYGUARD_MOCK=true streamlit run app.py   # Linux/macOS
$env:RUNWAYGUARD_MOCK="true"; streamlit run app.py  # PowerShell
```

---

## 🧪 Run Tests

```bash
pytest tests/ -v
```

All 12 tests run fully offline using `unittest.mock` — no Groq API key needed.

Expected output:
```
12 passed in <20s
```

---

## 🏛 Architecture

```
User Input (Image/Camera)
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│                  RUNWAYSENTINEL PIPELINE                    │
│                                                             │
│  ① Detection (detection.py)                                 │
│     Groq Vision (Llama-4 Scout) → structured FOD report    │
│                                                             │
│  ② Risk Engine (risk.py)                                    │
│     obj_weight × loc_centrality × confidence → 0–10 score  │
│     CLEAR / LOW / MEDIUM / HIGH / CRITICAL                  │
│                                                             │
│  ③ Agentic Layer (agent.py)                                 │
│     Groq tool-calling → halt_traffic() / alert_ground_crew()│
│                        / dispatch_sweep_team() / log_only() │
│                                                             │
│  ④ Reporting (reporting.py)                                 │
│     LLM incident report (temperature=0.0, auditable)       │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
   Streamlit Command Center UI (app.py)
```

---

## ⚙️ Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `GROQ_API_KEY` | Yes (or `MOCK=true`) | — | Groq API key |
| `RUNWAYGUARD_MOCK` | No | `false` | Set `true` for offline demo |
| `VISION_MODEL` | No | `meta-llama/llama-4-scout-17b-16e-instruct` | Groq vision model |
| `AGENT_MODEL` | No | `llama3-70b-8192` | Groq tool-calling model |
| `REPORT_MODEL` | No | `llama3-70b-8192` | Groq report generation model |

---

## 📄 License

MIT License — see `LICENSE`.

---

*Built for the GenAI × AgenticAI Hackathon. Runway Sentinel AI is a prototype demonstrating autonomous vision-to-action pipelines.*
