# ✈️ RunwayGuard AI
> **Autonomous Multi-Agent Runway FOD Detection & Hazard Response System**

RunwayGuard AI bridges the gap between raw computer vision and autonomous airfield safety operations. By combining Groq-powered vision inference, a deterministic risk engine, native LLM tool-calling, and a command-center UI, RunwayGuard AI detects Foreign Object Debris (FOD) and automatically executes protective ground actions in seconds.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Powered by Groq](https://img.shields.io/badge/Powered%20by-Groq-f55036.svg)](https://groq.com/)
[![Built with Gradio](https://img.shields.io/badge/UI-Gradio%20Blocks-orange.svg)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests: 12 Passed](https://img.shields.io/badge/Tests-12%20Passed%20(100%25)-brightgreen.svg)]()

---

## 🌟 Key Features

* **Real-time Vision Ingestion:** Continuous frame-by-frame and snapshot FOD detection via Groq Vision API (`meta-llama/llama-4-scout-17b-16e-instruct`).
* **Deterministic Risk Engine:** Transparent, rules-based risk assessment matrix (combining object classification, location, and confidence thresholds)—no black-box hallucinations.
* **Agentic Tool Execution:** Uses native LLM tool-calling (`tool_choice="required"`) to execute actual backend functions (`halt_traffic()`, `dispatch_sweep_team()`, `alert_ground_crew()`, `log_only()`).
* **Instant Incident Reporting:** Generates structured, audit-ready incident reports capturing raw detection and verified agent response at `temperature=0.0`.
* **Tactical Command UI:** Dark-mode, high-contrast command center built with Gradio Blocks and custom CSS.

---

## 📁 Repository Branch Structure

To maintain modularity, test-driven development, and clean Git history, each core module is developed on a dedicated feature branch before merging into `main`:

* `main` — Production-ready, HF Spaces deployable codebase.
* `feature/detection-cv` — Groq vision integration & frame ingestion logic (`src/detection.py`).
* `feature/risk-engine` — Deterministic risk matrix & scoring rules (`src/risk.py`).
* `feature/agentic-layer` — Function calling & action execution tools (`src/agent.py`).
* `feature/reporting-genai` — Incident synthesis & report generator (`src/reporting.py`).
* `feature/ui-dashboard` — Gradio command center UI & custom CSS (`src/ui/blocks.py`, `app.py`).

---

## 🏛️ System Architecture

RunwayGuard AI operates as an inspectable, four-stage pipeline:

```
                  ┌──────────────────────────────┐
                  │      Runway Image Input      │
                  │   (Single Frame / Feed)      │
                  └──────────────┬───────────────┘
                                 │
                                 ▼
                  ┌──────────────────────────────┐
                  │    1. Vision Detection       │
                  │   (Groq LLaMA-4 Scout)       │
                  │   - Object classification    │
                  │   - Bounding confidence      │
                  │   - Runway zone estimation   │
                  └──────────────┬───────────────┘
                                 │  Structured Payload
                                 ▼
                  ┌──────────────────────────────┐
                  │  2. Deterministic Risk Engine │
                  │      (Pure Python Logic)     │
                  │   Score = Raw x Obj x Loc x Conf
                  │   Zero LLM - 100% Auditable  │
                  └──────────────┬───────────────┘
                                 │  Risk Score & Category
                                 ▼
                  ┌──────────────────────────────┐
                  │   3. Agentic Layer (Tools)   │
                  │   (Groq Native Tool Calling) │
                  │   tool_choice="required"     │
                  │   - halt_traffic()           │
                  │   - alert_ground_crew()      │
                  │   - dispatch_sweep_team()    │
                  │   - log_only()               │
                  └──────────────┬───────────────┘
                                 │  Execution Logs
                                 ▼
                  ┌──────────────────────────────┐
                  │     4. Incident Reporter     │
                  │  (LLaMA 70B @ Temperature 0) │
                  │  Strictly Grounded Synthesis │
                  └──────────────┬───────────────┘
                                 │
                                 ▼
                  ┌──────────────────────────────┐
                  │    Enterprise Command UI     │
                  │     (Gradio Blocks + CSS)    │
                  └──────────────────────────────┘
```

---

## 📂 Codebase Directory Layout

```text
runwayguard/
├── app.py                  # Gradio entrypoint ONLY. Zero backend logic.
├── requirements.txt        # Pinned dependencies (using opencv-python-headless).
├── README.md               # Architecture, branching, setup, and deployment guide.
├── LIMITATIONS.md          # Engineering disclosure of simulated vs. physical components.
├── .env.example            # Template for environment configuration.
├── .gitignore              # Excludes secrets, caches, and dev notebooks.
├── notebooks/              # Isolated developer notebooks (excluded from deployment).
│   ├── 01_detection_dev.ipynb
│   ├── 02_agent_dev.ipynb
│   └── 03_end_to_end_demo.ipynb
├── tests/                  # Offline pytest suite (100% mocked Groq calls).
│   ├── conftest.py
│   └── test_pipeline.py
└── src/
    ├── __init__.py
    ├── config.py           # Fail-fast environment variable validation.
    ├── detection.py        # Vision model inference and base64 encoding.
    ├── risk.py             # Deterministic, inspectable risk calculation.
    ├── agent.py            # Real Python action tools & Groq tool calling.
    ├── reporting.py        # Factual GenAI incident report synthesis.
    └── ui/
        ├── __init__.py
        └── blocks.py       # Gradio command center layout & custom CSS.
```

---

## 🛠️ Quickstart (Local Setup)

### 1. Clone the Repository & Set Up Environment

```bash
git clone https://github.com/GenAI-AgenticAI-Hackathon/runwayguard-ai.git
cd runwayguard-ai

# Create and activate virtual environment
python -m venv venv

# On Windows (PowerShell):
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install production dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Credentials

Copy the template configuration and supply your Groq API key:

```bash
cp .env.example .env
```

Edit `.env`:
```env
GROQ_API_KEY=gsk_your_actual_groq_api_key_here
RUNWAYGUARD_MOCK=false
```

> **Fail-Fast Safety Check:** If `GROQ_API_KEY` is omitted and `RUNWAYGUARD_MOCK` is not enabled, `src/config.py` halts startup immediately with a clear resolution banner.

### 3. Launch the Application

```bash
python app.py
```

Open your browser to `http://localhost:7860` to access the live command center.

### 4. Offline Synthetic Demo Mode

To run the complete interactive command center without external network calls or an API key:

```bash
# On Windows (PowerShell):
$env:RUNWAYGUARD_MOCK="true"; python app.py

# On Linux/macOS:
RUNWAYGUARD_MOCK=true python app.py
```

---

## 🤗 Deployment: Hugging Face Spaces (Zero-Friction)

RunwayGuard AI is pre-configured for instant zero-dependency deployment to Hugging Face Spaces:

### Step 1: Create a New Space
1. Navigate to [Hugging Face Spaces](https://huggingface.co/new-space).
2. Set Space Name: `runwayguard-ai`.
3. Set SDK: **Gradio** (Gradio 4+).
4. Set Hardware: **CPU Basic** (Free Tier is fully supported).

### Step 2: Add Repository Secrets
1. In your Space, navigate to **Settings** → **Variables and secrets**.
2. Under **Secrets**, add:
   - Name: `GROQ_API_KEY`
   - Value: `gsk_your_actual_groq_api_key`
3. *(Optional for demo mode)*:
   - Under **Variables**, add `RUNWAYGUARD_MOCK` = `true`.

### Step 3: Push Repository to Hugging Face
Add your Space as a remote and push:

```bash
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/runwayguard-ai
git push space main
```

> **Note:** `.gitignore` automatically prevents local `.env` secrets and `notebooks/` from being pushed to Hugging Face.

---

## 🧪 Running Offline Automated Tests

All tests run **100% offline** using `unittest.mock` to avoid consuming API credits during CI/CD:

```bash
# Run pytest with full verbosity
python -m pytest tests/ -v

# Run with test coverage report
python -m pytest tests/ --cov=src --cov-report=term-missing
```

### Verified Test Suite
```text
tests/test_pipeline.py::TestRiskAssessment::test_clear_detection_yields_zero_risk PASSED
tests/test_pipeline.py::TestRiskAssessment::test_critical_detection_yields_high_score PASSED
tests/test_pipeline.py::TestRiskAssessment::test_medium_detection_categorization PASSED
tests/test_pipeline.py::TestRiskAssessment::test_score_never_exceeds_ten PASSED
tests/test_pipeline.py::TestDetection::test_detect_fod_valid_payload PASSED
tests/test_pipeline.py::TestDetection::test_detect_fod_strips_markdown_fences PASSED
tests/test_pipeline.py::TestAgent::test_agent_executes_halt_traffic_tool PASSED
tests/test_pipeline.py::TestAgent::test_agent_fallback_mechanism PASSED
tests/test_pipeline.py::TestReporting::test_generate_report_sections PASSED
tests/test_pipeline.py::TestConfigFailFast::test_config_raises_when_api_key_missing_and_not_mock PASSED
tests/test_pipeline.py::TestConfigFailFast::test_config_succeeds_in_mock_mode_without_key PASSED
tests/test_pipeline.py::TestUIBlocks::test_create_ui_returns_blocks_instance PASSED
======================= 12 passed in 19.15s =======================
```

---

## 🛡️ Enterprise Engineering Guarantees

1. **Zero Silent Failures**: `src/config.py` validates required credentials at import time.
2. **Strict Decoupling**: Backend modules (`src/`) contain zero Gradio code; UI code lives strictly in `src/ui/`.
3. **Auditable Risk Engine**: Risk assessment is 100% deterministic Python math ($Score = Raw \times Weight_{obj} \times Weight_{loc} \times Conf$).
4. **Programmatic Action Execution**: Native Groq tool calling forces concrete Python function execution with fallback safeguards.
5. **Hallucination-Free Synthesis**: Incident reports are generated at `temperature=0.0` from verified telemetry logs only.

---

## ⚠️ Limitations & Disclosures

See [`LIMITATIONS.md`](LIMITATIONS.md) for full disclosure of simulated vs. physical components (e.g., frame-by-frame vs. real-time edge tracking, airfield radio hardware integration).

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for details.
