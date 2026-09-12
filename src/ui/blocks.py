"""
RunwayGuard AI — Gradio Command Center Interface.

Builds the enterprise dark-mode cockpit UI.
Separation of concerns: This is the only module that imports Gradio.
"""
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import gradio as gr

from src import config

if not config.USE_MOCK:
    from src.agent import audit_log, run_agent
    from src.detection import detect_fod
    from src.reporting import generate_report
    from src.risk import compute_risk_score

# ── Mock Scenarios for Offline / Demo Mode ────────────────────────────────────
MOCK_SCENARIOS = [
    {
        "fod_present": True,
        "object_class": "Metal Debris",
        "risk_raw": "HIGH",
        "risk_category": "CRITICAL",
        "numeric_score": 8.74,
        "location_estimate": "Touchdown Zone Center",
        "confidence": 0.94,
        "action": "halt_traffic",
        "action_reason": "High-risk metallic FOD detected on active touchdown zone. Immediate suspension of all runway operations required.",
        "action_detail": "All inbound and outbound traffic halted on RWY 27L. ATC notified.",
    },
    {
        "fod_present": True,
        "object_class": "Cone",
        "risk_raw": "MEDIUM",
        "risk_category": "MEDIUM",
        "numeric_score": 4.21,
        "location_estimate": "Runway Edge Right",
        "confidence": 0.81,
        "action": "alert_ground_crew",
        "action_reason": "Medium-risk obstruction cone detected near runway edge. Ground crew notification issued.",
        "action_detail": "Ground crew vehicle dispatched. FOD located at edge marking, priority B.",
    },
    {
        "fod_present": True,
        "object_class": "Bird",
        "risk_raw": "MEDIUM",
        "risk_category": "MEDIUM",
        "numeric_score": 3.15,
        "location_estimate": "Threshold Left",
        "confidence": 0.76,
        "action": "dispatch_sweep_team",
        "action_reason": "Wildlife intrusion detected near threshold. Bio-hazard sweep team dispatched.",
        "action_detail": "Wildlife management team en route. Runway operational pending clearance.",
    },
    {
        "fod_present": False,
        "object_class": "None",
        "risk_raw": "LOW",
        "risk_category": "CLEAR",
        "numeric_score": 0.0,
        "location_estimate": "N/A",
        "confidence": 0.98,
        "action": "log_only",
        "action_reason": "No FOD detected. Runway surface verified clear.",
        "action_detail": "Routine scan logged. No action required. Runway status: CLEAR.",
    },
]

SAMPLE_IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0c/Bujumbura_Airport_Runway.jpg/1280px-Bujumbura_Airport_Runway.jpg"

_history_rows: List[List[str]] = []


def _now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _risk_color(category: str) -> str:
    return {
        "CLEAR": "#10b981",
        "LOW": "#22d3ee",
        "MEDIUM": "#f59e0b",
        "HIGH": "#f97316",
        "CRITICAL": "#ef4444",
    }.get(category, "#94a3b8")


def _action_color(action: str) -> str:
    return {
        "halt_traffic": "#ef4444",
        "alert_ground_crew": "#f59e0b",
        "dispatch_sweep_team": "#f97316",
        "log_only": "#10b981",
    }.get(action, "#94a3b8")


def _status_html(state: str) -> str:
    color_map = {
        "IDLE": ("#10b981", "●", "idle-dot"),
        "ANALYZING": ("#f59e0b", "◉", "analyzing-dot"),
        "ALERT": ("#ef4444", "▲", "alert-dot"),
        "CLEAR": ("#10b981", "✔", "clear-dot"),
    }
    color, icon, cls = color_map.get(state, ("#94a3b8", "●", "idle-dot"))
    return (
        f'<div class="status-pill">'
        f'<span class="{cls}" style="color:{color};font-size:1.1rem;">{icon}</span>'
        f'<span style="color:{color};font-weight:700;letter-spacing:.12em;font-size:.82rem;">{state}</span>'
        f'</div>'
    )


def _build_detection_html(s: Dict[str, Any]) -> str:
    rc = _risk_color(s.get("risk_category", "CLEAR"))
    conf_pct = int(float(s.get("confidence", 0.0)) * 100)
    fod_badge = (
        f'<span class="badge" style="background:{rc}22;color:{rc};border:1px solid {rc};">FOD DETECTED</span>'
        if s.get("fod_present")
        else '<span class="badge" style="background:#10b98122;color:#10b981;border:1px solid #10b981;">RUNWAY CLEAR</span>'
    )
    return f"""
<div class="panel detection-panel">
  <div class="panel-title">&#11041; VISION DETECTION OUTPUT</div>
  <div class="panel-body">
    <div class="kv-row">{fod_badge}</div>
    <div class="kv-row">
      <span class="kv-label">OBJECT CLASS</span>
      <span class="kv-value mono">{str(s.get("object_class", "NONE")).upper()}</span>
    </div>
    <div class="kv-row">
      <span class="kv-label">RISK RATING</span>
      <span class="kv-value" style="color:{rc};font-weight:700;font-size:1rem;letter-spacing:.1em;">
        {s.get("risk_category")}
        <span class="risk-score mono" style="color:{rc}88;">({s.get("numeric_score")}/10)</span>
      </span>
    </div>
    <div class="kv-row">
      <span class="kv-label">LOCATION</span>
      <span class="kv-value mono">{str(s.get("location_estimate", "UNKNOWN")).upper()}</span>
    </div>
    <div class="kv-row">
      <span class="kv-label">CONFIDENCE</span>
      <span class="kv-value mono">{conf_pct}%</span>
    </div>
    <div class="conf-bar-bg">
      <div class="conf-bar-fill" style="width:{conf_pct}%;background:{rc};"></div>
    </div>
    <div class="kv-row" style="margin-top:6px;">
      <span class="kv-label">TIMESTAMP</span>
      <span class="kv-value mono" style="font-size:.72rem;">{_now_ts()}</span>
    </div>
  </div>
</div>"""


def _build_action_html(s: Dict[str, Any]) -> str:
    action_name = str(s.get("action", "log_only"))
    ac = _action_color(action_name)
    icon_map = {
        "halt_traffic": "🚨",
        "alert_ground_crew": "⚠️",
        "dispatch_sweep_team": "🚜",
        "log_only": "📋",
    }
    icon = icon_map.get(action_name, "⚡")
    glow = "glow-pulse-red" if action_name == "halt_traffic" else "glow-pulse-amber"
    return f"""
<div class="panel action-panel" style="--action-color:{ac};animation:{glow} 2.5s ease-in-out infinite;border-color:{ac}!important;">
  <div class="panel-title" style="color:{ac};">⚡ AGENTIC DECISION ENGINE</div>
  <div class="panel-body">
    <div class="terminal-box" style="--action-color:{ac};">
      <div class="term-line">
        <span class="term-prompt mono">$</span>
        <span class="term-cmd mono" style="color:{ac};">
          {icon}&nbsp;&nbsp;[ACTION EXECUTED]&nbsp;&nbsp;{action_name}()
        </span>
      </div>
      <div class="term-separator" style="border-color:{ac}44;"></div>
      <div class="term-block">
        <div class="term-label mono">REASON:</div>
        <div class="term-text mono">{s.get("action_reason", "Operational guideline threshold reached.")}</div>
      </div>
      <div class="term-block" style="margin-top:8px;">
        <div class="term-label mono">OUTCOME:</div>
        <div class="term-text mono" style="color:#94a3b8;">{s.get("action_detail", "Telemetry dispatched to ground control.")}</div>
      </div>
      <div class="term-block" style="margin-top:10px;">
        <div class="term-label mono">TIMESTAMP:</div>
        <div class="term-text mono" style="color:#64748b;">{_now_ts()}</div>
      </div>
    </div>
  </div>
</div>"""


def _build_report_html(s: Dict[str, Any]) -> str:
    rc = _risk_color(s.get("risk_category", "CLEAR"))
    status_icon = "🔴" if s.get("fod_present") else "🟢"
    status_text = "FOD CONFIRMED — RUNWAY COMPROMISED" if s.get("fod_present") else "RUNWAY CLEAR — NO ACTION REQUIRED"
    return f"""
<div class="panel report-panel">
  <div class="panel-title">📋 INCIDENT REPORT</div>
  <div class="panel-body report-body">
    <div class="report-section">
      <div class="report-section-title">▸ STATUS</div>
      <div class="report-section-body mono">{status_icon}&nbsp;{status_text}</div>
    </div>
    <div class="report-section">
      <div class="report-section-title">▸ DETECTION SUMMARY</div>
      <div class="report-section-body">
        Object class <span class="mono" style="color:#e2e8f0;">{s.get("object_class")}</span>
        detected at <span class="mono" style="color:#e2e8f0;">{s.get("location_estimate")}</span>
        with <span class="mono" style="color:{rc};">{int(float(s.get("confidence", 0.0))*100)}%</span> model confidence.
        Vision model raw risk rating: <span style="color:{rc};font-weight:600;">{s.get("risk_raw")}</span>.
      </div>
    </div>
    <div class="report-section">
      <div class="report-section-title">▸ RISK ASSESSMENT</div>
      <div class="report-section-body">
        Deterministic score: <span class="mono" style="color:{rc};font-weight:700;">{s.get("numeric_score")}/10.0</span>
        &mdash; Category: <span style="color:{rc};font-weight:700;">{s.get("risk_category")}</span>.
        Computed from object_weight &times; location_centrality &times; confidence. <span style="color:#475569;">(No LLM.)</span>
      </div>
    </div>
    <div class="report-section">
      <div class="report-section-title">▸ ACTION TAKEN</div>
      <div class="report-section-body mono" style="font-size:.78rem;color:#cbd5e1;">
        {s.get("action")}() &mdash; {s.get("action_detail")}
      </div>
    </div>
    <div class="report-footer mono">
      RunwayGuard AI &middot; {_now_ts()} &middot; Classification: UNCLASSIFIED
    </div>
  </div>
</div>"""


def _no_image_html() -> Tuple[str, str, str]:
    warn = """
<div class="panel" style="border-color:#f59e0b44;">
  <div class="panel-body" style="text-align:center;padding:2rem 1rem;">
    <div style="font-size:2rem;">⚠️</div>
    <div style="color:#f59e0b;font-weight:700;margin-top:8px;">NO INPUT PROVIDED</div>
    <div style="color:#64748b;font-size:.8rem;margin-top:4px;">
      Upload an image or click LOAD SAMPLE SCENARIO before initiating scan.
    </div>
  </div>
</div>"""
    return warn, warn, warn


def _pipeline_mock(image: Any, s_idx: int):
    if image is None:
        w1, w2, w3 = _no_image_html()
        return w1, w2, w3, _status_html("IDLE"), f'<div class="header-ts mono">{_now_ts()}</div>', gr.update(), s_idx

    time.sleep(1.5)
    s = MOCK_SCENARIOS[int(s_idx) % len(MOCK_SCENARIOS)]
    ts = _now_ts()

    _history_rows.insert(0, [
        ts,
        s["object_class"],
        f"{s['risk_category']} ({s['numeric_score']})",
        f"{s['action']}()",
        s["location_estimate"],
    ])

    next_idx = (int(s_idx) + 1) % len(MOCK_SCENARIOS)
    status = "ALERT" if s["fod_present"] else "CLEAR"

    return (
        _build_detection_html(s),
        _build_action_html(s),
        _build_report_html(s),
        _status_html(status),
        f'<div class="header-ts mono">{ts}</div>',
        gr.update(value=_history_rows[:8]),
        next_idx,
    )


def _pipeline_live(image: Any, s_idx: int):
    if image is None:
        w1, w2, w3 = _no_image_html()
        return w1, w2, w3, _status_html("IDLE"), f'<div class="header-ts mono">{_now_ts()}</div>', gr.update(), s_idx

    try:
        detection = detect_fod(image)
        risk = compute_risk_score(detection)
        agent_audit = run_agent(detection, risk)
        report_text = generate_report(detection, risk, agent_audit)

        primary_action = agent_audit["actions_executed"][0] if agent_audit["actions_executed"] else {
            "tool": "log_only", "args": {}, "result": {"detail": "No tool executed"}
        }

        s = {
            "fod_present": detection.get("fod_present", False),
            "object_class": detection.get("object_class", "unknown"),
            "risk_raw": detection.get("risk_raw", "LOW"),
            "risk_category": risk.get("risk_category", "CLEAR"),
            "numeric_score": risk.get("numeric_score", 0.0),
            "location_estimate": detection.get("location_estimate", "unknown"),
            "confidence": detection.get("confidence", 0.0),
            "action": primary_action.get("tool", "log_only"),
            "action_reason": primary_action.get("args", {}).get("reason", "Autonomous risk assessment threshold."),
            "action_detail": primary_action.get("result", {}).get("detail", "Operation executed."),
        }

        ts = _now_ts()
        _history_rows.insert(0, [
            ts,
            s["object_class"],
            f"{s['risk_category']} ({s['numeric_score']})",
            f"{s['action']}()",
            s["location_estimate"],
        ])

        status = "ALERT" if s["fod_present"] else "CLEAR"
        report_html = f"""
<div class="panel report-panel">
  <div class="panel-title">📋 GENERATED INCIDENT REPORT</div>
  <div class="panel-body report-body">
    <pre style="color:#cbd5e1;font-size:.72rem;white-space:pre-wrap;font-family:monospace;">{report_text}</pre>
  </div>
</div>"""

        return (
            _build_detection_html(s),
            _build_action_html(s),
            report_html,
            _status_html(status),
            f'<div class="header-ts mono">{ts}</div>',
            gr.update(value=_history_rows[:8]),
            s_idx,
        )
    except Exception as exc:
        err_box = f'<div class="panel" style="border-color:#ef4444;"><div class="panel-body" style="color:#ef4444;font-family:monospace;">ERROR: {str(exc)}</div></div>'
        return err_box, err_box, err_box, _status_html("IDLE"), f'<div class="header-ts mono">{_now_ts()}</div>', gr.update(), s_idx


_pipeline_fn = _pipeline_mock if config.USE_MOCK else _pipeline_live


# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = """
:root {
  --bg-base:#0a0f1e; --bg-surface:#0f172a; --bg-panel:#111827;
  --bg-panel2:#1a2235; --border:#1e2d45; --border-dim:#141e2e;
  --text-pri:#e2e8f0; --text-sec:#94a3b8; --text-dim:#475569;
  --green:#10b981; --amber:#f59e0b; --red:#ef4444; --cyan:#22d3ee;
  --mono:'Courier New','Fira Code',monospace;
}

body { background:var(--bg-base)!important; margin:0; padding:0; }
.gradio-container { background:var(--bg-base)!important; max-width:100%!important;
                    padding:0!important; margin:0!important; }
footer,.footer,#footer { display:none!important; }
.block { border:none!important; background:transparent!important;
         box-shadow:none!important; padding:4px!important; }
.wrap { border:none!important; box-shadow:none!important; }
.gap { gap:8px!important; }
label>span { color:var(--text-sec)!important; font-size:.72rem!important;
             text-transform:uppercase; letter-spacing:.08em; }
.gradio-container * { box-sizing:border-box; }
button { transition:all .15s ease!important; }

@keyframes glow-pulse-red {
  0%,100% { box-shadow:0 0 8px #ef444466,0 0 24px #ef444422,inset 0 0 8px #ef444408; }
  50%      { box-shadow:0 0 22px #ef4444bb,0 0 48px #ef444455,inset 0 0 18px #ef444416; }
}
@keyframes glow-pulse-amber {
  0%,100% { box-shadow:0 0 8px #f59e0b55,0 0 20px #f59e0b22; }
  50%      { box-shadow:0 0 22px #f59e0bbb,0 0 44px #f59e0b44; }
}
@keyframes blink {
  0%,100% { opacity:1; }
  50%      { opacity:.2; }
}
@keyframes fadein {
  from { opacity:0; transform:translateY(5px); }
  to   { opacity:1; transform:translateY(0); }
}

#header-bar { background:var(--bg-panel)!important; border-bottom:1px solid var(--border)!important;
              padding:0 20px!important; min-height:52px!important; align-items:center!important; }
#header-bar .block { padding:0!important; }

.status-pill { display:inline-flex; align-items:center; gap:6px; background:var(--bg-base);
               border:1px solid var(--border); border-radius:20px; padding:4px 12px;
               font-size:.78rem; font-family:var(--mono); white-space:nowrap; }
.analyzing-dot { animation:blink .8s ease-in-out infinite; }
.alert-dot     { animation:blink .4s ease-in-out infinite; }
.header-ts { font-family:var(--mono); font-size:.72rem; color:var(--text-dim); }
.logo { font-size:1.05rem; font-weight:800; color:var(--text-pri);
        letter-spacing:-.02em; display:flex; align-items:center; gap:8px; }
.logo-icon { background:var(--red); color:white; border-radius:5px;
             padding:3px 8px; font-size:.82rem; font-weight:900; letter-spacing:.05em; }
.logo-sub { color:var(--text-sec); font-size:.65rem; font-weight:400; letter-spacing:.15em; }
.header-badge { font-size:.65rem; font-family:var(--mono); border:1px solid var(--green);
                color:var(--green); border-radius:4px; padding:2px 8px; letter-spacing:.08em; }

.panel { background:var(--bg-panel); border:1px solid var(--border); border-radius:8px;
         overflow:hidden; animation:fadein .35s ease; }
.panel-title { font-size:.68rem; font-weight:700; letter-spacing:.14em; color:var(--text-sec);
               text-transform:uppercase; padding:7px 14px; background:var(--bg-panel2);
               border-bottom:1px solid var(--border); }
.panel-body { padding:11px 14px; }

.kv-row { display:flex; align-items:center; gap:10px; padding:5px 0;
          border-bottom:1px solid var(--border-dim); }
.kv-row:last-child { border-bottom:none; }
.kv-label { font-size:.63rem; font-weight:600; letter-spacing:.12em; color:var(--text-dim);
            text-transform:uppercase; min-width:108px; }
.kv-value { font-size:.82rem; color:var(--text-pri); font-weight:500; }
.risk-score { font-size:.7rem; margin-left:6px; }
.badge { font-size:.63rem; font-weight:700; letter-spacing:.1em; padding:3px 10px; border-radius:4px; }
.conf-bar-bg { background:var(--bg-base); border-radius:3px; height:4px;
               margin-top:8px; overflow:hidden; }
.conf-bar-fill { height:100%; border-radius:3px; transition:width .7s ease; }

.action-panel { border-color:var(--action-color,var(--red))!important; }
.action-panel .panel-title { border-bottom-color:var(--action-color,var(--red))!important; }
.terminal-box { background:#050a14; border-radius:6px; padding:12px 14px;
                border:1px solid #1a2540; position:relative; overflow:hidden; }
.terminal-box::before { content:''; position:absolute; top:0; left:0; right:0; height:2px;
                        background:linear-gradient(90deg,transparent,var(--action-color,var(--red)),transparent);
                        opacity:.6; }
.term-line { display:flex; align-items:baseline; gap:8px; margin-bottom:4px; }
.term-prompt { color:#475569; }
.term-cmd    { font-size:.88rem; font-weight:700; }
.term-separator { border:none; border-top:1px solid; margin:9px 0; }
.term-label { font-size:.62rem; letter-spacing:.12em; color:#475569; margin-bottom:2px; }
.term-text  { font-size:.77rem; color:#cbd5e1; line-height:1.5; }

.report-body { padding:10px 14px!important; }
.report-section { margin-bottom:9px; }
.report-section-title { font-size:.63rem; font-weight:700; letter-spacing:.12em; color:var(--cyan);
                        text-transform:uppercase; margin-bottom:3px; }
.report-section-body { font-size:.77rem; color:var(--text-sec); line-height:1.55; }
.report-footer { font-size:.6rem; color:var(--text-dim); margin-top:10px; padding-top:7px;
                 border-top:1px solid var(--border); text-align:center; }

#image-upload-area .wrap { background:var(--bg-panel)!important; border:1px dashed var(--border)!important;
                           border-radius:8px!important; }

.ctrl-btn-sample { width:100%; background:transparent!important; border:1px solid var(--cyan)!important;
                   color:var(--cyan)!important; font-size:.78rem!important; border-radius:6px!important;
                   padding:8px 0!important; cursor:pointer; letter-spacing:.08em; font-weight:600; }
.ctrl-btn-sample:hover { background:#22d3ee18!important; }

.scan-btn { width:100%; background:var(--red)!important; border:none!important; color:white!important;
            font-size:.95rem!important; font-weight:800!important; letter-spacing:.1em!important;
            border-radius:8px!important; padding:13px 0!important; cursor:pointer;
            box-shadow:0 0 18px #ef444444; }
.scan-btn:hover { box-shadow:0 0 36px #ef4444aa!important; transform:translateY(-1px)!important; }
.scan-btn:active { transform:translateY(0)!important; }

.history-table table { background:var(--bg-panel)!important; font-size:.68rem!important;
                       font-family:var(--mono)!important; color:var(--text-sec)!important;
                       border-collapse:collapse!important; width:100%; }
.history-table th { background:var(--bg-panel2)!important; color:var(--text-dim)!important;
                    font-size:.6rem!important; letter-spacing:.1em!important; padding:5px 7px!important;
                    border-bottom:1px solid var(--border)!important; text-transform:uppercase; font-weight:700; }
.history-table td { padding:4px 7px!important; border-bottom:1px solid var(--border-dim)!important;
                    color:var(--text-sec)!important; white-space:nowrap; }
.history-table tr:hover td { background:var(--bg-panel2)!important; }

.mono { font-family:var(--mono)!important; }
.section-label { font-size:.6rem; letter-spacing:.16em; color:var(--text-dim); text-transform:uppercase;
                 font-weight:700; margin-bottom:4px; display:block; }
.divider { border:none; border-top:1px solid var(--border); margin:8px 0; }
"""

PLACEHOLDER_DETECTION = """
<div class="panel detection-panel">
  <div class="panel-title">&#11041; VISION DETECTION OUTPUT</div>
  <div class="panel-body" style="text-align:center;padding:24px 14px;">
    <div style="font-size:2rem;opacity:.2;">📡</div>
    <div style="color:#475569;font-size:.77rem;margin-top:8px;font-family:'Courier New',monospace;">
      AWAITING INPUT &mdash; SYSTEM STANDBY
    </div>
  </div>
</div>"""

PLACEHOLDER_ACTION = """
<div class="panel" style="border-color:#1e2d45;">
  <div class="panel-title">⚡ AGENTIC DECISION ENGINE</div>
  <div class="panel-body">
    <div class="terminal-box" style="--action-color:#1e2d45;">
      <span class="mono" style="color:#1e2d45;font-size:.8rem;">$ AWAITING DETECTION RESULT...</span>
    </div>
  </div>
</div>"""

PLACEHOLDER_REPORT = """
<div class="panel report-panel">
  <div class="panel-title">📋 INCIDENT REPORT</div>
  <div class="panel-body" style="text-align:center;padding:28px 14px;">
    <div style="color:#1e2d45;font-size:.77rem;font-family:'Courier New',monospace;">
      NO INCIDENT DATA<br>
      <span style="color:#0a0f1e;font-size:.7rem;">Run a scan to generate a report</span>
    </div>
  </div>
</div>"""

INITIAL_HISTORY = [["—", "—", "—", "—", "—"]] * 4


def create_ui() -> gr.Blocks:
    """Instantiate and configure the Gradio Blocks command center UI."""
    with gr.Blocks(css=CSS, title="RunwayGuard AI", theme=gr.themes.Base()) as demo:

        scenario_idx = gr.State(value=0)

        # ── Header Bar ────────────────────────────────────────────────────────
        with gr.Row(elem_id="header-bar"):
            mode_badge = "MOCK / DEMO" if config.USE_MOCK else "LIVE DEPLOYED"
            mode_color = "#f59e0b" if config.USE_MOCK else "#10b981"
            gr.HTML(f"""
            <div style="display:flex;align-items:center;gap:16px;width:100%;padding:0;">
              <div class="logo">
                <span class="logo-icon">RG</span>
                <div>
                  RunwayGuard<span style="color:#ef4444;">AI</span>
                  <div class="logo-sub">FOD DETECTION &amp; RESPONSE SYSTEM</div>
                </div>
              </div>
              <div style="width:1px;height:32px;background:#1e2d45;"></div>
              <span class="header-badge" style="border-color:{mode_color};color:{mode_color};">{mode_badge}</span>
              <span class="header-badge" style="border-color:#22d3ee;color:#22d3ee;">GROQ VISION &middot; LLAMA-4</span>
              <span class="header-badge" style="border-color:#7c3aed;color:#a78bfa;">AGENTIC LAYER ACTIVE</span>
            </div>""")

            status_display = gr.HTML(value=_status_html("IDLE"))
            ts_display = gr.HTML(value=f'<div class="header-ts mono">{_now_ts()}</div>')

        # ── Main Body ─────────────────────────────────────────────────────────
        with gr.Row(equal_height=False):

            # Left Column
            with gr.Column(scale=3, min_width=260):
                gr.HTML('<span class="section-label">✈ RUNWAY IMAGE INPUT</span>')

                image_input = gr.Image(
                    type="filepath",
                    label="Upload / Webcam / Clipboard",
                    sources=["upload", "webcam", "clipboard"],
                    elem_id="image-upload-area",
                    height=210,
                )

                gr.HTML('<hr class="divider">')
                gr.HTML('<span class="section-label">DEMO SCENARIOS</span>')

                sample_btn = gr.Button(
                    "▶  LOAD SAMPLE SCENARIO",
                    elem_classes=["ctrl-btn-sample"],
                    size="sm",
                )

                gr.HTML("""
                <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:5px;margin:8px 0;">
                  <div style="background:#ef444412;border:1px solid #ef444444;border-radius:5px;padding:5px 4px;text-align:center;">
                    <div style="color:#ef4444;font-size:.62rem;font-weight:700;font-family:'Courier New',monospace;">CRIT</div>
                    <div style="color:#475569;font-size:.55rem;">Metal</div>
                  </div>
                  <div style="background:#f59e0b12;border:1px solid #f59e0b44;border-radius:5px;padding:5px 4px;text-align:center;">
                    <div style="color:#f59e0b;font-size:.62rem;font-weight:700;font-family:'Courier New',monospace;">MED</div>
                    <div style="color:#475569;font-size:.55rem;">Cone</div>
                  </div>
                  <div style="background:#f9731612;border:1px solid #f9731644;border-radius:5px;padding:5px 4px;text-align:center;">
                    <div style="color:#f97316;font-size:.62rem;font-weight:700;font-family:'Courier New',monospace;">MED</div>
                    <div style="color:#475569;font-size:.55rem;">Bird</div>
                  </div>
                  <div style="background:#10b98112;border:1px solid #10b98144;border-radius:5px;padding:5px 4px;text-align:center;">
                    <div style="color:#10b981;font-size:.62rem;font-weight:700;font-family:'Courier New',monospace;">CLR</div>
                    <div style="color:#475569;font-size:.55rem;">None</div>
                  </div>
                </div>""")

                gr.HTML('<hr class="divider">')

                scan_btn = gr.Button(
                    "⟳  INITIATE SCAN",
                    elem_classes=["scan-btn"],
                    size="lg",
                )

                gr.HTML(f"""
                <div style="margin-top:10px;padding:8px;background:#0a0f1e;border:1px solid #1e2d45;border-radius:6px;">
                  <div style="font-size:.6rem;color:#475569;font-family:'Courier New',monospace;line-height:1.8;">
                    MODE&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; {'MOCK (synthetic)' if config.USE_MOCK else 'LIVE (Groq API)'}<br>
                    PIPELINE&nbsp; CV → RISK → AGENT → REPORT<br>
                    VISION&nbsp;&nbsp;&nbsp; {config.VISION_MODEL}<br>
                    AGENT&nbsp;&nbsp;&nbsp;&nbsp; {config.AGENT_MODEL} (tool_choice=required)<br>
                    RISK&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Pure Python (zero LLM in scoring)
                  </div>
                </div>""")

            # Middle Column
            with gr.Column(scale=4, min_width=340):
                gr.HTML('<span class="section-label">🧠 AI PIPELINE OUTPUT</span>')
                detection_out = gr.HTML(value=PLACEHOLDER_DETECTION, elem_id="detection-out")
                gr.HTML('<div style="height:8px;"></div>')
                action_out = gr.HTML(value=PLACEHOLDER_ACTION, elem_id="action-out")

            # Right Column
            with gr.Column(scale=4, min_width=320):
                gr.HTML('<span class="section-label">📡 SYNTHESIS &amp; AUDIT</span>')
                report_out = gr.HTML(value=PLACEHOLDER_REPORT, elem_id="report-out")
                gr.HTML('<hr class="divider">')
                gr.HTML('<span class="section-label">📜 SCAN HISTORY LOG</span>')
                history_table = gr.Dataframe(
                    value=INITIAL_HISTORY,
                    headers=["TIMESTAMP (UTC)", "OBJECT", "RISK SCORE", "ACTION", "LOCATION"],
                    datatype=["str", "str", "str", "str", "str"],
                    interactive=False,
                    wrap=False,
                    elem_classes=["history-table"],
                    row_count=(5, "fixed"),
                    col_count=(5, "fixed"),
                )

        # ── Interactions ──────────────────────────────────────────────────────
        def on_sample_load(s_idx: int):
            return SAMPLE_IMAGE_URL, (int(s_idx) + 1) % len(MOCK_SCENARIOS)

        def on_scan_start():
            return _status_html("ANALYZING"), f'<div class="header-ts mono">{_now_ts()}</div>'

        sample_btn.click(
            fn=on_sample_load,
            inputs=[scenario_idx],
            outputs=[image_input, scenario_idx],
        )

        scan_btn.click(
            fn=on_scan_start,
            inputs=None,
            outputs=[status_display, ts_display],
            show_progress=False,
        ).then(
            fn=_pipeline_fn,
            inputs=[image_input, scenario_idx],
            outputs=[
                detection_out,
                action_out,
                report_out,
                status_display,
                ts_display,
                history_table,
                scenario_idx,
            ],
            show_progress="minimal",
        )

    return demo
