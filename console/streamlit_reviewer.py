"""
ComplyAgent - Streamlit Reviewer Console (v4 - navy / teal design system)

Frontend only. Agents, LangGraph workflow, risk model, screening,
document intake and audit persistence are unchanged.
"""

from html import escape
from pathlib import Path
import json
import sys

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.screening_agent import IdentityInput, load_identity_data
from agents.risk_agent import RiskEvidence
from agents.decision_agent import DecisionEvidence
from agents.document_agent import DocumentEvidence
from orchestration.graph import build_graph
from vision.document_intake import analyze_uploaded_document

st.set_page_config(
    page_title="ComplyAgent | Compliance Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def render_html(markup: str) -> None:
    """Render custom HTML (st.html when available, flattened markdown otherwise)."""
    cleaned = "\n".join(l.strip() for l in markup.splitlines() if l.strip())
    if hasattr(st, "html"):
        st.html(cleaned)
    else:
        st.markdown(cleaned, unsafe_allow_html=True)


# ============================================================
# STYLE  -  navy ink + single teal accent; status colours reserved for status
# ============================================================

render_html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,700&family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500&display=swap');
:root{--ink:#0b1b2b;--ink2:#12283d;--bg:#f4f6f9;--card:#ffffff;--line:#e3e8ef;--text:#0f1b2d;--muted:#5b6778;
--accent:#0d9488;--accent2:#2dd4bf;--ok:#16a34a;--warn:#d97706;--bad:#dc2626;}
html,body,[class*="css"],.stApp{font-family:'Inter',sans-serif;color:var(--text);color-scheme:light;}
#MainMenu,footer,[data-testid="stToolbar"],[data-testid="stDecoration"]{visibility:hidden;}
[data-testid="stHeader"]{background:transparent;}
[data-testid="stAppViewContainer"]{background:var(--bg);}
.block-container,[data-testid="stMainBlockContainer"]{padding-top:1rem;padding-bottom:3rem;padding-left:2.5rem !important;padding-right:2.5rem !important;max-width:100% !important;width:100%;}

/* ---------- sidebar (light, native widgets stay readable) ---------- */
[data-testid="stSidebar"]{background:#ffffff;border-right:1px solid var(--line);}
[data-testid="stSidebar"] label,[data-testid="stSidebar"] p{color:var(--text);}
.sb-brand{display:flex;gap:12px;align-items:center;margin:2px 0 4px;}
.sb-mark{width:40px;height:40px;border-radius:11px;display:flex;align-items:center;justify-content:center;font-size:20px;background:linear-gradient(135deg,#0b1b2b,#0d9488);}
.sb-name{font-size:20px;font-weight:800;color:var(--ink);letter-spacing:-.5px;}
.sb-sub{font-size:10.5px;color:var(--muted);letter-spacing:.6px;font-weight:600;}
.side-h{margin:24px 0 10px;font-size:11px;font-weight:700;letter-spacing:1.4px;color:var(--muted);text-transform:uppercase;}
.case-card{margin-top:12px;padding:16px;border-radius:14px;border:1px solid var(--line);border-top:3px solid var(--accent);background:#f8fafc;}
.case-tag{font-size:10px;letter-spacing:1.4px;font-weight:700;color:var(--accent);}
.case-id{font-family:'JetBrains Mono',monospace;font-size:18px;color:var(--ink);margin:8px 0 2px;}
.case-name{font-size:13.5px;color:var(--text);font-weight:600;}
.case-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:12px;padding-top:12px;border-top:1px dashed #cbd5e1;}
.case-k{font-size:10px;letter-spacing:1px;color:var(--muted);text-transform:uppercase;font-weight:700;}
.case-v{font-family:'JetBrains Mono',monospace;font-size:12px;color:var(--ink);margin-top:3px;word-break:break-all;}
.tl{margin:4px 0 0 7px;padding-left:20px;border-left:2px solid var(--line);}
.tl-item{position:relative;padding:7px 0;font-size:13.5px;color:var(--muted);font-weight:500;}
.tl-item::before{content:"";position:absolute;left:-27px;top:11px;width:10px;height:10px;border-radius:50%;background:#fff;border:2px solid #b8c4d4;}
.tl-item.on{color:var(--ink);font-weight:700;} .tl-item.on::before{border-color:var(--accent);}
.tl-item.done{color:var(--ink);} .tl-item.done::before{background:var(--accent);border-color:var(--accent);}
.tl-item small{display:block;font-size:10.5px;color:var(--muted);font-family:'JetBrains Mono',monospace;font-weight:500;}
.svc{display:flex;justify-content:space-between;gap:8px;padding:8px 2px;font-size:12.5px;color:var(--text);border-bottom:1px solid var(--line);white-space:nowrap;}
.svc span:last-child{color:#15803d;font-weight:600;}
.dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--ok);margin-right:7px;}

/* ---------- nav ---------- */
.nav{display:flex;align-items:center;justify-content:space-between;background:var(--card);border:1px solid var(--line);border-radius:14px;padding:10px 12px 10px 18px;margin-bottom:16px;box-shadow:0 1px 2px rgba(16,24,40,.04);}
.nav-l{display:flex;align-items:center;gap:10px;font-weight:800;font-size:17px;color:var(--ink);letter-spacing:-.3px;}
.nav-l i{width:26px;height:26px;border-radius:8px;background:linear-gradient(135deg,#0b1b2b,#0d9488);display:inline-flex;align-items:center;justify-content:center;font-style:normal;font-size:14px;}
.nav-m{display:flex;gap:4px;}
.nav-m span{padding:7px 14px;border-radius:8px;font-size:13px;font-weight:600;color:var(--muted);}
.nav-m span.on{background:var(--ink);color:#fff;} .nav-m span.ok{color:var(--accent);}
.nav-r{font-size:11.5px;font-weight:600;color:var(--muted);padding:7px 14px;border-radius:8px;background:#f1f5f9;border:1px solid var(--line);}

/* ---------- hero ---------- */
.hero{position:relative;overflow:hidden;border-radius:22px;padding:40px 44px 30px;color:#fff;margin-bottom:8px;
background:radial-gradient(640px 300px at 92% -10%,rgba(45,212,191,.28),transparent 60%),linear-gradient(135deg,#0b1b2b,#12283d);box-shadow:0 18px 44px rgba(11,27,43,.22);}
.hero::before{content:"";position:absolute;inset:0;opacity:.6;background-image:radial-gradient(rgba(255,255,255,.08) 1px,transparent 1px);background-size:22px 22px;}
.hero>*{position:relative;z-index:1;}
.eyebrow{display:inline-flex;gap:10px;align-items:center;font-size:12px;letter-spacing:1.8px;text-transform:uppercase;color:var(--accent2);font-weight:700;}
.eyebrow b{width:26px;height:2px;background:var(--accent2);}
.hero h1{font-family:'Fraunces',serif;font-size:46px;line-height:1.1;font-weight:700;letter-spacing:-1.2px;margin:14px 0 10px;color:#fff;padding:0;max-width:1000px;}
.hero h1 em{font-style:italic;color:#5eead4;}
.tagline{font-size:17px;font-weight:600;margin-bottom:8px;color:#cbd5e1;}
.hero p{color:#9fb2c7;font-size:14.5px;line-height:1.7;margin:0;max-width:780px;}
.chain{display:grid;grid-template-columns:repeat(6,1fr);margin-top:32px;}
.node{position:relative;padding:14px 14px 0 0;}
.node::before{content:"";position:absolute;top:6px;left:18px;right:-2px;height:2px;background:rgba(255,255,255,.16);}
.node:last-child::before{display:none;} .node.done::before{background:var(--accent2);}
.node .pt{position:relative;width:14px;height:14px;border-radius:50%;background:#0b1b2b;border:2px solid rgba(255,255,255,.35);margin-bottom:12px;z-index:1;}
.node.done .pt{background:var(--accent2);border-color:var(--accent2);box-shadow:0 0 12px rgba(45,212,191,.6);}
.node.on .pt{border-color:#fff;}
.node .nl{font-size:11px;letter-spacing:1.2px;text-transform:uppercase;color:#7f93aa;font-weight:700;}
.node .nv{font-family:'JetBrains Mono',monospace;font-size:13px;color:#fff;margin-top:4px;}
.node.done .nl{color:var(--accent2);}

/* ---------- sections / cards ---------- */
.sec{margin:36px 0 6px;display:flex;align-items:flex-end;gap:14px;}
.sec .sh small{display:block;font-size:11px;font-weight:700;letter-spacing:1.6px;color:var(--accent);margin-bottom:2px;}
.sec .sh .t{font-size:24px;font-weight:800;color:var(--ink);letter-spacing:-.5px;}
.sec i{flex:1;height:1px;background:var(--line);margin-bottom:8px;}
.sec-d{color:var(--muted);font-size:13.5px;margin:0 0 16px;}
.card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:20px 24px;box-shadow:0 1px 2px rgba(16,24,40,.04),0 4px 16px rgba(16,24,40,.04);height:100%;}
.lbl{font-size:11px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:var(--muted);margin-bottom:6px;}
.kv{display:flex;justify-content:space-between;gap:16px;padding:10px 0;border-bottom:1px solid #eef2f6;font-size:13.5px;}
.kv:last-child{border-bottom:none;} .kv span{color:var(--muted);}
.kv b{color:var(--ink);font-family:'JetBrains Mono',monospace;font-size:12.5px;text-align:right;word-break:break-all;font-weight:500;}
.note{font-size:13px;color:var(--muted);line-height:1.65;margin-top:12px;padding:12px 16px;border-radius:12px;background:#f8fafc;border:1px solid var(--line);}
.note code{font-family:'JetBrains Mono',monospace;background:#fff;border:1px solid var(--line);padding:2px 6px;border-radius:6px;}
.idcard{display:grid;grid-template-columns:230px 1fr;border-radius:18px;overflow:hidden;background:var(--card);border:1px solid var(--line);box-shadow:0 1px 2px rgba(16,24,40,.04),0 8px 24px rgba(16,24,40,.06);}
.id-side{padding:26px 20px;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:12px;background:linear-gradient(165deg,#12283d,#0b1b2b);}
.id-photo{width:92px;height:92px;border-radius:22px;display:flex;align-items:center;justify-content:center;font-size:32px;font-weight:800;background:linear-gradient(135deg,#0d9488,#5eead4);color:#0b1b2b;}
.id-tag{font-size:10px;letter-spacing:1.8px;color:var(--accent2);font-weight:700;}
.id-no{font-family:'JetBrains Mono',monospace;font-size:13px;color:#fff;}
.id-main{padding:24px 28px;display:grid;grid-template-columns:repeat(3,1fr);gap:20px 24px;align-content:center;}
.id-main .v{font-size:16px;font-weight:650;color:var(--ink);word-break:break-word;}
.id-main .v.m{font-family:'JetBrains Mono',monospace;font-size:14.5px;}

/* ---------- verdict ---------- */
.vgrid{display:grid;grid-template-columns:1fr 280px;gap:18px;margin:6px 0 18px;}
.vpanel{background:var(--card);border:1px solid var(--line);border-left:6px solid #64748b;border-radius:18px;padding:26px 28px;box-shadow:0 1px 2px rgba(16,24,40,.04),0 8px 24px rgba(16,24,40,.06);}
.vpanel.s-approve{border-left-color:#16a34a;} .vpanel.s-review{border-left-color:#d97706;} .vpanel.s-reject{border-left-color:#dc2626;}
.vrow{display:flex;gap:28px;align-items:center;margin-top:12px;}
.vbadge{flex:none;min-width:230px;display:flex;gap:14px;align-items:center;padding:16px 20px;border-radius:14px;background:#f1f5f9;color:#334155;}
.vi{width:46px;height:46px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:24px;font-weight:800;color:#fff;background:#64748b;}
.vb-l{font-size:11px;letter-spacing:1.2px;text-transform:uppercase;font-weight:700;opacity:.8;}
.vb-v{font-size:30px;font-weight:800;letter-spacing:-.8px;line-height:1.1;}
.vbadge.s-approve{background:#e7f6ec;color:#14532d;} .vbadge.s-approve .vi{background:#16a34a;}
.vbadge.s-review{background:#fef3c7;color:#78350f;} .vbadge.s-review .vi{background:#d97706;}
.vbadge.s-reject{background:#fee2e2;color:#7f1d1d;} .vbadge.s-reject .vi{background:#dc2626;}
.vreason{font-size:15px;line-height:1.7;color:var(--text);font-weight:500;}
.vtags{margin-top:12px;}
.gcard{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:22px;display:flex;flex-direction:column;align-items:center;justify-content:center;box-shadow:0 1px 2px rgba(16,24,40,.04),0 8px 24px rgba(16,24,40,.06);}
.ring{width:150px;height:150px;border-radius:50%;display:flex;align-items:center;justify-content:center;}
.ring-in{width:116px;height:116px;border-radius:50%;background:#fff;display:flex;flex-direction:column;align-items:center;justify-content:center;}
.ring-n{font-size:38px;font-weight:800;line-height:1;color:var(--ink);letter-spacing:-1px;}
.ring-s{font-size:11px;color:var(--muted);margin-top:2px;}
.bands{display:flex;gap:4px;margin-top:16px;}
.bands span{font-size:10px;letter-spacing:.8px;font-weight:700;padding:4px 8px;border-radius:6px;background:#f1f5f9;color:var(--muted);text-transform:uppercase;}
.bands span.on{color:#fff;}

/* ---------- chips / bars / evidence ---------- */
.chip{display:inline-block;margin:0 6px 6px 0;padding:5px 11px;border-radius:8px;font-size:12px;font-weight:600;background:#fef3c7;color:#92400e;border:1px solid #fde68a;}
.chip.g{background:#e7f6ec;color:#166534;border-color:#bbf7d0;} .chip.r{background:#fee2e2;color:#991b1b;border-color:#fecaca;}
.bar{margin-bottom:16px;}
.bar-h{display:flex;justify-content:space-between;font-size:13px;font-weight:600;color:var(--ink);margin-bottom:6px;}
.bar-h span:last-child{font-family:'JetBrains Mono',monospace;color:var(--muted);font-weight:500;}
.bar-t{height:10px;border-radius:99px;background:#e8edf3;overflow:hidden;} .bar-f{height:100%;border-radius:99px;}
.ev{display:flex;gap:11px;background:var(--card);border:1px solid var(--line);border-left:3px solid var(--accent);border-radius:10px;padding:11px 14px;margin-bottom:8px;font-size:13px;color:#1f2d3d;line-height:1.55;}
.ev b{color:var(--accent);}
.audit{background:#f0fdfa;border:1px solid #99f6e4;border-radius:14px;padding:18px 22px;font-size:13.5px;line-height:2.1;color:var(--ink);}
.audit code{font-family:'JetBrains Mono',monospace;font-size:12px;background:#fff;padding:3px 8px;border-radius:6px;border:1px solid #cdeee8;margin-right:4px;}

/* ---------- widgets ---------- */
[data-testid="stFileUploader"] section{background:var(--card);border:2px dashed #b8c4d4;border-radius:16px;padding:22px;}
[data-testid="stFileUploader"] section:hover{border-color:var(--accent);}
[data-testid="stMetric"]{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:14px 16px;box-shadow:0 1px 2px rgba(16,24,40,.04);}
[data-testid="stMetricLabel"]{color:var(--muted) !important;font-size:11px !important;text-transform:uppercase;letter-spacing:.8px;font-weight:700;}
[data-testid="stMetricValue"]{color:var(--ink) !important;font-weight:800 !important;letter-spacing:-.5px;}
.stButton > button,.stDownloadButton > button{border-radius:12px;border:none;background:linear-gradient(135deg,#0d9488,#0f766e);color:#fff;font-weight:700;padding:14px 20px;box-shadow:0 8px 20px rgba(13,148,136,.28);}
.stButton > button:hover,.stDownloadButton > button:hover{color:#fff;transform:translateY(-1px);}
button[data-baseweb="tab"]{font-size:14px;font-weight:600;padding:11px 20px;}
button[data-baseweb="tab"][aria-selected="true"]{color:var(--ink);}
div[data-baseweb="tab-highlight"]{background:var(--accent) !important;height:3px !important;}
.footer{text-align:center;color:var(--muted);font-size:12px;padding:36px 0 4px;line-height:1.9;}
@media(max-width:1000px){.idcard,.vgrid{grid-template-columns:1fr;}.id-main{grid-template-columns:1fr 1fr;}.chain{grid-template-columns:repeat(2,1fr);}.vrow{flex-direction:column;align-items:flex-start;}.nav-m{display:none;}.hero h1{font-size:34px;}}
.case-grid{grid-template-columns:1fr !important;}
.vpanel{display:flex;flex-direction:column;justify-content:center;}
.evgrid{display:grid;grid-template-columns:1fr 1fr;gap:8px 12px;} .evgrid .ev{margin-bottom:0;}
[data-testid="stImage"] img{height:auto;width:auto;max-width:100%;max-height:540px;object-fit:contain;border-radius:14px;border:1px solid var(--line);}
button[data-baseweb="tab"][aria-selected="true"],button[data-baseweb="tab"][aria-selected="true"] *{color:#0b1b2b !important;}
[data-baseweb="tab-highlight"]{background-color:#0d9488 !important;height:3px !important;}
@media(max-width:1000px){.evgrid{grid-template-columns:1fr;}}
</style>
""")

# ============================================================
# HELPERS
# ============================================================

LEVEL_COLOR = {"low": "#16a34a", "medium": "#d97706", "high": "#ea580c", "critical": "#dc2626"}
LEVELS = ["low", "medium", "high", "critical"]


def e(v) -> str:
    return escape(str(v))


def g(obj, name, default="—"):
    value = getattr(obj, name, default)
    return default if value is None else value


def section(no: str, title: str, desc: str = "") -> None:
    step = f"<small>STEP {e(no)}</small>" if no else ""
    render_html(f'<div class="sec"><div class="sh">{step}<span class="t">{e(title)}</span></div><i></i></div>')
    if desc:
        render_html(f'<div class="sec-d">{e(desc)}</div>')


def evidence_list(items) -> None:
    render_html('<div class="evgrid">' + "".join(f'<div class="ev"><b>✓</b><div>{e(i)}</div></div>' for i in items) + "</div>")


def chips(items, kind="") -> str:
    return "".join(f'<span class="chip {kind}">{e(i)}</span>' for i in items)


def kv_card(title: str, rows: list) -> str:
    body = "".join(f'<div class="kv"><span>{e(k)}</span><b>{e(v)}</b></div>' for k, v in rows)
    return f'<div class="card"><div class="lbl">{e(title)}</div>{body}</div>'


def bar(label: str, value: float, color: str) -> str:
    pct = max(0.0, min(100.0, value))
    return (
        f'<div class="bar"><div class="bar-h"><span>{e(label)}</span><span>{value:.2f}</span></div>'
        f'<div class="bar-t"><div class="bar-f" style="width:{pct:.1f}%;background:{color}"></div></div></div>'
    )


def initials(name: str) -> str:
    return "".join(p[0] for p in str(name).split()[:2]).upper() or "?"


def build_identity(row: pd.Series) -> IdentityInput:
    return IdentityInput(
        customer_id=str(row["customer_id"]),
        full_name=str(row["full_name"]),
        date_of_birth=str(row["date_of_birth"]),
        document_number=str(row["document_number"]),
        nationality=str(row["nationality"]),
        country=str(row["country"]),
    )


# ============================================================
# DATA
# ============================================================

DATA_DIR = PROJECT_ROOT / "data_generation" / "synthetic_documents"
IDENTITY_PATH = DATA_DIR / "identity_data.csv"
CASE_MAPPING_PATH = DATA_DIR / "case_mapping.csv"
MAPPING_COLUMNS = {
    "customer_id", "synthetic_document_number", "full_name", "identity_document_type",
    "midv_document_id", "midv_document_type", "midv_image_path",
    "midv_ground_truth_path", "source_dataset",
}


@st.cache_data
def load_cases() -> pd.DataFrame:
    if not IDENTITY_PATH.exists():
        raise FileNotFoundError(f"Identity data not found: {IDENTITY_PATH}")
    df = load_identity_data()
    if df.empty:
        raise ValueError("Identity dataset is empty.")
    return df


@st.cache_data
def load_case_mapping() -> pd.DataFrame:
    if not CASE_MAPPING_PATH.exists():
        raise FileNotFoundError(f"Case mapping not found: {CASE_MAPPING_PATH}")
    df = pd.read_csv(CASE_MAPPING_PATH)
    if df.empty:
        raise ValueError("Case mapping is empty.")
    missing = MAPPING_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"case_mapping.csv is missing required columns: {sorted(missing)}")
    return df


def get_case_mapping(customer_id: str, mapping: pd.DataFrame) -> pd.Series:
    matches = mapping[mapping["customer_id"].astype(str) == str(customer_id)]
    if matches.empty:
        raise ValueError(f"No synthetic document mapping found for {customer_id}.")
    if len(matches) > 1:
        raise ValueError(f"Multiple document mappings found for {customer_id}.")
    return matches.iloc[0]


try:
    identities = load_cases()
    case_mapping = load_case_mapping()
except Exception as exc:
    st.error(f"Unable to load ComplyAgent data: {exc}")
    st.stop()

# ============================================================
# SIDEBAR
# ============================================================

STEPS = ["Customer", "Document", "Screening", "Risk", "Decision", "Audit"]

with st.sidebar:
    render_html("""
        <div class="sb-brand"><div class="sb-mark">🛡️</div>
        <div><div class="sb-name">ComplyAgent</div><div class="sb-sub">AI-POWERED COMPLIANCE</div></div></div>
        <div class="side-h">Active case</div>
    """)
    customer_ids = identities["customer_id"].astype(str).tolist()
    selected_customer = st.selectbox(
        "Select synthetic case", customer_ids, index=0, label_visibility="collapsed"
    )

    if st.session_state.get("last_customer") != selected_customer:
        st.session_state["analysis_complete"] = False
        for key in ["document_image", "document_evidence", "screening", "risk", "decision", "audit_trace"]:
            st.session_state.pop(key, None)
        st.session_state["last_customer"] = selected_customer

    matches = identities[identities["customer_id"].astype(str) == selected_customer]
    if matches.empty:
        st.error(f"Customer not found: {selected_customer}")
        st.stop()
    row = matches.iloc[0]

    done = bool(st.session_state.get("analysis_complete", False))
    tl = "".join(
        f'<div class="tl-item {"done" if done or i == 0 else ("on" if i == 1 else "")}">{s}'
        f'<small>{"COMPLETED" if done or i == 0 else ("AWAITING UPLOAD" if i == 1 else "PENDING")}</small></div>'
        for i, s in enumerate(STEPS)
    )
    svc = "".join(
        f'<div class="svc"><span>{n}</span><span><span class="dot"></span>Online</span></div>'
        for n in ["LangGraph", "PostgreSQL", "OCR + ELA", "Risk Engine"]
    )
    render_html(f"""
        <div class="case-card"><div class="case-tag">● CASE FILE</div>
        <div class="case-id">{e(row["customer_id"])}</div><div class="case-name">{e(row["full_name"])}</div>
        <div class="case-grid">
          <div><div class="case-k">Document ID</div><div class="case-v">{e(row["document_number"])}</div></div>
          <div><div class="case-k">Country</div><div class="case-v">{e(row["country"])}</div></div>
        </div></div>
        <div class="side-h">Pipeline</div><div class="tl">{tl}</div>
        <div class="side-h">System health</div>{svc}
    """)

# ============================================================
# MAPPING
# ============================================================

try:
    mapped = get_case_mapping(selected_customer, case_mapping)
except Exception as exc:
    st.error(f"Unable to resolve synthetic document mapping: {exc}")
    st.stop()

synthetic_document_number = str(mapped["synthetic_document_number"])
midv_document_id = str(mapped["midv_document_id"])
midv_document_type = str(mapped["midv_document_type"])
source_dataset = str(mapped["source_dataset"])

# ============================================================
# NAV + HERO
# ============================================================

S = st.session_state
analysis_done = bool(S.get("analysis_complete", False))
d_ev, sc_ev, rk_ev, dc_ev = S.get("document_evidence"), S.get("screening"), S.get("risk"), S.get("decision")

if analysis_done and all(x is not None for x in (d_ev, sc_ev, rk_ev, dc_ev)):
    chain_vals = [
        ("Identity", e(selected_customer), "done"),
        ("Document", f"OCR {d_ev.ocr.confidence:.2f}", "done"),
        ("Screening", e(str(sc_ev.screening_status).upper()), "done"),
        ("Risk", f"{rk_ev.overall_risk_score:.0f} / 100", "done"),
        ("Decision", e(dc_ev.decision), "done"),
        ("Audit", "Complete" if dc_ev.trace_complete else "Incomplete", "done"),
    ]
else:
    chain_vals = [("Identity", e(selected_customer), "done"), ("Document", "Awaiting upload", "on")] + [
        (n, "Pending", "") for n in ["Screening", "Risk", "Decision", "Audit"]
    ]
chain_html = "".join(
    f'<div class="node {c}"><div class="pt"></div><div class="nl">{n}</div><div class="nv">{v}</div></div>'
    for n, v, c in chain_vals
)
nav_items = [("Dossier", "ok"), ("Intake", "ok" if analysis_done else "on"), ("Analysis", "ok" if analysis_done else ""),
             ("Verdict", "ok" if analysis_done else ""), ("Audit", "ok" if analysis_done else "")]
nav_html = "".join(f'<span class="{c}">{n}</span>' for n, c in nav_items)

render_html(f"""
    <div class="nav">
      <div class="nav-l"><i>🛡️</i>ComplyAgent</div>
      <div class="nav-m">{nav_html}</div>
      <div class="nav-r">Synthetic benchmark • Research / demonstration</div>
    </div>
    <div class="hero">
      <div class="eyebrow"><b></b>Compliance Review Center</div>
      <h1>Compliance decisions you can <em>trace to the evidence.</em></h1>
      <div class="tagline">AI-Powered KYC &amp; AML Compliance Intelligence</div>
      <p>Multi-agent assessment combining document intelligence, sanctions screening,
      AML risk scoring and complete evidence traceability. No real customer PII is used.</p>
      <div class="chain">{chain_html}</div>
    </div>
""")

FOOTER = """
<div class="footer"><strong>ComplyAgent</strong> · Multi-Agent KYC/AML Compliance System<br>
Synthetic benchmark • Research / demonstration environment · Not validated for real-world regulatory decision making</div>
"""

# ============================================================
# 01 CUSTOMER DOSSIER
# ============================================================

section("01", "Customer dossier", "Synthetic identity loaded from the benchmark dataset.")
render_html(f"""
    <div class="idcard">
      <div class="id-side"><div class="id-tag">SYNTHETIC IDENTITY</div>
        <div class="id-photo">{e(initials(row["full_name"]))}</div>
        <div class="id-no">{e(row["customer_id"])}</div></div>
      <div class="id-main">
        <div><div class="lbl">Full name</div><div class="v">{e(row["full_name"])}</div></div>
        <div><div class="lbl">Date of birth</div><div class="v">{e(row["date_of_birth"])}</div></div>
        <div><div class="lbl">Document number</div><div class="v m">{e(row["document_number"])}</div></div>
        <div><div class="lbl">Nationality</div><div class="v">{e(row["nationality"])}</div></div>
        <div><div class="lbl">Country</div><div class="v">{e(row["country"])}</div></div>
        <div><div class="lbl">Document type</div><div class="v">{e(mapped["identity_document_type"])}</div></div>
      </div>
    </div>
""")

# ============================================================
# 02 DOCUMENT INTAKE
# ============================================================

section("02", "Document intake", "Upload a JPG, JPEG, PNG, TIF or TIFF identity document for OCR and tamper-evidence (ELA) analysis.")
uploaded_file = st.file_uploader(
    "Upload identity document",
    type=["jpg", "jpeg", "png", "tif", "tiff"],
    label_visibility="collapsed",
)

if uploaded_file is None:
    left, right = st.columns(2)
    with left:
        render_html(kv_card("Benchmark mapping", [
            ("Customer ID", selected_customer),
            ("Synthetic document number", synthetic_document_number),
            ("Identity document type", mapped["identity_document_type"]),
        ]))
    with right:
        render_html(kv_card("Source sample", [
            ("MIDV document ID", midv_document_id),
            ("MIDV document type", midv_document_type),
            ("Source dataset", source_dataset),
        ]))
    render_html("""<div class="note">This is a deterministic synthetic benchmark linkage. It does <b>not</b> claim
    that the MIDV document genuinely belongs to the Faker-generated identity. Upload a document to start the
    complete compliance workflow.</div>""")
    render_html(FOOTER)
    st.stop()

file_bytes = uploaded_file.getvalue()
render_html(kv_card("Uploaded document", [
    ("File", uploaded_file.name),
    ("Size", f"{len(file_bytes) / 1024:.1f} KB"),
    ("Compliance document ID", synthetic_document_number),
    ("MIDV sample", midv_document_id),
    ("Document type", midv_document_type),
]))

analyze = st.button("🔍  Analyze Document & Run Compliance Check", type="primary", use_container_width=True)

if analyze:
    ok = False
    try:
        with st.spinner("Running document intelligence and compliance workflow..."):
            document_image, document_evidence = analyze_uploaded_document(
                file_bytes=file_bytes,
                document_id=synthetic_document_number,
                document_type=midv_document_type,
            )
            identity = build_identity(row)
            result = build_graph().invoke({
                "identity": identity,
                "document": document_evidence,
                "screening": None,
                "risk": None,
                "decision": None,
                "audit_trace": None,
            })
        S.update({
            "document_image": document_image,
            "document_evidence": document_evidence,
            "screening": result["screening"],
            "risk": result["risk"],
            "decision": result["decision"],
            "audit_trace": result.get("audit_trace"),
            "analysis_complete": True,
        })
        ok = True
    except Exception as exc:
        st.error("Compliance workflow failed.")
        st.exception(exc)
        st.stop()
    if ok:
        st.rerun()

if not S.get("analysis_complete", False):
    render_html(FOOTER)
    st.stop()

# ============================================================
# RESULTS
# ============================================================

document_evidence: DocumentEvidence = S["document_evidence"]
screening = S["screening"]
risk: RiskEvidence = S["risk"]
decision: DecisionEvidence = S["decision"]
document_image = S["document_image"]
audit_trace = S.get("audit_trace")

ml, rules, tx = risk.ml_risk, risk.rules, risk.transaction_pattern
sanctions = screening.sanctions_match
score = float(risk.overall_risk_score)
level = str(risk.risk_level).lower()
level_color = LEVEL_COLOR.get(level, "#64748b")
ml_score = float(ml.risk_score)
rule_score = float(rules.rule_score) * 100

# ---------------- 03 document analysis ----------------
section("03", "Document analysis", "Optical character recognition and Error Level Analysis (tamper evidence).")
c1, c2 = st.columns([1.15, 1])
with c1:
    st.image(document_image, channels="BGR", caption="Uploaded identity document", use_container_width=True)
with c2:
    a, b = st.columns(2)
    a.metric("OCR Confidence", f"{document_evidence.ocr.confidence:.2f}")
    b.metric("OCR Words", document_evidence.ocr.word_count)
    c, d = st.columns(2)
    c.metric("ELA Score", f"{document_evidence.tamper.ela_score:.4f}")
    d.metric("ELA Threshold", f"{document_evidence.tamper.threshold:.4f}")
    suspicious = bool(document_evidence.tamper.suspicious)
    render_html(
        kv_card("Document assessment", [
            ("Document quality", document_evidence.document_quality),
            ("ELA suspicious", "YES" if suspicious else "NO"),
            ("Manual review required", "YES" if document_evidence.requires_review else "NO"),
        ])
    )
    thr = float(document_evidence.tamper.threshold) or 1.0
    ratio = float(document_evidence.tamper.ela_score) / thr * 100
    render_html(
        '<div class="card" style="margin:12px 0;"><div class="lbl">Tamper evidence</div>'
        + bar("ELA score as % of threshold", ratio, "#dc2626" if suspicious else "#16a34a")
        + '<div style="font-size:12px;color:#5b6778;">Values above 100 exceed the tamper threshold.</div></div>'
    )
    with st.expander("View extracted OCR text"):
        if document_evidence.ocr.text:
            st.code(document_evidence.ocr.text)
        else:
            st.info("No OCR text was extracted.")

# ---------------- 04 verdict ----------------
decision_value = str(decision.decision)
stamp_cls = {"APPROVE": "s-approve", "REVIEW": "s-review", "REJECT": "s-reject"}.get(decision_value, "s-other")
dec_icon = {"APPROVE": "✓", "REVIEW": "!", "REJECT": "×"}.get(decision_value, "?")
deg = max(0.0, min(100.0, score)) * 3.6
bands = "".join(
    f'<span class="{"on" if l == level else ""}" style="{"background:" + LEVEL_COLOR[l] if l == level else ""}">{l[:4]}</span>'
    for l in LEVELS
)
tags = chips(
    [f"Screening: {str(screening.screening_status).upper()}"], "g" if not sanctions.matched else "r"
) + chips(
    ["Manual review required" if decision.requires_review else "No manual review"],
    "" if decision.requires_review else "g",
) + chips(["Trace complete" if decision.trace_complete else "Trace incomplete"], "g" if decision.trace_complete else "r")

section("04", "Final verdict", "Decision synthesised from document, screening and risk evidence.")
render_html(f"""
    <div class="vgrid">
      <div class="vpanel {stamp_cls}">
        <div class="lbl">Compliance decision</div>
        <div class="vrow">
          <div class="vbadge {stamp_cls}"><div class="vi">{dec_icon}</div><div><div class="vb-l">Final decision</div><div class="vb-v">{e(decision_value)}</div></div></div>
          <div><div class="vreason">{e(decision.decision_reason)}</div><div class="vtags">{tags}</div></div>
        </div>
      </div>
      <div class="gcard">
        <div class="lbl">Overall risk</div>
        <div class="ring" style="background:conic-gradient({level_color} {deg:.1f}deg,#e8edf3 0deg);">
          <div class="ring-in"><div class="ring-n">{score:.0f}</div><div class="ring-s">/ 100</div></div>
        </div>
        <div class="bands">{bands}</div>
      </div>
    </div>
""")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Risk Score", f"{score:.2f}/100")
k2.metric("Risk Level", risk.risk_level.upper())
k3.metric("Screening", screening.screening_status.upper())
k4.metric("Trace Complete", "YES" if decision.trace_complete else "NO")

# ---------------- 05 evidence tabs ----------------
section("05", "Evidence workbench", "Every signal behind the decision, in one place.")
tab_overview, tab_screening, tab_risk, tab_decision, tab_audit = st.tabs(
    ["Overview", "Screening", "Risk Intelligence", "Decision", "Audit Trail"]
)

with tab_overview:
    o1, o2 = st.columns(2)
    with o1:
        render_html(kv_card("Document intelligence", [
            ("Compliance document ID", document_evidence.document_id),
            ("MIDV sample", midv_document_id),
            ("Document type", document_evidence.document_type),
            ("Quality", document_evidence.document_quality),
            ("OCR confidence", f"{document_evidence.ocr.confidence:.2f}"),
            ("ELA suspicious", document_evidence.tamper.suspicious),
        ]))
    with o2:
        render_html(kv_card("Compliance status", [
            ("Screening", screening.screening_status),
            ("Risk level", risk.risk_level),
            ("Risk score", f"{score:.2f}/100"),
            ("Decision", decision.decision),
            ("Requires review", decision.requires_review),
        ]))
    section("", "Evidence summary")
    evidence_list(decision.evidence)

with tab_screening:
    m1, m2, m3 = st.columns(3)
    m1.metric("Status", screening.screening_status.upper())
    m2.metric("Sanctions Match", "YES" if sanctions.matched else "NO")
    m3.metric("Match Score", f"{sanctions.match_score:.2f}")
    render_html('<div style="height:12px"></div>' + kv_card("Screening detail", [
        ("Source", sanctions.source),
        ("Matched name", sanctions.matched_name or "None"),
        ("Reason", sanctions.reason),
    ]))
    with st.expander("View complete screening evidence"):
        st.json(screening.model_dump())

with tab_risk:
    render_html(
        '<div class="card"><div class="lbl">Score composition</div><div style="height:10px"></div>'
        + bar("ML risk score", ml_score, "#0d9488")
        + bar("Rules engine score", rule_score, "#1e3a5f")
        + bar("Overall risk", score, level_color) + "</div>"
    )
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("ML Risk Score", f"{ml_score:.2f}/100")
    r1.caption(str(g(ml, "model")))
    r2.metric("Rule Score", f"{rule_score:.2f}/100")
    r2.caption(f"{rules.rule_count} rule(s) triggered")
    r3.metric("Transaction Risk", "HIGH" if tx.high_risk else "LOW")
    r3.caption(str(tx.scenario))
    prob = g(ml, "risk_probability", None)
    r4.metric("ML Probability", f"{prob:.2%}" if isinstance(prob, (int, float)) else "—")
    r4.caption(f"Prediction: {g(ml, 'prediction')}")

    t1, t2 = st.columns(2)
    with t1:
        render_html(kv_card("Transaction pattern", [
            ("Scenario", tx.scenario),
            ("24h transactions", g(tx, "transaction_count_24h")),
            ("24h amount", f"{tx.total_amount_24h:,.2f}" if isinstance(g(tx, "total_amount_24h", None), (int, float)) else "—"),
            ("Structuring count", g(tx, "structuring_count")),
            ("Rapid movement count", g(tx, "rapid_movement_count")),
            ("New beneficiaries", g(tx, "new_beneficiary_count")),
        ]))
    with t2:
        ind = chips(tx.indicators) if tx.indicators else '<span class="chip g">No indicators triggered</span>'
        rl = chips(rules.rules_triggered) if rules.rules_triggered else '<span class="chip g">No rules triggered</span>'
        render_html(f'<div class="card"><div class="lbl">AML indicators</div>{ind}'
                    f'<div style="height:18px"></div><div class="lbl">Rules triggered</div>{rl}</div>')
    with st.expander("View transaction-pattern details"):
        st.json(tx.model_dump())
    if risk.evidence:
        section("", "Risk evidence")
        evidence_list(risk.evidence)

with tab_decision:
    render_html(kv_card("Decision summary", [
        ("Decision", decision.decision),
        ("Risk level", decision.risk_level),
        ("Risk score", f"{decision.overall_risk_score:.2f}/100"),
        ("Screening status", decision.screening_status),
        ("Requires review", decision.requires_review),
        ("Trace complete", decision.trace_complete),
    ]))
    render_html(f'<div class="note" style="font-size:14px;color:#0d1b1e;"><b>Reason.</b> {e(decision.decision_reason)}</div>')
    section("", "Evidence")
    evidence_list(decision.evidence)

with tab_audit:
    if decision.trace_complete:
        st.success("Complete evidence trace is available.")
    else:
        st.error("Audit trace is incomplete.")
    q1, q2, q3 = st.columns(3)
    q1.metric("Customer", decision.customer_id)
    q2.metric("Document", decision.document_id)
    q3.metric("Decision", decision.decision)
    if isinstance(audit_trace, dict) and audit_trace.get("trace_id"):
        render_html(f'<div class="note">Trace ID <code>{e(audit_trace["trace_id"])}</code></div>')
    render_html("""<div class="audit" style="margin-top:12px;"><strong>✓ Trace components</strong><br>
        <code>Document Evidence</code><code>Screening Evidence</code><code>Risk Evidence</code><code>Decision Evidence</code><br>
        Persisted by the LangGraph audit node to <code>JSONL</code><code>PostgreSQL</code></div>""")
    full_trace = {
        "document_evidence": document_evidence.model_dump(),
        "screening_evidence": screening.model_dump(),
        "risk_evidence": risk.model_dump(),
        "decision_evidence": decision.model_dump(),
    }
    st.download_button(
        "⬇  Download audit evidence (JSON)",
        data=json.dumps(full_trace, indent=2, default=str),
        file_name=f"complyagent_audit_{decision.customer_id}.json",
        mime="application/json",
    )
    with st.expander("View complete audit evidence"):
        st.json(full_trace)

render_html(FOOTER)