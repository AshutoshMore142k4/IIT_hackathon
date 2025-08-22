import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import shap
import plotly.express as px
import json
import os
import time
import requests

from typing import Union

st.set_page_config(page_title='Portfolio Dashboard', page_icon='📊', layout='wide')
shap.initjs()

# ----------- Styling -----------
st.markdown(
    """
    <style>
    .kpi-card {background: #0f172a; padding: 16px; border-radius: 10px; color: #e2e8f0;}
    .kpi-value {font-size: 26px; font-weight: 700; color: #22d3ee;}
    .kpi-label {font-size: 13px; opacity: 0.8;}
    .section-title {font-size: 20px; font-weight: 700; margin: 12px 0 6px 0;}
    .file-row {padding:6px 8px; border-radius:6px;}
    .file-row:hover {background:#0b1220}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------- Persistence helpers -----------
UPLOAD_DIR = os.path.join('dashboard', 'user_uploads')
INDEX_FILE = os.path.join(UPLOAD_DIR, 'uploads_index.json')
os.makedirs(UPLOAD_DIR, exist_ok=True)

@st.cache_data(show_spinner=False)
def _read_csv_cached(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def _get_api_url() -> str:
    try:
        return st.secrets["config"]["API_URL"].rstrip('/')
    except Exception:
        return "http://127.0.0.1:5000"

@st.cache_data(ttl=15, show_spinner=False)
def fetch_alerts() -> dict:
    url = f"{_get_api_url()}/alerts/"
    try:
        r = requests.get(url, timeout=5)
        if r.ok:
            return r.json()
    except Exception:
        pass
    return {"count": 0, "alerts": [], "threshold": None, "refresh_interval_seconds": None}

@st.cache_data(ttl=60, show_spinner=False)
def fetch_clients() -> list[int]:
    url = f"{_get_api_url()}/clients/"
    try:
        r = requests.get(url, timeout=5)
        if r.ok:
            return r.json()
    except Exception:
        pass
    return []

@st.cache_data(ttl=10, show_spinner=False)
def fetch_realtime(client_id: int) -> dict:
    url = f"{_get_api_url()}/realtime/scores/{int(client_id)}"
    try:
        r = requests.get(url, timeout=5)
        if r.ok:
            return r.json()
    except Exception:
        pass
    return {}

@st.cache_data(ttl=10, show_spinner=False)
def fetch_predict(client_id: int) -> dict:
    url = f"{_get_api_url()}/predict/{int(client_id)}?return_data=true"
    try:
        r = requests.get(url, timeout=5)
        if r.ok:
            return r.json()
    except Exception:
        pass
    return {}

@st.cache_data(ttl=30, show_spinner=False)
def fetch_explain(client_id: int) -> dict:
    url = f"{_get_api_url()}/explain/{int(client_id)}?return_data=true"
    try:
        r = requests.get(url, timeout=10)
        if r.ok:
            # Some APIs might return NaN; try robust parsing
            try:
                return r.json()
            except Exception:
                txt = r.text.replace('NaN', 'null')
                return json.loads(txt)
    except Exception:
        pass
    return {}

def render_top_features(explain_payload: dict, top_n: int = 10):
    # Try to extract per-feature SHAP values for the single client
    # Expected shapes: shap_values: [n_samples, n_features] or [n_classes, n_samples, n_features]
    x = None
    cols = None
    shap_vals = None
    # columns
    cols = explain_payload.get('columns') or explain_payload.get('feature_names')
    # x_data
    x = explain_payload.get('x_data')
    if isinstance(x, dict):
        # may be mapping feature -> value
        try:
            cols = list(x.keys())
            x = [x[c] for c in cols]
        except Exception:
            x = None
    # shap_values
    sv = explain_payload.get('shap_values') or explain_payload.get('shap_values_raw')
    if isinstance(sv, list):
        # Could be nested; pick class 1 if present
        if len(sv) > 0 and isinstance(sv[0], list):
            # assume [classes][samples][features]
            shap_vals = np.array(sv[min(1, len(sv)-1)])
        else:
            shap_vals = np.array(sv)
    elif isinstance(sv, dict):
        # unsupported structure, try flatten
        try:
            shap_vals = np.array(list(sv.values()))
        except Exception:
            shap_vals = None
    if shap_vals is None:
        st.warning('No SHAP values available for this issuer.')
        return
    # select first sample row if 2D
    if shap_vals.ndim == 3:
        shap_vals = shap_vals[0]
    if shap_vals.ndim == 2:
        row = shap_vals[0]
    else:
        row = shap_vals
    if cols is None and isinstance(x, list) and len(x) == len(row):
        cols = [f'f{i}' for i in range(len(row))]
    if cols is None:
        cols = [f'f{i}' for i in range(len(row))]
    s = pd.Series(row, index=cols).sort_values(key=lambda v: v.abs(), ascending=False).head(top_n)
    st.markdown('<div class="section-title">Top Feature Drivers (local SHAP)</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    s[::-1].plot(kind='barh', ax=ax, color=['#ef4444' if v>0 else '#22c55e' for v in s[::-1].values])
    ax.set_xlabel('SHAP value')
    ax.set_ylabel('Feature')
    st.pyplot(fig)

def _migrate_files(files_raw):
    entries = []
    for item in files_raw:
        if isinstance(item, dict) and 'name' in item and 'path' in item:
            entries.append(item)
        else:
            # legacy: just stored filename, reconstruct
            fname = str(item)
            orig = fname.split('_', 1)[1] if '_' in fname else fname
            entries.append({'name': orig, 'path': os.path.join(UPLOAD_DIR, fname)})
    return entries

def load_index():
    try:
        if os.path.exists(INDEX_FILE):
            with open(INDEX_FILE, 'r') as f:
                idx = json.load(f)
            files = _migrate_files(idx.get('files', []))
            active = idx.get('active_idx')
            if isinstance(active, int) and (0 <= active < len(files)):
                pass
            else:
                active = 0 if files else None
            return files, active
    except Exception:
        pass
    return [], None

def save_index(files, active_idx):
    try:
        with open(INDEX_FILE, 'w') as f:
            json.dump({'files': files, 'active_idx': active_idx}, f)
    except Exception:
        pass

# ----------- Session multi-file state -----------
if 'files' not in st.session_state or 'active_idx' not in st.session_state:
    files, active = load_index()
    st.session_state.files = files
    st.session_state.active_idx = active
if 'active_df' not in st.session_state:
    st.session_state.active_df = None

# ----------- Helpers -----------
def normalise_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors='coerce')
    if s.std(skipna=True) == 0 or s.dropna().empty:
        return s.fillna(0)
    return (s - s.mean(skipna=True)) / (s.std(skipna=True) + 1e-9)


def build_from_csv(uploaded: pd.DataFrame) -> pd.DataFrame:
    df = uploaded.copy()
    cols_lower = {c.lower(): c for c in df.columns}
    score_col = next((cols_lower[k] for k in cols_lower if k in ['credit_score','score']), None)
    prob_col = next((cols_lower[k] for k in cols_lower if k in ['y_pred_proba','probability','default_probability','pd']), None)
    income_col = next((cols_lower[k] for k in cols_lower if 'income' in k), None)
    util_col = next((cols_lower[k] for k in cols_lower if 'util' in k or 'utilization' in k), None)

    if prob_col is None:
        num = df.select_dtypes(include=[np.number])
        if num.empty:
            df['y_pred_proba'] = 0.5
        else:
            z = normalise_series(num.sum(axis=1))
            df['y_pred_proba'] = (1 / (1 + np.exp(-z))).clip(0,1)
    else:
        df['y_pred_proba'] = pd.to_numeric(df[prob_col], errors='coerce').fillna(0.5).clip(0,1)

    if score_col is None:
        df['credit_score'] = ((1 - df['y_pred_proba']) * 550 + 300).clip(300, 850)
    else:
        df['credit_score'] = pd.to_numeric(df[score_col], errors='coerce').fillna(df[score_col].median()).clip(300, 850)

    if income_col is None:
        num = df.select_dtypes(include=[np.number])
        df['income'] = (num.sum(axis=1) * 10).abs()
    else:
        df['income'] = pd.to_numeric(df[income_col], errors='coerce').fillna(df[income_col].median())

    if util_col is None:
        df['credit_utilization'] = (1 - (df['credit_score'] - 300) / 550).clip(0,1)
    else:
        df['credit_utilization'] = pd.to_numeric(df[util_col], errors='coerce').fillna(0.5).clip(0,1)

    def risk_bucket(p):
        if pd.isna(p):
            return 'Unknown'
        if p >= 0.66:
            return 'High Risk'
        if p >= 0.33:
            return 'Medium Risk'
        return 'Low Risk'
    df['risk_category'] = df['y_pred_proba'].apply(risk_bucket)
    return df

# ----------- KPIs & Charts -----------
def show_kpis(df: pd.DataFrame):
    c1, c2, c3, c4 = st.columns(4)
    avg_score = df['credit_score'].mean()
    high_risk_pct = (df['risk_category'] == 'High Risk').mean() * 100
    avg_income = df['income'].mean()
    avg_util = df['credit_utilization'].mean()
    with c1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Average Credit Score</div><div class="kpi-value">{avg_score:.0f}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">High Risk (%)</div><div class="kpi-value">{high_risk_pct:.1f}%</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Average Income ($)</div><div class="kpi-value">{avg_income:,.0f}</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Average Utilization</div><div class="kpi-value">{avg_util:.2f}</div></div>', unsafe_allow_html=True)


def show_alerts_panel():
    st.markdown('<div class="section-title">Alerts</div>', unsafe_allow_html=True)
    data = fetch_alerts()
    meta = []
    if data.get('threshold') is not None:
        meta.append(f"threshold Δ≥{data['threshold']:.3f}")
    if data.get('refresh_interval_seconds'):
        mins = int((data['refresh_interval_seconds'] or 0) // 60)
        meta.append(f"refresh every {mins} min")
    st.caption("; ".join(meta) or "No meta")
    count = int(data.get('count', 0) or 0)
    if count == 0:
        st.success('No active alerts')
        return
    for a in data.get('alerts', [])[:50]:
        cid = a.get('id')
        delta = a.get('delta')
        direction = a.get('direction')
        ts = a.get('timestamp')
        st.write(f"ID {cid}: {direction} by {delta:+.3f} at {ts}")

def risk_pie(df: pd.DataFrame):
    st.markdown('<div class="section-title">Credit Risk Distribution</div>', unsafe_allow_html=True)
    counts = df['risk_category'].value_counts().reindex(['Low Risk','Medium Risk','High Risk','Unknown']).fillna(0)
    fig, ax = plt.subplots()
    ax.pie(counts.values, labels=counts.index, autopct='%1.1f%%', startangle=140)
    ax.axis('equal')
    st.pyplot(fig)


def score_hist(df: pd.DataFrame):
    st.markdown('<div class="section-title">Credit Score Distribution</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots()
    sns.histplot(df, x='credit_score', hue='risk_category', bins=30, ax=ax, palette='husl')
    st.pyplot(fig)


def income_vs_score(df: pd.DataFrame):
    st.markdown('<div class="section-title">Income vs Credit Score</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots()
    sns.scatterplot(df, x='income', y='credit_score', hue='risk_category', ax=ax, alpha=0.7)
    ax.set_xlabel('Annual Income ($)')
    ax.set_ylabel('Credit Score')
    st.pyplot(fig)


def correlation_heatmap(df: pd.DataFrame):
    st.markdown('<div class="section-title">Feature Correlation Heatmap</div>', unsafe_allow_html=True)
    num = df.select_dtypes(include=[np.number]).drop(columns=['y_pred_proba'], errors='ignore')
    corr = num.corr().clip(-1,1)
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(corr, cmap='RdBu_r', center=0, ax=ax)
    st.pyplot(fig)


def scatter_3d(df: pd.DataFrame):
    st.markdown('<div class="section-title">3D Portfolio View</div>', unsafe_allow_html=True)
    fig = px.scatter_3d(
        df,
        x='income', y='credit_score', z='credit_utilization',
        color='risk_category',
        size=(df['income'].abs() + 1),
        opacity=0.8,
        labels={'income':'Income', 'credit_score':'Credit Score', 'credit_utilization':'Utilization'},
    )
    st.plotly_chart(fig, use_container_width=True)

# ----------- Sidebar (single list at top, uploader at bottom) -----------
# Render list first
if st.session_state.files:
    st.sidebar.markdown('### Files')
    for i, entry in enumerate(st.session_state.files):
        fname = entry.get('name')
        path = entry.get('path')
        row = st.sidebar.container()
        c1, c2 = row.columns([0.85, 0.15])
        label = f"👉 {fname}" if i == (st.session_state.active_idx or -1) else fname
        if c1.button(label, key=f'file_{i}'):
            st.session_state.active_idx = i
            save_index(st.session_state.files, st.session_state.active_idx)
            if path and os.path.exists(path):
                st.session_state.active_df = build_from_csv(_read_csv_cached(path))
        if c2.button('×', key=f'del_{i}'):
            try:
                if path and os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass
            st.session_state.files.pop(i)
            if st.session_state.files:
                st.session_state.active_idx = min(i, len(st.session_state.files)-1)
            else:
                st.session_state.active_idx = None
                st.session_state.active_df = None
            save_index(st.session_state.files, st.session_state.active_idx)
            st.rerun()

# Uploader at the very bottom
st.sidebar.markdown('---')
new_file = st.sidebar.file_uploader('Upload CSV', type=['csv'], accept_multiple_files=False)
if new_file is not None:
    try:
        # Prevent duplicates by original name
        existing_names = {e.get('name') for e in st.session_state.files}
        if new_file.name in existing_names:
            st.sidebar.warning('File already uploaded.')
        else:
            ts_name = f"{int(time.time())}_{new_file.name}"
            save_path = os.path.join(UPLOAD_DIR, ts_name)
            with open(save_path, 'wb') as out:
                out.write(new_file.getbuffer())
            entry = {'name': new_file.name, 'path': save_path}
            st.session_state.files.insert(0, entry)  # add to top of list
            st.session_state.active_idx = 0
            save_index(st.session_state.files, st.session_state.active_idx)
            st.session_state.active_df = build_from_csv(pd.read_csv(save_path))
            st.success(f'Loaded {new_file.name}')
    except Exception as e:
        st.sidebar.error(f'Failed to load file: {e}')

# If active selected but df not loaded yet, load from persisted path
if st.session_state.active_df is None and st.session_state.active_idx is not None and st.session_state.files:
    path = st.session_state.files[st.session_state.active_idx].get('path')
    if path and os.path.exists(path):
        st.session_state.active_df = build_from_csv(_read_csv_cached(path))

# ----------- Main -----------
st.title('Portfolio Statistics')
with st.expander('Alerts (server)', expanded=True):
    show_alerts_panel()

with st.expander('Issuer detail', expanded=False):
    clients = fetch_clients()
    default_id = str(clients[0]) if clients else ''
    col_a, col_b = st.columns([0.6, 0.4])
    with col_a:
        cid_str = st.text_input('Client ID', value=default_id)
    with col_b:
        go = st.button('Load issuer')
    if go and cid_str.strip().isdigit():
        client_id = int(cid_str.strip())
        rt = fetch_realtime(client_id)
        pred = fetch_predict(client_id)
        exp = fetch_explain(client_id)
        c1, c2, c3 = st.columns(3)
        # Score
        score = None
        if isinstance(rt, dict):
            score = rt.get('current_score')
        if score is None and isinstance(pred, dict):
            score = pred.get('y_pred_proba') or pred.get('proba')
        with c1:
            st.metric('Current score (PD)', f"{(score if score is not None else float('nan')):.3f}")
        # Trend
        trend = rt.get('trend') if isinstance(rt, dict) else None
        prev = rt.get('previous_score') if isinstance(rt, dict) else None
        if score is not None and prev is not None:
            delta = score - prev
            with c2:
                st.metric('Trend', trend or 'FLAT', delta=f"{delta:+.3f}")
        else:
            with c2:
                st.metric('Trend', trend or 'N/A')
        # Refresh cadence
        with c3:
            rint = rt.get('refresh_interval_seconds') if isinstance(rt, dict) else None
            if rint:
                st.metric('Refresh (min)', f"{int(rint)//60}")
        # Top features
        render_top_features(exp, top_n=10)

if st.session_state.active_df is not None:
    df = st.session_state.active_df
    show_kpis(df)
    a, b = st.columns((1,1))
    with a:
        risk_pie(df)
    with b:
        score_hist(df)
    a, b = st.columns((1,1))
    with a:
        income_vs_score(df)
    with b:
        correlation_heatmap(df)
    scatter_3d(df)
else:
    st.info('Upload a CSV file from the sidebar to view portfolio statistics. The selected dataset will persist across reloads.')