"""
Interface Streamlit - Analisis Regresi Logistik Biner
Status Penerima Bantuan PKH pada Data Kemiskinan Ekstrem

Cara menjalankan:
    pip install -r requirements.txt
    streamlit run app.py
"""

import os
import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score
)

# ======================================================================
# KONFIGURASI HALAMAN
# ======================================================================
st.set_page_config(
    page_title="Analisis Kemiskinan Ekstrem - PKH",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ======================================================================
# CSS — TEMA GELAP (selaras dengan tampilan regresi.py / DataLens)
# ======================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }

    /* ═══ BACKGROUND ═══ */
    .stApp {
        background: #0B1120;
        background-image:
            radial-gradient(ellipse 80% 50% at 20% 0%, rgba(14,165,233,0.12) 0%, transparent 60%),
            radial-gradient(ellipse 60% 40% at 80% 100%, rgba(99,102,241,0.10) 0%, transparent 60%);
        min-height: 100vh;
    }

    /* ═══ SIDEBAR ═══ */
    [data-testid="stSidebar"] {
        background: #0D1526 !important;
        border-right: 1px solid rgba(14,165,233,0.18);
    }
    [data-testid="stSidebar"] * { color: #CBD5E1 !important; }
    [data-testid="stSidebar"] .stFileUploader label,
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stNumberInput label {
        color: #38BDF8 !important;
        font-size: 0.75rem; font-weight: 700;
        letter-spacing: 0.1em; text-transform: uppercase;
    }
    [data-testid="stSidebar"] .stFileUploader,
    [data-testid="stSidebar"] .stSelectbox > div > div {
        background: rgba(14,165,233,0.07) !important;
        border: 1px solid rgba(14,165,233,0.25) !important;
        border-radius: 10px !important;
        color: #E2E8F0 !important;
    }

    /* ── File uploader ── */
    [data-testid="stFileUploader"] section,
    [data-testid="stFileUploadDropzone"],
    .stFileUploader > div,
    .stFileUploader section,
    .stFileUploader section > div,
    [data-testid="stFileUploadDropzone"] > div {
        background: rgba(14,165,233,0.07) !important;
        border: 1px dashed rgba(14,165,233,0.35) !important;
        border-radius: 10px !important;
    }
    [data-testid="stFileUploadDropzone"] span,
    [data-testid="stFileUploadDropzone"] p,
    [data-testid="stFileUploadDropzone"] small,
    .stFileUploader span,
    .stFileUploader p,
    .stFileUploader small {
        color: #94A3B8 !important;
    }
    [data-testid="stFileUploadDropzone"] button,
    .stFileUploader button {
        background: rgba(14,165,233,0.15) !important;
        border: 1px solid rgba(14,165,233,0.4) !important;
        color: #7DD3FC !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }

    /* ── Selectbox / dropdown popover ── */
    [data-baseweb="popover"],
    [data-baseweb="menu"],
    [role="listbox"],
    [data-baseweb="select"] ul {
        background: #0D1526 !important;
        border: 1px solid rgba(14,165,233,0.25) !important;
        border-radius: 10px !important;
    }
    [data-baseweb="option"] { background: #0D1526 !important; color: #CBD5E1 !important; }
    [data-baseweb="option"]:hover,
    [aria-selected="true"][data-baseweb="option"] {
        background: rgba(14,165,233,0.15) !important;
        color: #38BDF8 !important;
    }

    /* ── Slider / number input ── */
    .stSlider [data-baseweb="slider"] > div > div { background: rgba(14,165,233,0.2) !important; }
    .stSlider [role="slider"] {
        background: linear-gradient(135deg, #0284C7, #4F46E5) !important;
        border: 2px solid #38BDF8 !important;
    }
    .stNumberInput button {
        background: rgba(14,165,233,0.1) !important;
        border: 1px solid rgba(14,165,233,0.2) !important;
        color: #7DD3FC !important;
        border-radius: 6px !important;
    }

    /* ═══ INPUT TEXT & NUMBER — high contrast ═══ */
    .stTextArea textarea,
    .stTextInput input,
    .stNumberInput input {
        background: #1E2D45 !important;
        border: 1.5px solid rgba(56,189,248,0.45) !important;
        border-radius: 10px !important;
        color: #F1F5F9 !important;
        font-size: 1rem !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        caret-color: #38BDF8 !important;
    }
    .stTextArea textarea:focus,
    .stTextInput input:focus,
    .stNumberInput input:focus {
        border-color: #38BDF8 !important;
        box-shadow: 0 0 0 3px rgba(56,189,248,0.22) !important;
        outline: none !important;
    }
    ::placeholder { color: #64748B !important; opacity: 1 !important; }

    /* ═══ MENU NAVIGASI DI SIDEBAR (pengganti tab di halaman utama) ═══ */
    [data-testid="stSidebar"] .stRadio [role="radiogroup"] {
        gap: 0.35rem;
    }
    [data-testid="stSidebar"] .stRadio [role="radiogroup"] label {
        background: rgba(14,165,233,0.05) !important;
        border: 1px solid rgba(14,165,233,0.16) !important;
        border-radius: 10px !important;
        padding: 0.55rem 0.85rem !important;
        width: 100%;
        transition: all 0.15s ease;
        cursor: pointer;
    }
    [data-testid="stSidebar"] .stRadio [role="radiogroup"] label:hover {
        background: rgba(14,165,233,0.13) !important;
        border-color: rgba(56,189,248,0.45) !important;
    }
    [data-testid="stSidebar"] .stRadio [role="radiogroup"] label:has(input:checked) {
        background: linear-gradient(135deg, #0284C7, #4F46E5) !important;
        border-color: transparent !important;
        box-shadow: 0 4px 16px rgba(14,165,233,0.35);
    }
    [data-testid="stSidebar"] .stRadio [role="radiogroup"] label:has(input:checked) p,
    [data-testid="stSidebar"] .stRadio [role="radiogroup"] label:has(input:checked) [data-testid="stMarkdownContainer"] {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }
    [data-testid="stSidebar"] .stRadio [role="radiogroup"] p {
        color: #CBD5E1 !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
    }

    /* ═══ TEKS UMUM DI AREA UTAMA — kontras di atas latar gelap ═══ */
    section.main [data-testid="stMarkdownContainer"] h1,
    section.main [data-testid="stMarkdownContainer"] h2,
    section.main [data-testid="stMarkdownContainer"] h3,
    section.main [data-testid="stMarkdownContainer"] h4,
    section.main [data-testid="stMarkdownContainer"] h5,
    section.main [data-testid="stMarkdownContainer"] h6,
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] h3,
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] h4,
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] h5,
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] h6 {
        color: #F1F5F9 !important;
        font-family: 'Space Grotesk', sans-serif !important;
    }
    section.main [data-testid="stMarkdownContainer"] p,
    section.main [data-testid="stMarkdownContainer"] li,
    section.main [data-testid="stMarkdownContainer"] span,
    section.main [data-testid="stMarkdownContainer"] strong,
    section.main [data-testid="stMarkdownContainer"] em,
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] li,
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] span,
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] strong,
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] em {
        color: #CBD5E1 !important;
    }
    [data-testid="stCaptionContainer"] p,
    [data-testid="stCaptionContainer"] { color: #64748B !important; }
    [data-testid="stExpander"] summary,
    .streamlit-expanderHeader p,
    .streamlit-expanderHeader [data-testid="stMarkdownContainer"] p {
        color: #BAE6FD !important; font-weight: 600 !important;
    }
    [data-testid="stExpanderDetails"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stExpanderDetails"] [data-testid="stMarkdownContainer"] li {
        color: #CBD5E1 !important;
    }
    section.main label, [data-testid="stMain"] label,
    section.main [data-testid="stWidgetLabel"] p, [data-testid="stMain"] [data-testid="stWidgetLabel"] p {
        color: #7DD3FC !important; font-weight: 600 !important; font-size: 0.85rem !important;
    }
    [data-testid="stText"], .stText, [data-testid="stJson"] {
        color: #E2E8F0 !important;
        background: rgba(14,165,233,0.05) !important;
        border-radius: 10px !important;
    }
    [data-testid="stJson"] * { color: #E2E8F0 !important; }

    /* ═══ SELECTBOX / MULTISELECT — berlaku di seluruh halaman, bukan cuma sidebar ═══ */
    .stSelectbox > div > div, .stMultiSelect > div > div {
        background: #1E2D45 !important;
        border: 1px solid rgba(56,189,248,0.4) !important;
        border-radius: 10px !important;
    }
    .stSelectbox [data-baseweb="select"] span,
    .stMultiSelect [data-baseweb="select"] span,
    .stSelectbox div, .stMultiSelect div {
        color: #F1F5F9 !important;
    }
    .stSelectbox svg, .stMultiSelect svg { fill: #7DD3FC !important; }
    .stMultiSelect [data-baseweb="tag"] {
        background: rgba(14,165,233,0.28) !important;
        border: 1px solid rgba(14,165,233,0.5) !important;
    }
    .stMultiSelect [data-baseweb="tag"] span { color: #F1F5F9 !important; }

    /* ═══ RADIO / CHECKBOX (di luar sidebar) ═══ */
    .stRadio [data-testid="stMarkdownContainer"] p,
    .stCheckbox [data-testid="stMarkdownContainer"] p {
        color: #E2E8F0 !important;
    }

    /* ═══ LANDING PAGE (halaman utama / welcome) ═══ */
    .landing-wrap {
        position: relative; overflow: hidden;
        min-height: 82vh;
        display: flex; flex-direction: column; align-items: center; justify-content: center;
        text-align: center;
        border-radius: 28px;
        padding: 3.5rem 2.5rem;
        background: linear-gradient(160deg,
            rgba(2,132,199,0.16) 0%,
            rgba(79,70,229,0.12) 45%,
            rgba(16,185,129,0.09) 100%);
        border: 1px solid rgba(14,165,233,0.25);
        animation: landingFadeIn 0.7s ease;
    }
    @keyframes landingFadeIn {
        from { opacity: 0; transform: translateY(14px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .landing-glow-1 {
        position: absolute; top: -120px; left: -100px;
        width: 420px; height: 380px; border-radius: 50%;
        background: radial-gradient(ellipse, rgba(14,165,233,0.25) 0%, transparent 70%);
        pointer-events: none; animation: floatGlow 8s ease-in-out infinite;
    }
    .landing-glow-2 {
        position: absolute; bottom: -100px; right: -80px;
        width: 380px; height: 320px; border-radius: 50%;
        background: radial-gradient(ellipse, rgba(99,102,241,0.22) 0%, transparent 70%);
        pointer-events: none; animation: floatGlow 9s ease-in-out infinite reverse;
    }
    .landing-glow-3 {
        position: absolute; top: 40%; left: 50%;
        width: 300px; height: 300px; border-radius: 50%;
        background: radial-gradient(ellipse, rgba(16,185,129,0.14) 0%, transparent 70%);
        pointer-events: none; transform: translate(-50%, -50%);
    }
    @keyframes floatGlow {
        0%, 100% { transform: translate(0, 0); }
        50% { transform: translate(20px, -18px); }
    }
    .landing-logo {
        width: 84px; height: 84px; margin-bottom: 1.4rem;
        background: linear-gradient(135deg, #0284C7, #4F46E5, #10B981);
        background-size: 200% 200%;
        animation: gradientShift 5s ease infinite;
        border-radius: 22px; display: flex; align-items: center; justify-content: center;
        font-size: 2.6rem;
        box-shadow: 0 10px 40px rgba(14,165,233,0.4);
    }
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .landing-badge {
        display: inline-flex; align-items: center; gap: 0.4rem;
        background: rgba(14,165,233,0.15);
        border: 1px solid rgba(56,189,248,0.4);
        color: #7DD3FC;
        font-size: 0.72rem; font-weight: 700;
        letter-spacing: 0.16em; text-transform: uppercase;
        padding: 0.4rem 1.1rem; border-radius: 100px; margin-bottom: 1.3rem;
    }
    .landing-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 3rem; font-weight: 800;
        color: #F1F5F9; line-height: 1.15; margin: 0.3rem 0 0.6rem;
        letter-spacing: -0.02em;
    }
    .landing-title span {
        background: linear-gradient(135deg, #38BDF8, #818CF8, #34D399);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    }
    .landing-sub {
        color: #94A3B8; font-size: 1.08rem; line-height: 1.7;
        max-width: 640px; margin: 0 auto 0.4rem;
    }
    .landing-name {
        color: #7DD3FC; font-weight: 700;
    }
    .landing-features {
        display: flex; justify-content: center; gap: 1rem; margin: 2.2rem 0 2.4rem;
        flex-wrap: wrap; max-width: 780px;
    }
    .landing-feature {
        background: rgba(13,21,38,0.55);
        border: 1px solid rgba(14,165,233,0.2);
        border-radius: 16px; padding: 1.1rem 1.3rem;
        width: 165px; text-align: center;
        transition: transform 0.2s, box-shadow 0.2s, border-color 0.2s;
    }
    .landing-feature:hover {
        transform: translateY(-4px);
        border-color: rgba(56,189,248,0.5);
        box-shadow: 0 10px 26px rgba(14,165,233,0.18);
    }
    .landing-feature-icon { font-size: 1.7rem; margin-bottom: 0.5rem; }
    .landing-feature-title {
        font-family: 'Space Grotesk', sans-serif;
        color: #F1F5F9; font-weight: 700; font-size: 0.87rem; margin-bottom: 0.25rem;
    }
    .landing-feature-desc { color: #64748B; font-size: 0.74rem; line-height: 1.5; }
    .landing-cta-hint {
        margin-top: 0.9rem; color: #475569; font-size: 0.78rem;
    }

    /* ═══ METRIC ═══ */
    [data-testid="stMetric"] {
        background: rgba(14,165,233,0.07) !important;
        border: 1px solid rgba(14,165,233,0.2) !important;
        border-top: 3px solid #0EA5E9 !important;
        border-radius: 12px; padding: 1rem 1.2rem;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(14,165,233,0.12);
    }
    [data-testid="stMetricLabel"] {
        color: #38BDF8 !important;
        font-size: 0.73rem !important; font-weight: 700 !important; letter-spacing: 0.08em;
    }
    [data-testid="stMetricValue"] {
        color: #F1F5F9 !important;
        font-family: 'Space Grotesk', sans-serif !important;
    }

    /* ═══ BUTTONS ═══ */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #0284C7 0%, #4F46E5 100%) !important;
        border: none !important; border-radius: 10px !important;
        color: #FFFFFF !important; font-weight: 700 !important;
        font-size: 0.9rem !important; padding: 0.65rem 1.4rem !important;
        box-shadow: 0 4px 18px rgba(14,165,233,0.3) !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 24px rgba(14,165,233,0.45) !important;
    }
    .stButton > button[kind="secondary"] {
        background: rgba(14,165,233,0.08) !important;
        border: 1px solid rgba(14,165,233,0.3) !important;
        color: #7DD3FC !important; border-radius: 10px !important; font-weight: 600 !important;
    }

    /* ═══ ALERTS ═══ */
    .stSuccess { background: rgba(16,185,129,0.12) !important; color: #A7F3D0 !important; border-radius: 10px !important; border-left: 3px solid #10B981 !important; }
    .stInfo    { background: rgba(14,165,233,0.12) !important; color: #BAE6FD !important; border-radius: 10px !important; border-left: 3px solid #0EA5E9 !important; }
    .stWarning { background: rgba(245,158,11,0.12) !important; color: #FDE68A !important; border-radius: 10px !important; border-left: 3px solid #F59E0B !important; }
    .stError   { background: rgba(239,68,68,0.12) !important; color: #FECACA !important; border-radius: 10px !important; border-left: 3px solid #EF4444 !important; }

    /* ═══ DATAFRAME ═══ */
    .stDataFrame { border-radius: 12px !important; overflow: hidden; }
    [data-testid="stDataFrame"] > div {
        background: rgba(14,165,233,0.04) !important;
        border: 1px solid rgba(14,165,233,0.18) !important;
        border-radius: 12px !important;
    }

    /* ═══ CUSTOM DATA TABLE (pengganti tampilan bawaan ala Colab) ═══ */
    .data-table-wrap {
        width: 100%;
        overflow-x: auto;
        border: 1px solid rgba(14,165,233,0.18);
        border-radius: 12px;
        background: rgba(14,165,233,0.04);
        margin: 0.6rem 0 1rem 0;
    }
    .data-table-wrap::-webkit-scrollbar { height: 8px; }
    .data-table-wrap::-webkit-scrollbar-thumb { background: rgba(14,165,233,0.3); border-radius: 8px; }
    .data-table-wrap::-webkit-scrollbar-track { background: transparent; }
    .data-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.85rem;
        white-space: nowrap;
    }
    .data-table thead tr { background: rgba(2,132,199,0.2); }
    .data-table th {
        position: sticky; top: 0;
        padding: 0.7rem 1rem; text-align: left;
        font-weight: 700; font-size: 0.72rem;
        letter-spacing: 0.08em; text-transform: uppercase;
        color: #BAE6FD; border-bottom: 1px solid rgba(14,165,233,0.25);
        background: #0D1526;
    }
    .data-table td {
        padding: 0.6rem 1rem; color: #E2E8F0;
        border-bottom: 1px solid rgba(14,165,233,0.08);
        font-family: 'JetBrains Mono', monospace; font-size: 0.82rem;
    }
    .data-table tbody tr:nth-child(even) { background: rgba(14,165,233,0.03); }
    .data-table tbody tr:hover { background: rgba(14,165,233,0.09); }
    .data-table td:first-child { color: #7DD3FC; font-weight: 700; }
    .data-table-caption {
        padding: 0.5rem 1rem 0.7rem 1rem;
        color: #64748B; font-size: 0.78rem;
        border-top: 1px solid rgba(14,165,233,0.1);
    }

    /* ═══ KV CHIPS (pengganti st.write(dict) / JSON viewer bawaan) ═══ */
    .kv-chip-wrap {
        display: flex; flex-wrap: wrap; gap: 0.5rem;
        margin: 0.5rem 0 1rem 0;
    }
    .kv-chip {
        display: inline-flex; align-items: center; gap: 0.4rem;
        background: #0D1526;
        border: 1px solid rgba(14,165,233,0.25);
        border-radius: 8px;
        padding: 0.35rem 0.7rem;
        font-size: 0.8rem;
    }
    .kv-chip .kv-key { color: #94A3B8; }
    .kv-chip .kv-arrow { color: #38BDF8; }
    .kv-chip .kv-val {
        color: #7DD3FC; font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        background: rgba(14,165,233,0.12);
        padding: 0.05rem 0.4rem; border-radius: 4px;
    }

    /* ═══ EXPANDER ═══ */
    .streamlit-expanderHeader {
        background: rgba(14,165,233,0.07) !important;
        border: 1px solid rgba(14,165,233,0.18) !important;
        border-radius: 10px !important;
        color: #BAE6FD !important; font-weight: 600 !important;
    }
    .streamlit-expanderContent {
        background: rgba(11,17,32,0.8) !important;
        border: 1px solid rgba(14,165,233,0.12) !important;
        border-top: none !important; border-radius: 0 0 10px 10px !important;
    }

    /* ═══ DOWNLOAD ═══ */
    .stDownloadButton > button {
        background: rgba(16,185,129,0.12) !important;
        border: 1px solid rgba(16,185,129,0.35) !important;
        color: #A7F3D0 !important; border-radius: 10px !important;
        font-weight: 600 !important; font-size: 0.85rem !important;
    }
    .stDownloadButton > button:hover {
        background: rgba(16,185,129,0.2) !important; transform: translateY(-1px);
    }

    /* ═══ SPINNER / PROGRESS ═══ */
    .stSpinner > div { border-color: #0EA5E9 transparent transparent transparent !important; }
    .stProgress > div > div > div { background: linear-gradient(90deg, #0EA5E9, #6366F1) !important; }

    /* ── SIDEBAR BRAND ── */
    .sb-brand {
        display: flex; align-items: center; gap: 0.7rem;
        padding-bottom: 1.2rem;
        border-bottom: 1px solid rgba(14,165,233,0.15);
        margin-bottom: 1.2rem;
    }
    .sb-logo {
        width: 40px; height: 40px;
        background: linear-gradient(135deg, #0284C7, #4F46E5);
        border-radius: 10px; display: flex; align-items: center;
        justify-content: center; font-size: 1.3rem;
        box-shadow: 0 4px 12px rgba(14,165,233,0.25);
    }
    .sb-name {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.1rem; font-weight: 700; color: #F1F5F9 !important;
    }
    .sb-sub { font-size: 0.68rem; color: #64748B !important; text-transform: uppercase; letter-spacing: 0.1em; }
    .sb-sect {
        font-size: 0.67rem; font-weight: 700; letter-spacing: 0.14em;
        text-transform: uppercase; color: #0EA5E9 !important;
        margin: 1.2rem 0 0.5rem; display: block;
    }
    .sb-info {
        background: rgba(14,165,233,0.07);
        border: 1px solid rgba(14,165,233,0.18);
        border-radius: 10px; padding: 0.9rem 1rem;
        font-size: 0.8rem; color: #94A3B8 !important; line-height: 1.6; margin-top: 1rem;
    }

    /* ── HERO ── */
    .hero-wrap {
        position: relative; overflow: hidden;
        border-radius: 24px; margin-bottom: 2rem;
        padding: 2.6rem 3.2rem; text-align: center;
        background: linear-gradient(135deg,
            rgba(2,132,199,0.15) 0%,
            rgba(79,70,229,0.10) 50%,
            rgba(16,185,129,0.08) 100%);
        border: 1px solid rgba(14,165,233,0.25);
    }
    .hero-glow-l {
        position: absolute; top: -80px; left: -60px;
        width: 320px; height: 260px; border-radius: 50%;
        background: radial-gradient(ellipse, rgba(14,165,233,0.22) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-glow-r {
        position: absolute; bottom: -60px; right: -40px;
        width: 240px; height: 200px; border-radius: 50%;
        background: radial-gradient(ellipse, rgba(99,102,241,0.18) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-badge {
        display: inline-flex; align-items: center; gap: 0.4rem;
        background: rgba(14,165,233,0.15);
        border: 1px solid rgba(56,189,248,0.4);
        color: #7DD3FC;
        font-size: 0.7rem; font-weight: 700;
        letter-spacing: 0.14em; text-transform: uppercase;
        padding: 0.35rem 1rem; border-radius: 100px; margin-bottom: 1rem;
    }
    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.4rem; font-weight: 700;
        color: #F1F5F9; line-height: 1.1; margin: 0.5rem 0;
        letter-spacing: -0.02em;
    }
    .hero-title span {
        background: linear-gradient(135deg, #38BDF8, #818CF8, #34D399);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    }
    .hero-sub {
        color: #94A3B8; font-size: 1rem; margin-top: 0.7rem;
        max-width: 620px; margin-left: auto; margin-right: auto; line-height: 1.6;
    }
    .hero-chips {
        display: flex; justify-content: center; gap: 0.5rem; margin-top: 1.4rem; flex-wrap: wrap;
    }
    .hero-chip {
        display: inline-flex; align-items: center; gap: 0.35rem;
        background: rgba(14,165,233,0.08);
        border: 1px solid rgba(14,165,233,0.2);
        color: #94A3B8; font-size: 0.78rem; font-weight: 500;
        padding: 0.3rem 0.85rem; border-radius: 100px;
    }
    .hero-chip-arr { color: rgba(14,165,233,0.4); font-size: 0.9rem; padding-bottom: 0; }

    /* ── SECTION TITLE ── */
    .sec-title {
        display: flex; align-items: center; gap: 0.6rem;
        font-size: 0.75rem; font-weight: 700;
        letter-spacing: 0.13em; text-transform: uppercase;
        color: #38BDF8; margin-bottom: 1.2rem; margin-top: 0.5rem;
    }
    .sec-title::after {
        content: ''; flex: 1; height: 1px;
        background: linear-gradient(to right, rgba(14,165,233,0.35), transparent);
        border-radius: 1px;
    }

    /* ── BADGE (status prediksi) ── */
    .badge-in {
        background: rgba(16,185,129,0.18); color: #6EE7B7;
        border: 1px solid rgba(16,185,129,0.4);
        font-size: 0.72rem; font-weight: 700; padding: 0.22rem 0.7rem; border-radius: 100px;
    }
    .badge-out {
        background: rgba(14,165,233,0.15); color: #7DD3FC;
        border: 1px solid rgba(14,165,233,0.35);
        font-size: 0.72rem; font-weight: 700; padding: 0.22rem 0.7rem; border-radius: 100px;
    }

    /* ── EMPTY STATE ── */
    .empty-wrap { text-align: center; padding: 4rem 2rem; }
    .empty-icon { font-size: 4rem; margin-bottom: 1.2rem; }
    .empty-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.6rem; font-weight: 700; color: #F1F5F9; margin-bottom: 0.5rem;
    }
    .empty-sub { color: #64748B; font-size: 0.95rem; }
    .empty-sub code {
        color: #38BDF8; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem;
        background: rgba(14,165,233,0.14); padding: 0.12rem 0.45rem; border-radius: 4px;
    }

    /* ── FOOTER ── */
    .footer {
        text-align: center; padding: 2rem 1rem 1rem;
        border-top: 1px solid rgba(14,165,233,0.1); margin-top: 3rem;
        color: #475569; font-size: 0.78rem;
    }
    .footer span { color: #0EA5E9; }

    /* ═══════════════════════════════════════════════════════════
       TAB "TENTANG" — profil, identitas & informasi sistem
       (selaras dengan tema sky/indigo/emerald di atas)
       ═══════════════════════════════════════════════════════════ */
    .tentang-hero {
        position: relative; overflow: hidden;
        border-radius: 24px; padding: 2.2rem 2.4rem; margin-bottom: 1.6rem;
        background: linear-gradient(135deg,
            rgba(2,132,199,0.16) 0%, rgba(79,70,229,0.12) 55%, rgba(16,185,129,0.10) 100%);
        border: 1px solid rgba(14,165,233,0.25);
        display: flex; align-items: center; gap: 1.6rem; flex-wrap: wrap;
    }
    .tentang-hero .th-icon {
        width: 64px; height: 64px; border-radius: 18px; flex-shrink: 0;
        background: linear-gradient(135deg, #0284C7, #4F46E5);
        display: flex; align-items: center; justify-content: center;
        font-size: 1.8rem; box-shadow: 0 8px 26px rgba(14,165,233,0.35);
    }
    .tentang-hero h1 {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.7rem; font-weight: 800; color: #F1F5F9; margin: 0 0 0.3rem;
    }
    .tentang-hero p { color: #94A3B8; font-size: 0.92rem; margin: 0; line-height: 1.6; }
    .tentang-chip {
        display: inline-block; margin-top: 0.6rem;
        background: rgba(14,165,233,0.1); border: 1px solid rgba(14,165,233,0.25);
        color: #7DD3FC; font-size: 0.72rem; font-weight: 600;
        padding: 0.25rem 0.8rem; border-radius: 100px;
    }

    .profile-card {
        display: flex; align-items: center; gap: 1.2rem;
        padding: 1.4rem 1.6rem; border-radius: 18px; margin-bottom: 1rem;
        background: linear-gradient(135deg, rgba(13,21,38,0.9), rgba(14,165,233,0.05));
        border: 1px solid rgba(14,165,233,0.2);
    }
    .profile-avatar {
        width: 68px; height: 68px; border-radius: 50%; flex-shrink: 0;
        background: linear-gradient(135deg, #10B981, #0EA5E9);
        display: flex; align-items: center; justify-content: center;
        font-size: 1.6rem; font-weight: 800; color: #06231d;
        box-shadow: 0 0 0 3px rgba(16,185,129,0.2);
    }
    .profile-info h3 { font-family: 'Space Grotesk', sans-serif; font-size: 1.15rem; font-weight: 800; color: #F1F5F9; margin: 0 0 0.35rem; }
    .profile-meta { display: flex; flex-wrap: wrap; gap: 0.3rem 1rem; font-size: 0.8rem; color: #94A3B8; }
    .profile-meta span { display: inline-flex; align-items: center; gap: 0.35rem; }

    .about-card {
        display: flex; gap: 0.9rem; align-items: flex-start;
        padding: 1.1rem 1.3rem; border-radius: 14px; margin-bottom: 0.7rem;
        background: rgba(13,21,38,0.55); border: 1px solid rgba(14,165,233,0.15);
        border-left: 3px solid var(--a-color, #0EA5E9);
    }
    .about-card .ic {
        font-size: 1.2rem; width: 36px; height: 36px; border-radius: 10px; flex-shrink: 0;
        background: rgba(14,165,233,0.1); display: flex; align-items: center; justify-content: center;
    }
    .about-card h4 { font-size: 0.9rem; font-weight: 700; color: var(--a-color, #F1F5F9); margin: 0 0 0.4rem; }
    .about-card p { font-size: 0.83rem; color: #94A3B8; line-height: 1.7; margin: 0; }
    .about-card ul { margin: 0; padding-left: 0; list-style: none; }
    .about-card li {
        font-size: 0.83rem; color: #94A3B8; line-height: 1.9; padding-left: 1.4rem; position: relative;
    }
    .about-card li::before {
        content: '✓'; position: absolute; left: 0; top: 2px;
        color: #10B981; font-weight: 700; background: rgba(16,185,129,0.15);
        width: 15px; height: 15px; border-radius: 50%;
        display: inline-flex; align-items: center; justify-content: center; font-size: 0.6rem;
    }

    .dosen-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 0.9rem; margin-top: 0.7rem; }
    .dosen-card {
        background: rgba(13,21,38,0.6); border: 1px solid rgba(14,165,233,0.18);
        border-radius: 14px; padding: 1rem 1.1rem; transition: all 0.2s ease;
    }
    .dosen-card:hover { border-color: rgba(56,189,248,0.4); transform: translateY(-2px); }
    .dosen-card .role { font-size: 0.65rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #F59E0B; margin-bottom: 0.35rem; }
    .dosen-card .name { font-size: 0.95rem; font-weight: 700; color: #F1F5F9; }
    .dosen-card .detail { font-size: 0.75rem; color: #64748B; margin-top: 0.25rem; }

    .step-flow { display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 0.9rem 0; }
    .step-pill {
        font-size: 0.75rem; font-weight: 600; padding: 0.45rem 0.85rem; border-radius: 100px;
        background: rgba(14,165,233,0.07); border: 1px solid rgba(14,165,233,0.2); color: #94A3B8;
    }
    .step-pill b { color: #38BDF8; }
</style>
""", unsafe_allow_html=True)

DEFAULT_DATA_PATH = os.path.join(os.path.dirname(__file__), "DATA_KEMISKINAN_EKSTREM.xlsx")

# ======================================================================
# HALAMAN UTAMA (LANDING PAGE) — tampil sebelum masuk ke sistem
# ======================================================================
st.session_state.setdefault("masuk_sistem", False)


def _mulai_analisis():
    st.session_state["masuk_sistem"] = True


def _kembali_beranda():
    st.session_state["masuk_sistem"] = False


if not st.session_state["masuk_sistem"]:
    st.markdown("""
    <div class="landing-wrap">
        <div class="landing-glow-1"></div>
        <div class="landing-glow-2"></div>
        <div class="landing-glow-3"></div>
        <div class="landing-logo">📊</div>
        <div class="landing-badge">✦ Statistika Inferensial · Regresi Logistik Biner</div>
        <div class="landing-title">Selamat Datang di <span>PKH Analytics</span></div>
        <p class="landing-sub">
            Platform analisis <span class="landing-name">Status Penerima Bantuan PKH</span>
            pada Data Kemiskinan Ekstrem — mulai dari pembersihan data, encoding fitur,
            pemodelan dengan K-Fold Cross Validation + SMOTE + Regresi Logistik,
            interpretasi statistik, evaluasi performa, hingga prediksi data baru.
            Semua dalam satu tempat, siap dijalankan hanya dengan beberapa klik.
        </p>
        <div class="landing-features">
            <div class="landing-feature">
                <div class="landing-feature-icon">🧹</div>
                <div class="landing-feature-title">Preprocessing</div>
                <div class="landing-feature-desc">Bersihkan anomali, missing value & duplikat otomatis</div>
            </div>
            <div class="landing-feature">
                <div class="landing-feature-icon">🤖</div>
                <div class="landing-feature-title">K-Fold + SMOTE</div>
                <div class="landing-feature-desc">Model regresi logistik yang seimbang & teruji</div>
            </div>
            <div class="landing-feature">
                <div class="landing-feature-icon">📈</div>
                <div class="landing-feature-title">Interpretasi</div>
                <div class="landing-feature-desc">Koefisien, p-value & signifikansi tiap variabel</div>
            </div>
            <div class="landing-feature">
                <div class="landing-feature-icon">🔮</div>
                <div class="landing-feature-title">Prediksi</div>
                <div class="landing-feature-desc">Terapkan model ke data baru secara instan</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    _sp1, _mid, _sp2 = st.columns([1, 1, 1])
    with _mid:
        st.button("🚀 Mulai Analisis", type="primary", width='stretch',
                   on_click=_mulai_analisis, key="btn_mulai_analisis")
    st.markdown('<div class="landing-cta-hint" style="text-align:center;">'
                'Klik tombol di atas untuk masuk ke sistem analisis.</div>',
                unsafe_allow_html=True)

    st.markdown(
        '<div class="footer">© 2026 <span>PKH Analytics</span> — '
        'Analisis Regresi Logistik Biner · Dibuat dengan ♥ menggunakan Streamlit</div>',
        unsafe_allow_html=True,
    )
    st.stop()

# ======================================================================
# DEFINISI KOLOM & MAPPING (sesuai regresi.py, dengan sedikit perbaikan
# agar semua kategori yang benar-benar muncul di data ikut tertangani)
# ======================================================================
KOLOM_KATEGORIKAL = [
    'Penerima PKH', 'Jenis Kelamin Kepala Keluarga', 'Pekerjaan Kepala Keluarga',
    'Pendidikan Kepala Keluarga', 'Status Kawin Kepala Keluarga', 'Kepemilikan Rumah',
    'Jenis Atap', 'Jenis Dinding', 'Jenis Lantai', 'Sumber Penerangan',
    'Daya Listrik Terpasang', 'Bahan Bakar Memasak', 'Sumber Air Minum',
    'Memiliki Fasilitas BAB', 'Resiko Stunting'
]

ALL_FEATURES = [
    'Pendidikan Kepala Keluarga', 'Daya Listrik Terpasang', 'Jenis Kelamin Kepala Keluarga',
    'Pekerjaan Kepala Keluarga', 'Status Kawin Kepala Keluarga', 'Kepemilikan Rumah',
    'Jenis Lantai', 'Jenis Dinding', 'Jenis Atap', 'Sumber Air Minum',
    'Bahan Bakar Memasak', 'Sumber Penerangan', 'Memiliki Fasilitas BAB', 'Resiko Stunting'
]

# Kategori valid per kolom -> dipakai untuk mendeteksi & membersihkan nilai
# anomali (contoh: kolom 'Resiko Stunting' pada data asli ternyata memuat
# nilai '450 VA' yang jelas salah input / kolom bergeser).
VALID_CATEGORIES = {
    'Penerima PKH': ['Penerima', 'Bukan Penerima'],
    'Jenis Kelamin Kepala Keluarga': ['Laki-laki', 'Perempuan'],
    'Pekerjaan Kepala Keluarga': ['Petani', 'Swasta', 'Pekerja Lepas', 'Pensiunan',
                                   'Belum/Tidak Bekerja', 'Nelayan', 'Pedagang'],
    'Pendidikan Kepala Keluarga': ['Tamat SD', 'Tidak Tamat SD', 'Tidak Sekolah', 'Tamat SMA',
                                    'Tamat SMP', 'Tamat PT/akademi', 'Masih SLTP/sederajat',
                                    'Masih SD/sederajat', 'Masih SLTA/sederajat', 'Masih PT/akademi'],
    'Status Kawin Kepala Keluarga': ['Kawin', 'Cerai mati', 'Cerai hidup', 'Belum kawin'],
    'Kepemilikan Rumah': ['Milik Sendiri', 'Bebas/Sewa/Menumpang', 'Kontrak/Sewa', 'Dinas', 'Lainnya'],
    'Jenis Atap': ['Asbes/Seng', 'Beton', 'Jerami/Ijuk/Rumbia/ Daun-daunan', 'Lainnya',
                   'Kayu/Sirap', 'Genteng', 'Bambu'],
    'Jenis Dinding': ['Plesteran anyaman bambu/kawat/Lainnya', 'Tembok', 'Kayu/Papan', 'Bambu'],
    'Jenis Lantai': ['Semen', 'Tanah', 'Keramik/Granit/Marmer/Ubin/Tegel/Teraso', 'Lainnya',
                     'Kayu/Papan', 'Bambu'],
    'Sumber Penerangan': ['Listrik PLN', 'Non Listrik', 'Listrik Bukan PLN'],
    'Daya Listrik Terpasang': ['450 VA', '900 VA', 'Tidak Ada'],
    'Bahan Bakar Memasak': ['Arang/Kayu', 'Minyak Tanah', 'Listrik/Gas', 'Lainnya'],
    'Sumber Air Minum': ['Sumur Terlindung', 'Ledeng/PAM', 'Air Permukaan (Sungai, Danau, dll)',
                          'Sumur Bor', 'Sumur Tidak Terlindung', 'Air Kemasan/Isi Ulang',
                          'Lainnya', 'Air Hujan'],
    'Memiliki Fasilitas BAB': ['Ya, dengan Septictank', 'Tidak, Jamban Umum/Bersama',
                                'Ya, Tanpa Septictank', 'Lainnya'],
    'Resiko Stunting': ['Beresiko Stunting', 'Bukan Target Sasaran', 'Tidak Beresiko Stunting'],
}

# ── Palet warna tema (selaras dengan regresi.py / DataLens) ──
SKY, INDIGO, EMRLD, AMBER, BG_DARK, PANEL = "#0EA5E9", "#6366F1", "#10B981", "#F59E0B", "#0B1120", "#0D1526"
STATUS_COLOR_MAP = {
    "Penerima": EMRLD, "Bukan Penerima": SKY,
    "Penerima PKH": EMRLD, "Bukan Penerima PKH": SKY,
}


def style_plotly(fig, title=None):
    """Terapkan tema gelap (transparan + palet sky/indigo/emerald) ke figure Plotly,
    supaya selaras secara visual dengan tema di regresi.py."""
    judul = title if title is not None else (fig.layout.title.text if fig.layout.title else None)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans, sans-serif", color="#CBD5E1"),
        title=dict(text=judul, font=dict(family="Space Grotesk, sans-serif", color="#F1F5F9", size=16)),
        legend=dict(bgcolor="rgba(13,21,38,0.6)", bordercolor="rgba(14,165,233,0.2)", borderwidth=1, font=dict(color="#CBD5E1")),
        margin=dict(t=60, l=20, r=20, b=40),
        colorway=[SKY, EMRLD, INDIGO, AMBER, "#F472B6", "#818CF8"],
    )
    fig.update_xaxes(gridcolor="rgba(14,165,233,0.10)", zerolinecolor="rgba(14,165,233,0.15)", color="#94A3B8")
    fig.update_yaxes(gridcolor="rgba(14,165,233,0.10)", zerolinecolor="rgba(14,165,233,0.15)", color="#94A3B8")
    return fig


def render_kv_chips(mapping: dict):
    """Render dict sebagai daftar chip 'key → value' bertema gelap.
    Pengganti st.write(dict), yang dirender Streamlit lewat komponen JSON
    viewer bawaan dengan tema terkunci terang (tidak bisa ditimpa CSS
    halaman) sehingga tulisannya jadi nyaris tak terbaca di atas latar gelap."""
    items = "".join(
        f'<span class="kv-chip"><span class="kv-key">{k}</span>'
        f'<span class="kv-arrow">→</span><span class="kv-val">{v}</span></span>'
        for k, v in mapping.items()
    )
    st.markdown(f'<div class="kv-chip-wrap">{items}</div>', unsafe_allow_html=True)


def render_data_table(df: pd.DataFrame, max_rows: int | None = None, caption: str | None = None):
    """Render DataFrame sebagai tabel HTML bertema gelap (selaras dengan gaya
    'batch-table'/'calc-table' di regresi.py), menggantikan tampilan bawaan
    st.dataframe yang terlihat seperti tabel Colab (grid putih generik)."""
    tampil = df.head(max_rows) if max_rows else df
    total_baris = len(df)

    def _fmt(v):
        if isinstance(v, float):
            return f"{v:,.2f}"
        if pd.isna(v):
            return "-"
        return str(v)

    header_html = "".join(f"<th>{col}</th>" for col in tampil.columns)
    rows_html = []
    for _, row in tampil.iterrows():
        cells = "".join(f"<td>{_fmt(v)}</td>" for v in row)
        rows_html.append(f"<tr>{cells}</tr>")

    ket = caption or f"Menampilkan {len(tampil)} dari {total_baris} baris · {len(df.columns)} kolom"

    html = f"""
    <div class="data-table-wrap">
        <table class="data-table">
            <thead><tr>{header_html}</tr></thead>
            <tbody>{''.join(rows_html)}</tbody>
        </table>
        <div class="data-table-caption">{ket}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


PENDIDIKAN_ORDER = ['Tidak Sekolah', 'Masih SD/sederajat', 'Tidak Tamat SD', 'Tamat SD',
                     'Masih SLTP/sederajat', 'Tamat SMP', 'Masih SLTA/sederajat', 'Tamat SMA',
                     'Masih PT/akademi', 'Tamat PT/akademi']
PENDIDIKAN_MAPPING = {cat: i for i, cat in enumerate(PENDIDIKAN_ORDER)}
LISTRIK_MAPPING = {'Tidak Ada': 0, '450 VA': 1, '900 VA': 2}
KOLOM_NOMINAL = [c for c in ALL_FEATURES if c not in ['Pendidikan Kepala Keluarga', 'Daya Listrik Terpasang']]


# ======================================================================
# FUNGSI-FUNGSI PIPELINE
# ======================================================================
def bersihkan_data(df_raw: pd.DataFrame):
    """Tangani nilai anomali, missing value, dan duplikat. Kembalikan
    dataframe bersih + laporan perubahan untuk ditampilkan di UI."""
    df = df_raw.copy()
    laporan = []

    # 1) Deteksi & bersihkan nilai di luar kategori valid -> jadi NaN
    for kolom, kategori_valid in VALID_CATEGORIES.items():
        if kolom in df.columns:
            mask_anomali = ~df[kolom].isin(kategori_valid) & df[kolom].notna()
            n_anomali = int(mask_anomali.sum())
            if n_anomali > 0:
                nilai_aneh = sorted(df.loc[mask_anomali, kolom].astype(str).unique().tolist())
                laporan.append(f"Kolom **{kolom}**: {n_anomali} nilai anomali "
                                f"({', '.join(nilai_aneh)}) diganti menjadi hilang (NaN).")
                df.loc[mask_anomali, kolom] = np.nan

    missing_sebelum = int(df.isnull().sum().sum())

    # 2) Isi missing value kolom kategorikal (termasuk yang habis dibersihkan) dengan modus
    for col in df.columns:
        is_teks = pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col])
        if is_teks and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    missing_sesudah = int(df.isnull().sum().sum())

    # 3) Hapus duplikat
    n_sebelum = df.shape[0]
    df = df.drop_duplicates()
    n_duplikat = n_sebelum - df.shape[0]

    laporan.append(f"Total nilai hilang sebelum imputasi: {missing_sebelum} → sesudah imputasi modus: {missing_sesudah}.")
    laporan.append(f"Baris duplikat dihapus: {n_duplikat} baris.")

    return df.reset_index(drop=True), laporan


def encode_fitur(df: pd.DataFrame):
    """Ordinal + label encoding untuk seluruh fitur (ALL_FEATURES),
    persis alur di regresi.py."""
    X = df[ALL_FEATURES].copy().reset_index(drop=True)

    X['Pendidikan Kepala Keluarga'] = X['Pendidikan Kepala Keluarga'].map(PENDIDIKAN_MAPPING)
    X['Daya Listrik Terpasang'] = X['Daya Listrik Terpasang'].map(LISTRIK_MAPPING)

    encoder_dict = {}
    for col in KOLOM_NOMINAL:
        enc = LabelEncoder()
        X[col] = enc.fit_transform(X[col].astype(str))
        encoder_dict[col] = enc

    X = X.replace([np.inf, -np.inf], np.nan)
    for col in X.columns:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].mode()[0])

    return X, encoder_dict


def encode_target(df: pd.DataFrame):
    return df['Penerima PKH'].map({'Penerima': 1, 'Bukan Penerima': 0}).reset_index(drop=True)


def encode_data_baru(df_baru: pd.DataFrame, encoder_dict, mode_referensi: pd.DataFrame):
    """Encode data baru memakai encoder yang SUDAH di-fit pada data latih,
    persis pola X_new di regresi.py, dengan penanganan kategori yang belum
    pernah terlihat (unseen)."""
    X_new = df_baru[ALL_FEATURES].copy().reset_index(drop=True)

    X_new['Pendidikan Kepala Keluarga'] = X_new['Pendidikan Kepala Keluarga'].map(PENDIDIKAN_MAPPING)
    X_new['Daya Listrik Terpasang'] = X_new['Daya Listrik Terpasang'].map(LISTRIK_MAPPING)
    X_new['Pendidikan Kepala Keluarga'] = X_new['Pendidikan Kepala Keluarga'].fillna(
        pd.Series(PENDIDIKAN_MAPPING).mode()[0])
    X_new['Daya Listrik Terpasang'] = X_new['Daya Listrik Terpasang'].fillna(
        pd.Series(LISTRIK_MAPPING).mode()[0])

    for col in KOLOM_NOMINAL:
        enc = encoder_dict[col]
        unseen = set(X_new[col].astype(str)) - set(enc.classes_)
        if unseen:
            mode_val = mode_referensi[col].mode()[0]
            X_new[col] = X_new[col].astype(str).replace(list(unseen), mode_val)
        X_new[col] = enc.transform(X_new[col].astype(str))

    for col in X_new.columns:
        if X_new[col].isnull().any():
            X_new[col] = X_new[col].fillna(X_new[col].mode()[0])

    return X_new


def jalankan_kfold(X: pd.DataFrame, y: pd.Series, n_splits: int, random_state: int, progress_cb=None):
    """K-Fold CV: impute -> scale -> SMOTE (train only) -> Logistic Regression.
    Mengembalikan tabel hasil per fold + artefak model/scaler/imputer terbaik."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    hasil_fold = []
    best_fold, best_accuracy, best_model, best_scaler, best_imputer = None, 0, None, None, None
    best_train_idx, best_test_idx = None, None

    for fold, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        imputer = SimpleImputer(strategy='mean')
        X_train_imp = imputer.fit_transform(X_train)
        X_test_imp = imputer.transform(X_test)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imp)
        X_test_scaled = scaler.transform(X_test_imp)

        smote = SMOTE(random_state=random_state)
        X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

        model = LogisticRegression(max_iter=1000, random_state=random_state)
        model.fit(X_train_res, y_train_res)

        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        hasil_fold.append({
            "Fold": fold, "Akurasi": acc, "Presisi": prec, "Recall": rec, "F1-Score": f1,
            "N Train": len(train_idx), "N Test": len(test_idx)
        })

        if acc > best_accuracy:
            best_accuracy = acc
            best_fold = fold
            best_model = model
            best_scaler = scaler
            best_imputer = imputer
            best_train_idx = train_idx
            best_test_idx = test_idx

        if progress_cb:
            progress_cb(fold, n_splits)

    df_hasil = pd.DataFrame(hasil_fold)
    return {
        "df_hasil": df_hasil,
        "mean_accuracy": df_hasil["Akurasi"].mean(),
        "best_fold": best_fold,
        "best_accuracy": best_accuracy,
        "best_model": best_model,
        "best_scaler": best_scaler,
        "best_imputer": best_imputer,
        "best_train_idx": best_train_idx,
        "best_test_idx": best_test_idx,
    }


def fit_statsmodels_logit(X, y, train_idx, imputer, scaler):
    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    X_train_imp = imputer.transform(X_train)
    X_train_scaled = scaler.transform(X_train_imp)

    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
    X_train_const = sm.add_constant(X_train_scaled_df)

    logit_model = sm.Logit(y_train, X_train_const)
    result = logit_model.fit(disp=False)

    coef_df = result.summary2().tables[1].copy()

    def tanda_signifikansi(p):
        if p < 0.001:
            return '***'
        elif p < 0.01:
            return '**'
        elif p < 0.05:
            return '*'
        return ''

    coef_df['Signifikansi'] = coef_df['P>|z|'].apply(tanda_signifikansi)
    coef_df = coef_df.rename(columns={
        'Coef.': 'Koefisien', 'Std.Err.': 'Std. Error', 'z': 'z-value', 'P>|z|': 'p-value'
    })
    return result, coef_df.reset_index().rename(columns={'index': 'Variabel'})


# ======================================================================
# DAFTAR MENU (ditampilkan sebagai navigasi di sidebar kiri)
# ======================================================================
MENU_ITEMS = [
    "🔍 Eksplorasi Data", "🧹 Preprocessing & Encoding", "🤖 Training K-Fold",
    "📈 Interpretasi Model", "✅ Evaluasi Model Terbaik", "🔮 Prediksi Data Baru", "ℹ️ Tentang"
]

# ======================================================================
# SIDEBAR - PENGATURAN GLOBAL
# ======================================================================
with st.sidebar:
    st.markdown("""
    <div class="sb-brand">
        <div class="sb-logo">📊</div>
        <div>
            <div class="sb-name">PKH Analytics</div>
            <div class="sb-sub">Logistic Regression</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.button("🏠 Kembali ke Beranda", type="secondary", width='stretch',
               on_click=_kembali_beranda, key="btn_kembali_beranda")

    st.markdown('<span class="sb-sect">🧭 Menu Analisis</span>', unsafe_allow_html=True)
    menu_terpilih = st.radio(
        "Navigasi menu", MENU_ITEMS,
        label_visibility="collapsed", key="menu_navigasi"
    )

    st.markdown('<span class="sb-sect">📁 Sumber Data</span>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload Data Kemiskinan (.xlsx)", type=["xlsx"])

    if uploaded is not None:
        df_raw = pd.read_excel(uploaded)
        sumber_data = uploaded.name
    elif os.path.exists(DEFAULT_DATA_PATH):
        df_raw = pd.read_excel(DEFAULT_DATA_PATH)
        sumber_data = "DATA_KEMISKINAN_EKSTREM.xlsx (bawaan, di folder yang sama)"
    else:
        df_raw = None
        sumber_data = None

    st.markdown('<span class="sb-sect">⚙️ Parameter Model</span>', unsafe_allow_html=True)
    n_splits = st.slider("Jumlah Fold (K-Fold CV)", min_value=3, max_value=50, value=10, step=1,
                          help="Skrip asli memakai 90 fold. Nilai lebih kecil jauh lebih cepat dan tetap valid secara statistik.")
    random_state = st.number_input("Random State", min_value=0, value=42, step=1)
    proporsi_prediksi_baru = st.slider("Proporsi data untuk simulasi 'Data Baru' (%)", 10, 40, 20, step=5,
                                        help="Sebagian data disisihkan (tidak ikut training) untuk mensimulasikan prediksi pada data baru, seperti df_test di regresi.py.")

    latih_button = st.button("🚀 Latih / Latih Ulang Model", type="primary", width='stretch')

    st.markdown("""
    <div class="sb-info">
        Interface ini mereplikasi pipeline <strong style="color:#38BDF8">regresi.py</strong>:
        preprocessing → encoding → K-Fold CV + SMOTE + Regresi Logistik →
        interpretasi (statsmodels) → evaluasi → prediksi data baru.
    </div>
    """, unsafe_allow_html=True)

# ======================================================================
# HERO
# ======================================================================
st.markdown("""
<div class="hero-wrap">
    <div class="hero-glow-l"></div>
    <div class="hero-glow-r"></div>
    <div class="hero-badge">✦ Statistika Inferensial</div>
    <div class="hero-title">Analisis Regresi Logistik Biner — <span>PKH</span></div>
    <div class="hero-sub">Status Penerima Bantuan PKH pada Data Kemiskinan Ekstrem — preprocessing, encoding,
    K-Fold CV + SMOTE + Regresi Logistik, interpretasi model, evaluasi, hingga prediksi data baru.</div>
    <div class="hero-chips">
        <span class="hero-chip">🧹 Preprocessing</span>
        <span class="hero-chip-arr">→</span>
        <span class="hero-chip">🔢 Encoding</span>
        <span class="hero-chip-arr">→</span>
        <span class="hero-chip">🤖 K-Fold + SMOTE</span>
        <span class="hero-chip-arr">→</span>
        <span class="hero-chip">📈 Interpretasi</span>
        <span class="hero-chip-arr">→</span>
        <span class="hero-chip">🔮 Prediksi</span>
    </div>
</div>
""", unsafe_allow_html=True)

if df_raw is None:
    st.markdown("""
    <div class="empty-wrap">
        <div class="empty-icon">📂</div>
        <div class="empty-title">Belum Ada Data</div>
        <div class="empty-sub">Upload file <code>DATA_KEMISKINAN_EKSTREM.xlsx</code> melalui sidebar kiri untuk memulai analisis.</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

st.success(f"Data dimuat dari: **{sumber_data}** — {df_raw.shape[0]} baris, {df_raw.shape[1]} kolom.")

# Bersihkan data (dijalankan setiap kali data berubah)
df_clean, laporan_bersih = bersihkan_data(df_raw)

# Encoding fitur dijalankan tak bersyarat (bukan hanya saat menu "Preprocessing &
# Encoding" dipilih) supaya menu Training/Interpretasi/Evaluasi/Prediksi tetap bisa
# langsung dipakai dari sidebar, tanpa harus mampir dulu ke menu Preprocessing.
X_full, encoder_dict = encode_fitur(df_clean)
y_full = encode_target(df_clean)
st.session_state["X_full"] = X_full
st.session_state["y_full"] = y_full
st.session_state["encoder_dict"] = encoder_dict
st.session_state["df_clean"] = df_clean

# Simpan ke session_state supaya antar-tab konsisten
st.session_state.setdefault("trained", False)

# ----------------------------------------------------------------------
# MENU 1 — EKSPLORASI DATA
# ----------------------------------------------------------------------
if menu_terpilih == MENU_ITEMS[0]:
    st.markdown('<div class="sec-title">🔍 Ringkasan Data</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Jumlah Baris (mentah)", df_raw.shape[0])
    c2.metric("Jumlah Baris (bersih)", df_clean.shape[0])
    c3.metric("Jumlah Kolom", df_raw.shape[1])

    with st.expander("📋 Catatan pembersihan data", expanded=True):
        for baris in laporan_bersih:
            st.markdown(f"- {baris}")

    st.markdown("#### Filter Data untuk Eksplorasi")
    kolom_filter = st.multiselect("Pilih kolom untuk difilter (opsional)",
                                   options=KOLOM_KATEGORIKAL, default=[])
    df_filtered = df_clean.copy()
    filt_cols = st.columns(max(len(kolom_filter), 1)) if kolom_filter else None
    for i, kolom in enumerate(kolom_filter):
        pilihan = filt_cols[i].multiselect(kolom, options=sorted(df_clean[kolom].unique().tolist()),
                                            default=sorted(df_clean[kolom].unique().tolist()), key=f"filter_{kolom}")
        df_filtered = df_filtered[df_filtered[kolom].isin(pilihan)]

    render_data_table(
        df_filtered, max_rows=20,
        caption=f"Menampilkan {min(20, df_filtered.shape[0])} dari {df_filtered.shape[0]} baris (total sebelum filter: {df_clean.shape[0]} baris)."
    )

    st.markdown("#### Distribusi Variabel Kategorikal")
    kolom_pilih = st.selectbox("Pilih variabel", KOLOM_KATEGORIKAL)
    vc = df_filtered[kolom_pilih].value_counts(dropna=False).reset_index()
    vc.columns = [kolom_pilih, "Frekuensi"]
    fig = px.bar(vc, x=kolom_pilih, y="Frekuensi", text="Frekuensi", color=kolom_pilih)
    fig.update_layout(showlegend=False, xaxis_tickangle=-30)
    style_plotly(fig, f"Distribusi {kolom_pilih}")
    st.plotly_chart(fig, width='stretch')

    st.markdown("#### Proporsi Penerima vs Bukan Penerima PKH")
    vc_target = df_filtered['Penerima PKH'].value_counts().reset_index()
    vc_target.columns = ['Status', 'Jumlah']
    fig_pie = px.pie(vc_target, names='Status', values='Jumlah', hole=0.4,
                      color='Status', color_discrete_map=STATUS_COLOR_MAP)
    style_plotly(fig_pie, "Proporsi Penerima vs Bukan Penerima PKH")
    st.plotly_chart(fig_pie, width='stretch')

# ----------------------------------------------------------------------
# MENU 2 — PREPROCESSING & ENCODING
# ----------------------------------------------------------------------
elif menu_terpilih == MENU_ITEMS[1]:
    st.markdown('<div class="sec-title">🧹 Encoding Fitur</div>', unsafe_allow_html=True)

    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Ordinal Encoding**")
        st.markdown("Pendidikan Kepala Keluarga →")
        render_kv_chips(PENDIDIKAN_MAPPING)
        st.markdown("Daya Listrik Terpasang →")
        render_kv_chips(LISTRIK_MAPPING)
    with colB:
        st.markdown("**Label Encoding (nominal)**")
        st.markdown(", ".join(KOLOM_NOMINAL))

    st.markdown("#### Contoh Data Setelah Encoding")
    df_encoded_preview = X_full.copy()
    df_encoded_preview['Penerima PKH'] = y_full
    render_data_table(df_encoded_preview, max_rows=15)

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df_encoded_preview.to_excel(writer, sheet_name="Hasil_Encoding", index=False)
    st.download_button("⬇️ Download Hasil Encoding (.xlsx)", data=buf.getvalue(),
                        file_name="Hasil_Encoding_Data_PKH.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ----------------------------------------------------------------------
# MENU 3 — TRAINING K-FOLD
# ----------------------------------------------------------------------
elif menu_terpilih == MENU_ITEMS[2]:
    st.markdown(f'<div class="sec-title">🤖 K-Fold Cross Validation (k={n_splits}) + SMOTE + Regresi Logistik</div>', unsafe_allow_html=True)

    if latih_button or st.session_state.get("trained", False):
        if latih_button:
            X_full = st.session_state["X_full"]
            y_full = st.session_state["y_full"]

            # Sisihkan sebagian data sebagai simulasi "data baru" (df_test di regresi.py)
            df_train_split, df_new_sim = train_test_split(
                df_clean, test_size=proporsi_prediksi_baru / 100, random_state=random_state,
                stratify=df_clean['Penerima PKH']
            )
            st.session_state["df_new_sim"] = df_new_sim

            progress_bar = st.progress(0.0, text="Melatih model K-Fold...")

            def progress_cb(fold, total):
                progress_bar.progress(fold / total, text=f"Fold {fold}/{total} selesai")

            hasil = jalankan_kfold(X_full, y_full, n_splits, random_state, progress_cb)
            progress_bar.empty()

            st.session_state["hasil_kfold"] = hasil
            st.session_state["trained"] = True
            st.success(f"Training selesai! Fold terbaik: **Fold {hasil['best_fold']}** "
                       f"dengan akurasi **{hasil['best_accuracy']:.4f}**")

        hasil = st.session_state["hasil_kfold"]
        df_hasil = hasil["df_hasil"]

        c1, c2, c3 = st.columns(3)
        c1.metric("Rata-rata Akurasi", f"{hasil['mean_accuracy']:.4f}")
        c2.metric("Fold Terbaik", hasil["best_fold"])
        c3.metric("Akurasi Fold Terbaik", f"{hasil['best_accuracy']:.4f}")

        fig_line = px.line(df_hasil, x="Fold", y="Akurasi", markers=True,
                            title="Akurasi per Fold")
        fig_line.add_hline(y=hasil['mean_accuracy'], line_dash="dash", line_color="gray",
                            annotation_text="Rata-rata")
        fig_line.add_vline(x=hasil["best_fold"], line_dash="dot", line_color=AMBER,
                            annotation_text="Fold terbaik")
        style_plotly(fig_line, "Akurasi per Fold")
        fig_line.update_traces(line_color=SKY, marker=dict(color=EMRLD, size=8))
        st.plotly_chart(fig_line, width='stretch')

        st.markdown("#### Detail Hasil per Fold")
        st.dataframe(df_hasil.style.format({
            "Akurasi": "{:.4f}", "Presisi": "{:.4f}", "Recall": "{:.4f}", "F1-Score": "{:.4f}"
        }).highlight_max(subset=["Akurasi"], color="rgba(16,185,129,0.35)"), width='stretch')
    else:
        st.info("Klik **🚀 Latih / Latih Ulang Model** di sidebar untuk menjalankan K-Fold Cross Validation.")

# ----------------------------------------------------------------------
# MENU 4 — INTERPRETASI MODEL (statsmodels)
# ----------------------------------------------------------------------
elif menu_terpilih == MENU_ITEMS[3]:
    st.markdown('<div class="sec-title">📈 Pengaruh Variabel terhadap Target (Koefisien & p-value)</div>', unsafe_allow_html=True)
    if not st.session_state.get("trained", False):
        st.info("Latih model terlebih dahulu di tab **Training K-Fold**.")
    else:
        hasil = st.session_state["hasil_kfold"]
        X_full = st.session_state["X_full"]
        y_full = st.session_state["y_full"]

        result, coef_df = fit_statsmodels_logit(
            X_full, y_full, hasil["best_train_idx"], hasil["best_imputer"], hasil["best_scaler"]
        )
        st.session_state["logit_result"] = result
        st.session_state["coef_df"] = coef_df

        st.dataframe(coef_df.style.format({
            "Koefisien": "{:+.4f}", "Std. Error": "{:.4f}", "z-value": "{:+.3f}", "p-value": "{:.5f}"
        }), width='stretch')
        st.caption("Signifikansi: *** p<0.001, ** p<0.01, * p<0.05")

        fig_coef = px.bar(coef_df[coef_df['Variabel'] != 'const'].sort_values('Koefisien'),
                           x='Koefisien', y='Variabel', orientation='h', color='Koefisien',
                           color_continuous_scale=[[0, SKY], [0.5, "#1E2D45"], [1, EMRLD]],
                           title="Koefisien Regresi Logistik (data terstandarisasi)")
        style_plotly(fig_coef)
        fig_coef.update_layout(coloraxis_colorbar=dict(tickfont=dict(color="#94A3B8")))
        st.plotly_chart(fig_coef, width='stretch')

        with st.expander("📄 Ringkasan Lengkap statsmodels (summary)"):
            st.text(str(result.summary()))

# ----------------------------------------------------------------------
# MENU 5 — EVALUASI MODEL TERBAIK
# ----------------------------------------------------------------------
elif menu_terpilih == MENU_ITEMS[4]:
    st.markdown('<div class="sec-title">✅ Evaluasi Model pada Data Uji (Fold Terbaik)</div>', unsafe_allow_html=True)
    if not st.session_state.get("trained", False):
        st.info("Latih model terlebih dahulu di tab **Training K-Fold**.")
    else:
        hasil = st.session_state["hasil_kfold"]
        X_full = st.session_state["X_full"]
        y_full = st.session_state["y_full"]

        X_test_best = X_full.iloc[hasil["best_test_idx"]]
        y_test_best = y_full.iloc[hasil["best_test_idx"]]

        X_test_imp = hasil["best_imputer"].transform(X_test_best)
        X_test_scaled = hasil["best_scaler"].transform(X_test_imp)
        y_pred_best = hasil["best_model"].predict(X_test_scaled)

        acc = accuracy_score(y_test_best, y_pred_best)
        prec = precision_score(y_test_best, y_pred_best, zero_division=0)
        rec = recall_score(y_test_best, y_pred_best, zero_division=0)
        f1 = f1_score(y_test_best, y_pred_best, zero_division=0)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Akurasi", f"{acc:.4f}")
        c2.metric("Presisi", f"{prec:.4f}")
        c3.metric("Recall", f"{rec:.4f}")
        c4.metric("F1-Score", f"{f1:.4f}")

        cm = confusion_matrix(y_test_best, y_pred_best)
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm, x=["Prediksi: Bukan Penerima", "Prediksi: Penerima"],
            y=["Aktual: Bukan Penerima", "Aktual: Penerima"],
            text=cm, texttemplate="%{text}", textfont=dict(color="#F1F5F9", size=16),
            colorscale=[[0, "#0D1526"], [0.5, "#0284C7"], [1, "#10B981"]]
        ))
        style_plotly(fig_cm, "Confusion Matrix")
        st.plotly_chart(fig_cm, width='stretch')

        st.markdown("#### Classification Report")
        report_dict = classification_report(
            y_test_best, y_pred_best, target_names=['Bukan Penerima PKH', 'Penerima PKH'],
            output_dict=True, zero_division=0
        )
        st.dataframe(pd.DataFrame(report_dict).transpose().style.format("{:.4f}"), width='stretch')

# ----------------------------------------------------------------------
# MENU 6 — PREDIKSI DATA BARU
# ----------------------------------------------------------------------
elif menu_terpilih == MENU_ITEMS[5]:
    st.markdown('<div class="sec-title">🔮 Prediksi pada Data Baru</div>', unsafe_allow_html=True)
    if not st.session_state.get("trained", False):
        st.info("Latih model terlebih dahulu di tab **Training K-Fold**.")
    else:
        hasil = st.session_state["hasil_kfold"]
        encoder_dict = st.session_state["encoder_dict"]
        df_clean_ref = st.session_state["df_clean"]

        sumber = st.radio("Sumber data baru",
                           ["Gunakan data simulasi (bagian data yang disisihkan saat training)",
                            "Upload file data baru (.xlsx)"], index=0)

        if sumber.startswith("Upload"):
            file_baru = st.file_uploader("Upload data baru (kolom sama seperti data latih, tanpa/berisi kolom 'Penerima PKH')",
                                          type=["xlsx"], key="upload_baru")
            df_new_raw = pd.read_excel(file_baru) if file_baru is not None else None
        else:
            df_new_raw = st.session_state.get("df_new_sim")

        if df_new_raw is None:
            st.warning("Belum ada data baru yang tersedia.")
        else:
            df_new_clean, _ = bersihkan_data(df_new_raw)
            X_new = encode_data_baru(df_new_clean, encoder_dict, df_clean_ref)

            X_new_imp = hasil["best_imputer"].transform(X_new)
            X_new_scaled = hasil["best_scaler"].transform(X_new_imp)

            y_new_pred = hasil["best_model"].predict(X_new_scaled)
            y_new_proba = hasil["best_model"].predict_proba(X_new_scaled)[:, 1]

            hasil_prediksi = pd.DataFrame({
                "Prediksi": pd.Series(y_new_pred).map({0: "TIDAK", 1: "YA"}),
                "Probabilitas Penerima": y_new_proba
            })
            hasil_lengkap = pd.concat([df_new_clean.reset_index(drop=True), hasil_prediksi], axis=1)

            st.markdown(f"#### Hasil Prediksi ({len(hasil_lengkap)} baris)")
            st.dataframe(hasil_lengkap.head(50), width='stretch')

            buf2 = io.BytesIO()
            with pd.ExcelWriter(buf2, engine="xlsxwriter") as writer:
                hasil_lengkap.to_excel(writer, sheet_name="Hasil_Prediksi", index=False)
            st.download_button("⬇️ Download Hasil Prediksi (.xlsx)", data=buf2.getvalue(),
                                file_name="Hasil_Prediksi_PKH.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

            count_pred = hasil_prediksi['Prediksi'].value_counts().reset_index()
            count_pred.columns = ['Status', 'Jumlah']

            colX, colY = st.columns(2)
            with colX:
                fig_bar = px.bar(count_pred, x='Status', y='Jumlah', color='Status', text='Jumlah',
                                  color_discrete_map=STATUS_COLOR_MAP,
                                  title="Distribusi Prediksi Status Penerima PKH")
                style_plotly(fig_bar)
                fig_bar.update_layout(showlegend=False)
                st.plotly_chart(fig_bar, width='stretch')
            with colY:
                fig_pie2 = px.pie(count_pred, names='Status', values='Jumlah', hole=0.4,
                                   color='Status',
                                   color_discrete_map=STATUS_COLOR_MAP,
                                   title="Persentase Prediksi")
                style_plotly(fig_pie2)
                st.plotly_chart(fig_pie2, width='stretch')

# ----------------------------------------------------------------------
# MENU 7 — TENTANG (identitas & informasi sistem)
# ----------------------------------------------------------------------
elif menu_terpilih == MENU_ITEMS[6]:
    # NOTE: Ganti seluruh placeholder di bawah ini (nama, NIM, kampus, dosen,
    # dsb.) dengan identitas Anda yang sebenarnya.
    st.markdown("""
    <div class="tentang-hero">
        <div class="th-icon">📊</div>
        <div>
            <h1>Tentang Sistem</h1>
            <p>PKH Analytics · Analisis Regresi Logistik Biner untuk Status Penerima<br>
            Bantuan PKH pada Data Kemiskinan Ekstrem</p>
            <span class="tentang-chip">📈 Statistika Inferensial &amp; Machine Learning</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="profile-card">
        <div class="profile-avatar">NN</div>
        <div class="profile-info">
            <h3>Maria Sintia Imakulata Bau</h3>
            <div class="profile-meta">
                <span>🪪 NIM/NPM: 00000000</span>
                <span>📍 Universitas Timor</span>
                <span>✉️ email@contoh.com</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="about-card" style="--a-color:#818CF8">
        <div class="ic">👨‍🏫</div>
        <div style="flex:1;">
            <h4>Dosen Pembimbing &amp; Penguji</h4>
            <div class="dosen-grid">
                <div class="dosen-card">
                    <div class="role">Pembimbing I</div>
                    <div class="name">Nama Dosen Pembimbing I</div>
                    <div class="detail">NIP: -</div>
                </div>
                <div class="dosen-card">
                    <div class="role">Pembimbing II</div>
                    <div class="name">Nama Dosen Pembimbing II</div>
                    <div class="detail">NIP: -</div>
                </div>
                <div class="dosen-card">
                    <div class="role">Penguji</div>
                    <div class="name">Nama Dosen Penguji</div>
                    <div class="detail">NIP: -</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="about-card" style="--a-color:#F59E0B">
        <div class="ic">🎯</div>
        <div>
            <h4>Tujuan Sistem</h4>
            <p>Sistem ini dirancang untuk membantu proses analisis dan prediksi status penerima
            Bantuan Program Keluarga Harapan (PKH) menggunakan model Regresi Logistik Biner.
            Melalui sistem ini, data rumah tangga dapat diproses secara otomatis — mulai dari
            pembersihan data, encoding fitur, pelatihan model dengan K-Fold Cross Validation
            dan SMOTE, hingga interpretasi dan evaluasi performa model — sehingga menghasilkan
            informasi yang mudah dianalisis sebagai dasar pengambilan keputusan.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="about-card" style="--a-color:#10B981">
        <div class="ic">🧮</div>
        <div>
            <h4>Metode Regresi Logistik Biner</h4>
            <p>Metode yang digunakan dalam sistem ini adalah Regresi Logistik Biner, salah satu
            algoritma supervised learning yang populer untuk klasifikasi dua kelas. Proses yang
            dilakukan meliputi: 1) Pembersihan data dari nilai anomali dan missing value,
            2) Encoding fitur kategorikal, 3) Pembagian data dengan K-Fold Cross Validation
            disertai penyeimbangan kelas menggunakan SMOTE, 4) Pelatihan model Regresi Logistik,
            5) Interpretasi koefisien &amp; signifikansi variabel dengan statsmodels, dan
            6) Evaluasi performa model pada data uji sebelum digunakan untuk prediksi data baru.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="about-card" style="--a-color:#0EA5E9">
        <div class="ic">📊</div>
        <div>
            <p style="margin-bottom:0.5rem"><strong style="color:#F1F5F9">Manfaat Sistem</strong></p>
            <h4 style="display:none"></h4>
            <p style="margin-bottom:0.5rem">Hasil analisis dari sistem ini dapat digunakan untuk berbagai keperluan, antara lain:</p>
            <ul>
                <li>Mengidentifikasi karakteristik rumah tangga penerima maupun bukan penerima PKH</li>
                <li>Mengetahui variabel yang paling berpengaruh terhadap status penerimaan bantuan</li>
                <li>Membantu instansi terkait melakukan evaluasi ketepatan sasaran bantuan sosial</li>
                <li>Mendukung perencanaan kebijakan penanggulangan kemiskinan berbasis data</li>
                <li>Memprediksi status penerimaan PKH pada data rumah tangga baru</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="about-card" style="--a-color:#818CF8">
        <div class="ic">🔒</div>
        <div>
            <h4>Privasi &amp; Keamanan Data</h4>
            <p>Seluruh data yang diunggah dan diproses dalam sistem ini digunakan hanya untuk
            kebutuhan penelitian dan analisis akademik. Data hanya diproses pada sesi aplikasi
            yang sedang berjalan dan tidak dikirim ke pihak ketiga mana pun.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-title">Alur Kerja Aplikasi</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="step-flow">
        <span class="step-pill"><b>1.</b> Eksplorasi Data</span>
        <span class="step-pill"><b>2.</b> Preprocessing &amp; Encoding</span>
        <span class="step-pill"><b>3.</b> Training K-Fold</span>
        <span class="step-pill"><b>4.</b> Interpretasi Model</span>
        <span class="step-pill"><b>5.</b> Evaluasi Model Terbaik</span>
        <span class="step-pill"><b>6.</b> Prediksi Data Baru</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="about-card" style="--a-color:#F1F5F9;margin-top:0.4rem">
        <div class="ic">🛠️</div>
        <div>
            <h4>Teknologi</h4>
            <p>
            Backend/pipeline: <strong style="color:#F1F5F9">Python</strong> +
            <strong style="color:#F1F5F9">Pandas</strong> +
            <strong style="color:#F1F5F9">NumPy</strong> +
            <strong style="color:#F1F5F9">scikit-learn</strong> (Logistic Regression, K-Fold CV) +
            <strong style="color:#F1F5F9">imbalanced-learn</strong> (SMOTE) +
            <strong style="color:#F1F5F9">statsmodels</strong> (interpretasi statistik).<br>
            Frontend: <strong style="color:#F1F5F9">Streamlit</strong> +
            <strong style="color:#F1F5F9">Plotly</strong>.<br>
            Ekspor hasil ke file <strong style="color:#F1F5F9">Excel (.xlsx)</strong>.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ======================================================================
# FOOTER
# ======================================================================
st.markdown(
    '<div class="footer">© 2026 <span>PKH Analytics</span> — '
    'Analisis Regresi Logistik Biner · Dibuat dengan ♥ menggunakan Streamlit</div>',
    unsafe_allow_html=True,
)
