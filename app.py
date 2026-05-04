import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import json

# ── Configuration page ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Donation Predictor",
    page_icon="🎁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Sora', sans-serif; }
    .main { background-color: #f8fafc; }
    .header-box {
        background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%);
        padding: 2rem 2.5rem; border-radius: 16px;
        margin-bottom: 2rem; color: white;
    }
    .header-box h1 { font-size: 2rem; font-weight: 700; margin: 0; }
    .header-box p  { font-size: 1rem; opacity: 0.85; margin: 0.4rem 0 0 0; }
    .metric-card {
        background: white; border-radius: 12px;
        padding: 1.4rem 1.6rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.07);
        border-left: 5px solid #2563eb; margin-bottom: 1rem;
    }
    .metric-card h3 { font-size: 0.85rem; color: #64748b; margin: 0;
                      text-transform: uppercase; letter-spacing: 0.08em; }
    .metric-card h2 { font-size: 2rem; font-weight: 700;
                      margin: 0.3rem 0 0 0; color: #1e3a5f; }
    .result-yes {
        background: linear-gradient(135deg, #d1fae5, #a7f3d0);
        border-left: 5px solid #10b981; border-radius: 12px;
        padding: 1.5rem; text-align: center; margin-bottom: 1rem;
    }
    .result-no {
        background: linear-gradient(135deg, #fee2e2, #fecaca);
        border-left: 5px solid #ef4444; border-radius: 12px;
        padding: 1.5rem; text-align: center; margin-bottom: 1rem;
    }
    .result-yes h2, .result-no h2 { font-size: 1.4rem; margin: 0; }
    .section-title {
        font-size: 1.1rem; font-weight: 600; color: #1e3a5f;
        margin-bottom: 0.8rem; padding-bottom: 0.4rem;
        border-bottom: 2px solid #e2e8f0;
    }
    .info-badge {
        display: inline-block; background: #dbeafe; color: #1e40af;
        border-radius: 20px; padding: 0.2rem 0.8rem;
        font-size: 0.8rem; font-weight: 600; margin-bottom: 0.5rem;
    }
    .stButton > button {
        background: linear-gradient(135deg, #1e3a5f, #2563eb);
        color: white; border: none; border-radius: 10px;
        padding: 0.7rem 2rem; font-size: 1rem;
        font-weight: 600; width: 100%;
    }
    .stButton > button:hover { opacity: 0.9; }
</style>
""", unsafe_allow_html=True)

# ── Chargement des modèles ────────────────────────────────────────────────
@st.cache_resource
def load_models():
    m = {}
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers

        m['rf']            = joblib.load("models/random_forest.pkl")
        m['xgb']           = joblib.load("models/xgboost.pkl")
        m['scaler']        = joblib.load("models/scaler.pkl")
        m['pca']           = joblib.load("models/pca.pkl")
        m['target_scaler'] = joblib.load("models/target_scaler.pkl")
        m['medians']       = joblib.load("models/medians.pkl")

        n_components = m['pca'].n_components_

        # Reconstruire ANN Classification depuis config + poids
        with open("models/ann_clf_config.json") as f:
            clf_config = json.load(f)
        ann_clf = keras.Sequential.from_config(clf_config)
        ann_clf.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        ann_clf.load_weights("models/ann_clf_weights.weights.h5")
        m['ann_clf'] = ann_clf

        # Reconstruire ANN Regression depuis config + poids
        with open("models/ann_reg_config.json") as f:
            reg_config = json.load(f)
        ann_reg = keras.Sequential.from_config(reg_config)
        ann_reg.compile(optimizer='adam', loss='mse', metrics=['mae'])
        ann_reg.load_weights("models/ann_reg_weights.weights.h5")
        m['ann_reg'] = ann_reg

        m['loaded'] = True
    except Exception as e:
        m['loaded'] = False
        m['error']  = str(e)
    return m

models = load_models()

FEATURES = [
    'user__xp', 'user__streak', 'user__login_freq_last_30d',
    'user__avg_session_duration_min', 'user__total_sessions', 'user__quiz_avg_score',
    'don__total_donations', 'don__completed_donations', 'don__total_items_donated',
    'don__total_merci_points', 'don__avg_quantity_per_donation', 'don__anonymous_ratio',
    'don__days_since_last_donation', 'don__has_badge',
    'don__num_favorites_given', 'don__num_comments_given',
    'don__gamification_points',
    'sub__course_progress_avg', 'forum__forum_engagement_score'
]

# ── Header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-box">
    <h1>🎁 Donation Predictor</h1>
    <p>Optimiser la gestion des donations et fidéliser les donateurs — Machine Learning Appliqué</p>
</div>
""", unsafe_allow_html=True)

if not models.get('loaded'):
    st.error(f"⚠️ Erreur de chargement des modèles : {models.get('error', '')}")
    st.info("Assure-toi que le dossier `models/` contient tous les fichiers nécessaires.")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 👤 Profil Donateur")
    st.markdown("---")

    st.markdown('<div class="section-title">📊 Activité Utilisateur</div>', unsafe_allow_html=True)
    xp             = st.slider("XP utilisateur",             0, 10000, 500,  step=50)
    streak         = st.slider("Streak (jours consécutifs)", 0, 365,   30)
    login_freq     = st.slider("Connexions / 30 jours",      0, 30,    10)
    session_dur    = st.slider("Durée moy. session (min)",   0, 120,   25)
    total_sessions = st.slider("Total sessions",             0, 500,   80)
    quiz_score     = st.slider("Score quiz moyen (%)",       0, 100,   60)

    st.markdown('<div class="section-title">🎁 Historique Donations</div>', unsafe_allow_html=True)
    total_don       = st.slider("Total donations",           0, 100,   5)
    completed_don   = st.slider("Donations complétées",      0, 100,   4)
    total_items     = st.slider("Total items donnés",        0, 200,   10)
    merci_points    = st.slider("Points Merci",              0, 5000,  300, step=50)
    avg_qty         = st.slider("Quantité moy. par don",     0, 20,    3)
    anonymous_ratio = st.slider("Ratio anonyme (0-1)",       0.0, 1.0, 0.2, step=0.05)
    days_since      = st.slider("Jours depuis dernier don",  0, 365,   30)
    has_badge       = st.selectbox("A un badge ?", [0, 1],
                                   format_func=lambda x: "Oui ✅" if x else "Non ❌")
    favorites       = st.slider("Favoris donnés",            0, 100,   5)
    comments        = st.slider("Commentaires donnés",       0, 100,   8)
    gamif_points    = st.slider("Points gamification",       0, 10000, 500, step=50)

    st.markdown('<div class="section-title">📚 Cours & Forum</div>', unsafe_allow_html=True)
    course_progress = st.slider("Progression cours moy. (%)", 0, 100, 55)
    forum_score     = st.slider("Score engagement forum",     0, 100,  20)

    predict_btn = st.button("🔮 Prédire")

# ── Données input ─────────────────────────────────────────────────────────
input_data = pd.DataFrame([[
    xp, streak, login_freq, session_dur, total_sessions, quiz_score,
    total_don, completed_don, total_items, merci_points, avg_qty,
    anonymous_ratio, days_since, has_badge, favorites, comments,
    gamif_points, course_progress, forum_score
]], columns=FEATURES)

# ── Fonction de prédiction ────────────────────────────────────────────────
def predict(input_df):
    X = input_df.copy()
    for col in FEATURES:
        X[col] = pd.to_numeric(X[col], errors='coerce')
        if X[col].isnull().any():
            X[col] = X[col].fillna(models['medians'][col])

    X_scaled = models['scaler'].transform(X[FEATURES].values)
    X_pca    = models['pca'].transform(X_scaled)

    xgb_proba = float(models['xgb'].predict_proba(X_pca)[0][1])
    xgb_pred  = int(xgb_proba >= 0.5)

    rf_norm  = float(np.clip(models['rf'].predict(X_pca)[0], 0, 1))
    rf_items = float(models['target_scaler'].inverse_transform([[rf_norm]])[0][0])

    ann_proba = float(models['ann_clf'].predict(X_pca, verbose=0)[0][0])
    ann_pred  = int(ann_proba >= 0.5)

    ann_norm  = float(np.clip(models['ann_reg'].predict(X_pca, verbose=0)[0][0], 0, 1))
    ann_items = float(models['target_scaler'].inverse_transform([[ann_norm]])[0][0])

    return {
        'xgb_proba': xgb_proba, 'xgb_pred':  xgb_pred,
        'rf_items':  max(0, rf_items),
        'ann_proba': ann_proba,  'ann_pred':  ann_pred,
        'ann_items': max(0, ann_items)
    }

# ── Résultats ─────────────────────────────────────────────────────────────
if predict_btn:
    with st.spinner("Prédiction en cours..."):
        res = predict(input_data)

    st.markdown("## 📊 Résultats de la Prédiction")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎯 Donnera-t-il dans 30 jours ?")
        st.markdown('<span class="info-badge">Classification</span>', unsafe_allow_html=True)

        xgb_label = "✅ OUI — Donnera" if res['xgb_pred'] == 1 else "❌ NON — Ne donnera pas"
        xgb_class = "result-yes" if res['xgb_pred'] == 1 else "result-no"
        st.markdown(f"""
        <div class="{xgb_class}">
            <h3 style="margin:0;font-size:0.85rem;opacity:0.7">XGBoost</h3>
            <h2>{xgb_label}</h2>
            <p style="margin:0.3rem 0 0 0;font-size:0.9rem">
                Probabilité : <b>{res['xgb_proba']*100:.1f}%</b></p>
        </div>""", unsafe_allow_html=True)

        ann_label = "✅ OUI — Donnera" if res['ann_pred'] == 1 else "❌ NON — Ne donnera pas"
        ann_class = "result-yes" if res['ann_pred'] == 1 else "result-no"
        st.markdown(f"""
        <div class="{ann_class}">
            <h3 style="margin:0;font-size:0.85rem;opacity:0.7">ANN (Réseau de Neurones)</h3>
            <h2>{ann_label}</h2>
            <p style="margin:0.3rem 0 0 0;font-size:0.9rem">
                Probabilité : <b>{res['ann_proba']*100:.1f}%</b></p>
        </div>""", unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.barh(['XGBoost', 'ANN'],
                [res['xgb_proba']*100, res['ann_proba']*100],
                color=['#2563eb', '#f59e0b'], height=0.4)
        ax.axvline(x=50, color='red', linestyle='--', linewidth=1.5, label='Seuil 50%')
        for i, val in enumerate([res['xgb_proba'], res['ann_proba']]):
            ax.text(val*100 + 1, i, f'{val*100:.1f}%', va='center', fontweight='bold')
        ax.set_xlim(0, 115)
        ax.set_xlabel("Probabilité (%)")
        ax.set_title("Probabilité de don dans 30 jours", fontweight='bold')
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("### 📦 Combien d'items va-t-il donner ?")
        st.markdown('<span class="info-badge">Régression</span>', unsafe_allow_html=True)

        rf_items  = round(res['rf_items'], 1)
        ann_items = round(res['ann_items'], 1)

        st.markdown(f"""
        <div class="metric-card">
            <h3>Random Forest</h3>
            <h2>{rf_items} items</h2>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-card" style="border-left-color:#f59e0b">
            <h3>ANN (Réseau de Neurones)</h3>
            <h2>{ann_items} items</h2>
        </div>""", unsafe_allow_html=True)

        fig2, ax2 = plt.subplots(figsize=(6, 3))
        bars = ax2.bar(['Random Forest', 'ANN'], [rf_items, ann_items],
                       color=['#2563eb', '#f59e0b'], width=0.4, edgecolor='white')
        for bar, val in zip(bars, [rf_items, ann_items]):
            ax2.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.05, str(val),
                     ha='center', fontweight='bold', fontsize=12)
        ax2.set_ylabel("Nombre d'items prédits")
        ax2.set_title("Items prédits — Comparaison modèles", fontweight='bold')
        ax2.set_ylim(0, max(rf_items, ann_items) * 1.5 + 1)
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close()

    st.markdown("---")
    st.markdown("### 📋 Récapitulatif")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("XGBoost — Probabilité", f"{res['xgb_proba']*100:.1f}%")
    c2.metric("ANN — Probabilité",     f"{res['ann_proba']*100:.1f}%")
    c3.metric("Random Forest — Items", f"{rf_items}")
    c4.metric("ANN — Items",           f"{ann_items}")

    st.markdown("---")
    avg_proba = (res['xgb_proba'] + res['ann_proba']) / 2
    avg_items = (rf_items + ann_items) / 2

    if avg_proba >= 0.6:
        st.success(f"✅ **Donateur à fort potentiel** — Probabilité : **{avg_proba*100:.1f}%** | Volume : **{avg_items:.1f} items** → Envoyer une récompense gamification.")
    elif avg_proba >= 0.4:
        st.warning(f"⚠️ **Donateur incertain** — Probabilité : **{avg_proba*100:.1f}%** | Volume : **{avg_items:.1f} items** → Envoyer un rappel ou incitation.")
    else:
        st.error(f"❌ **Donateur inactif** — Probabilité : **{avg_proba*100:.1f}%** → Lancer une campagne de réengagement.")

else:
    st.markdown("### 👈 Remplis le profil dans la barre latérale puis clique sur **Prédire**")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class="metric-card">
            <h3>🌲 Random Forest</h3><h2>Régression</h2>
            <p style="color:#64748b;font-size:0.9rem;margin-top:0.5rem">
            Prédit le nombre d'items que le donateur va donner</p>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="metric-card" style="border-left-color:#f59e0b">
            <h3>⚡ XGBoost</h3><h2>Classification</h2>
            <p style="color:#64748b;font-size:0.9rem;margin-top:0.5rem">
            Prédit si le donateur va donner dans les 30 prochains jours</p>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="metric-card" style="border-left-color:#10b981">
            <h3>🧠 ANN</h3><h2>Régression + Classification</h2>
            <p style="color:#64748b;font-size:0.9rem;margin-top:0.5rem">
            Réseau de neurones pour les deux objectifs</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#dbeafe;border-radius:12px;padding:1.2rem 1.5rem;margin-top:1rem">
        <b>📌 Business Objective :</b> Optimiser la gestion des donations et fidéliser les donateurs<br>
        <b>📌 DSO :</b> Analyser les comportements de don pour prédire les donations futures et identifier les donateurs à fort potentiel<br>
        <b>📌 PCA :</b> 19 features → 15 composantes (90% variance conservée)
    </div>""", unsafe_allow_html=True)
