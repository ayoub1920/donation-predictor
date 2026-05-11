from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import json
import os

app = Flask(__name__)
CORS(app)  # Autorise les requêtes depuis React Native et Chrome Extension

# ── Chargement des modèles ────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

rf_model      = joblib.load(os.path.join(MODELS_DIR, "random_forest.pkl"))
xgb_model     = joblib.load(os.path.join(MODELS_DIR, "xgboost.pkl"))
scaler        = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
pca           = joblib.load(os.path.join(MODELS_DIR, "pca.pkl"))
target_scaler = joblib.load(os.path.join(MODELS_DIR, "target_scaler.pkl"))
medians       = joblib.load(os.path.join(MODELS_DIR, "medians.pkl"))

try:
    import tensorflow as tf
    from tensorflow import keras
    with open(os.path.join(MODELS_DIR, "ann_clf_config.json")) as f:
        ann_clf = keras.Sequential.from_config(json.load(f))
    ann_clf.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    ann_clf.load_weights(os.path.join(MODELS_DIR, "ann_clf_weights.weights.h5"))

    with open(os.path.join(MODELS_DIR, "ann_reg_config.json")) as f:
        ann_reg = keras.Sequential.from_config(json.load(f))
    ann_reg.compile(optimizer="adam", loss="mse", metrics=["mae"])
    ann_reg.load_weights(os.path.join(MODELS_DIR, "ann_reg_weights.weights.h5"))
    ANN_LOADED = True
except Exception as e:
    print(f"ANN non chargé : {e}")
    ANN_LOADED = False

FEATURES = [
    "user__xp", "user__streak", "user__login_freq_last_30d",
    "user__avg_session_duration_min", "user__total_sessions", "user__quiz_avg_score",
    "don__total_donations", "don__completed_donations", "don__total_items_donated",
    "don__total_merci_points", "don__avg_quantity_per_donation", "don__anonymous_ratio",
    "don__days_since_last_donation", "don__has_badge",
    "don__num_favorites_given", "don__num_comments_given",
    "don__gamification_points",
    "sub__course_progress_avg", "forum__forum_engagement_score"
]

def preprocess(data: dict):
    X = pd.DataFrame([data], columns=FEATURES)
    for col in FEATURES:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        if X[col].isnull().any():
            X[col] = X[col].fillna(medians[col])
    X_scaled = scaler.transform(X[FEATURES].values)
    X_pca    = pca.transform(X_scaled)
    return X_pca

# ── Routes ────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "Donation Predictor API", "models": ["RF","XGBoost","ANN"]})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Données JSON manquantes"}), 400

        X_pca = preprocess(data)

        # XGBoost Classification
        xgb_proba = float(xgb_model.predict_proba(X_pca)[0][1])
        xgb_pred  = int(xgb_proba >= 0.5)

        # Random Forest Regression
        rf_norm  = float(np.clip(rf_model.predict(X_pca)[0], 0, 1))
        rf_items = float(target_scaler.inverse_transform([[rf_norm]])[0][0])
        rf_items = max(0, round(rf_items, 1))

        # ANN
        ann_proba = ann_items = None
        if ANN_LOADED:
            ann_proba = float(ann_clf.predict(X_pca, verbose=0)[0][0])
            ann_norm  = float(np.clip(ann_reg.predict(X_pca, verbose=0)[0][0], 0, 1))
            ann_items = float(target_scaler.inverse_transform([[ann_norm]])[0][0])
            ann_items = max(0, round(ann_items, 1))

        # Recommandation
        avg_proba = (xgb_proba + (ann_proba or xgb_proba)) / 2
        if avg_proba >= 0.6:
            recommendation = "Fort potentiel"
            action = "Envoyer une récompense gamification ou message de remerciement"
            level = "high"
        elif avg_proba >= 0.4:
            recommendation = "Donateur incertain"
            action = "Envoyer un rappel ou une incitation"
            level = "medium"
        else:
            recommendation = "Donateur inactif"
            action = "Lancer une campagne de réengagement"
            level = "low"

        return jsonify({
            "success": True,
            "classification": {
                "xgboost": { "probability": round(xgb_proba, 4), "prediction": xgb_pred, "label": "Donnera" if xgb_pred == 1 else "Ne donnera pas" },
                "ann": { "probability": round(ann_proba, 4) if ann_proba else None, "prediction": int(ann_proba >= 0.5) if ann_proba else None }
            },
            "regression": {
                "random_forest": { "items": rf_items },
                "ann": { "items": ann_items }
            },
            "recommendation": {
                "label": recommendation,
                "action": action,
                "level": level,
                "avg_probability": round(avg_proba * 100, 1)
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/schema", methods=["GET"])
def schema():
    return jsonify({
        "features": FEATURES,
        "description": {
            "user__xp": "Points XP utilisateur (0-10000)",
            "user__streak": "Streak jours consécutifs (0-365)",
            "user__login_freq_last_30d": "Connexions / 30 jours (0-30)",
            "user__avg_session_duration_min": "Durée moy. session en min (0-120)",
            "user__total_sessions": "Total sessions (0-500)",
            "user__quiz_avg_score": "Score quiz moyen % (0-100)",
            "don__total_donations": "Total donations (0-100)",
            "don__completed_donations": "Donations complétées (0-100)",
            "don__total_items_donated": "Total items donnés (0-200)",
            "don__total_merci_points": "Points Merci (0-5000)",
            "don__avg_quantity_per_donation": "Quantité moy. par don (0-20)",
            "don__anonymous_ratio": "Ratio anonyme 0-1",
            "don__days_since_last_donation": "Jours depuis dernier don (0-365)",
            "don__has_badge": "A un badge (0 ou 1)",
            "don__num_favorites_given": "Favoris donnés (0-100)",
            "don__num_comments_given": "Commentaires donnés (0-100)",
            "don__gamification_points": "Points gamification (0-10000)",
            "sub__course_progress_avg": "Progression cours % (0-100)",
            "forum__forum_engagement_score": "Score engagement forum (0-100)"
        }
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
