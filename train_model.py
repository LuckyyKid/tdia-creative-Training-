import pandas as pd
import json
from sklearn.linear_model import LogisticRegression
import numpy as np

# --- CONFIGURATION PRINCIPALE ---
FILE_PATH = 'Training_Creative.csv' 
LABEL_COLUMN = 'LABEL' 

# Noms des 30 features
# F1 à F28 proviennent du CSV | F29 & F30 sont calculées
FEATURE_COLUMNS = [f'F{i}' for i in range(1, 29)] + ['F29_PainPointMatch', 'F30_IncentiveMatch']

# ==============================================================================
# 1. CONSTANTES ET DICTIONNAIRES BILINGUES (FR/EN)
# ==============================================================================

# Mots-clés des points de douleur/bénéfices (Awareness/Consideration) - Bilingue
PAIN_POINTS_BY_INDUSTRY = {
    "underwear": [
        "inconfort", "irrite", "invisible", "mauvaise qualité", "sensation", "pop underwear", "ride moi",
        "discomfort", "itch", "irritation", "poor quality", "sensation", "invisible", "wedgie" 
    ],
    "fashion": [
        "taille", "coupe", "style", "durabilité", "dernier cri", "tendance", "vieux", "démodé",
        "size", "fit", "style", "durability", "trend", "old", "outdated", "cheap"
    ],
    "beauty": [
        "rides", "imperfections", "sécheresse", "âge", "éclat", "boutons", "acné", "hydratation", "cernes",
        "wrinkles", "blemishes", "dryness", "age", "glow", "acne", "hydration", "dark circles", "pores"
    ],
    "fitness": [
        "fatigue", "douleur", "stagnation", "perte de poids", "motivation", "résultats", "calories", "régime",
        "fatigue", "pain", "stuck", "weight loss", "motivation", "results", "calories", "diet", "workout"
    ],
    "tech": [
        "lent", "bug", "complexe", "sécurité", "obsolète", "mise à jour", "batterie", "wifi", "piraté",
        "slow", "bug", "complex", "security", "outdated", "update", "battery", "wifi", "hacked", "crash"
    ]
}

# Mots-clés d'Incitation à l'Achat (Conversion) - Bilingue
INCENTIVE_KEYWORDS = [
    "réduction", "rabais", "offre", "promo", "soldes", "gratuit", "free", "%", "off", 
    "deal", "code", "coupon", "exclusif", "achetez", "magasinez", "shop", 
    "économiser", "maintenant", "dernier chance",
    "discount", "sale", "offer", "exclusive", "buy now", "save", "limited time", 
    "flash sale", "bogo", "clearance", "ships free", "coupon", "today" 
]

# ==============================================================================
# 2. PRÉPARATION DES DONNÉES ET CALCUL DES FEATURES AUTO-AJUSTABLES (F29/F30)
# ==============================================================================
print("Démarrage du processus d'entraînement...")

try:
    # Lecture TENTATIVE : Utilisation du séparateur et encodage standard
    df = pd.read_csv(FILE_PATH, sep=',', encoding='utf-8') 
except Exception:
    # Tente Latin-1 en cas d'échec
    df = pd.read_csv(FILE_PATH, sep=',', encoding='latin-1')
    
# Nettoyage des noms de colonnes : ESSENTIEL pour trouver F1, F2, industry, GOAL, etc.
df.columns = df.columns.str.strip() 

# -------------------------------------------------------------
# LOGIQUE POUR CALCULER F29 ET F30 (LISANT LES STRINGS DU CSV)
# -------------------------------------------------------------
def calculate_auto_features(row):
    """Calcule F29 (PainPointMatch) et F30 (IncentiveMatch) par ligne."""
    try:
        # LECTURE DES COLONNES STRING DU CSV
        vision = json.loads(row['vision_json']) # Contient 'visible_words'
        industry = row['industry'].lower().strip() # Contient 'underwear', 'beauty', etc.
        
        visible_words = [w.lower() for w in vision["text_elements"]["visible_words"]]
        visible_words_joined = " ".join(visible_words)
    except:
        # Retourne 0, 0 si le JSON est invalide ou des colonnes sont manquantes
        return 0, 0 
        
    # F29: Pain Point Match
    pain_point_match = 0
    if industry in PAIN_POINTS_BY_INDUSTRY:
        keywords = PAIN_POINTS_BY_INDUSTRY[industry]
        if any(k in visible_words_joined for k in keywords):
            pain_point_match = 1
            
    # F30: Incentive Match
    incentive_match = int(any(k in visible_words_joined for k in INCENTIVE_KEYWORDS))
    
    return pain_point_match, incentive_match

print("Calcul des features F29 (PainPoint) et F30 (Incentive) à partir du CSV...")
# Applique la fonction à chaque ligne et crée les nouvelles colonnes
df[['F29_PainPointMatch', 'F30_IncentiveMatch']] = df.apply(
    lambda row: calculate_auto_features(row), axis=1, result_type='expand'
)

# Conversion des labels en 1s et 0s 
df[LABEL_COLUMN] = df[LABEL_COLUMN].str.lower().map({'positive': 1, 'negative': 0})

# Remplacer les valeurs manquantes (NaN) par 0 pour le calcul final
df = df.fillna(0) 

# Définition des variables d'entraînement (30 colonnes: F1-F28 + F29/F30)
X = df[FEATURE_COLUMNS] 
Y = df[LABEL_COLUMN].astype(int) 

# --- 3. ENTRAÎNEMENT DU MODÈLE ---
print("\nDémarrage de l'entraînement de la Régression Logistique (30 features)...")

model = LogisticRegression(solver='liblinear', random_state=42, max_iter=1000)
model.fit(X, Y)

# --- 4. RÉSULTATS : LES POIDS DU CERVEAU ---
print("\n" + "="*50)
print("✅ ENTRAÎNEMENT TERMINÉ ET POIDS CALCULÉS")
print("="*50)

# Récupération des Poids et du Biais
weights_array = model.coef_[0]
bias_value = model.intercept_[0]

# --- 5. PRÉPARATION DE L'OUTPUT POUR L'API (À COPIER) ---

api_weights = {}
for i, feature_name in enumerate(FEATURE_COLUMNS):
    # Assurez-vous que l'indexation est correcte F1 -> F29/F30
    api_weights[feature_name] = round(weights_array[i], 4)

print("\n--- COPIEZ LES LIGNES SUIVANTES DANS VOTRE api_scorer.py ---\n")

print("POIDS (API_WEIGHTS) :")
print("API_WEIGHTS = {")
for name, weight in api_weights.items():
    print(f"    '{name}': {weight},")
print("}")

print("\nBIAIS (API_BIAS) :")
print(f"API_BIAS = {round(bias_value, 4)}")
print("\n" + "="*50)
print("Vous pouvez maintenant mettre à jour votre API et déployer sur Render.")
print("="*50)