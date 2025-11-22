import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np

# --- CONFIGURATION (À MODIFIER SI BESOIN) ---
FILE_PATH = 'Training_Creative.csv' # Vérifiez le nom du fichier
# Liste des noms de vos colonnes de Features
FEATURE_COLUMNS = [f'features{i}' for i in range(1, 29)] 
LABEL_COLUMN = 'benchmark_label' # Votre colonne cible (H)

# --- 1. LECTURE ET PRÉPARATION ---
try:
    df = pd.read_excel(FILE_PATH)
except FileNotFoundError:
    print(f"Erreur : Le fichier {FILE_PATH} n'a pas été trouvé.")
    exit()

# Conversion des labels en 1s et 0s (si ce n'est pas déjà fait)
df[LABEL_COLUMN] = df[LABEL_COLUMN].str.lower().map({'positive': 1, 'negative': 0})

# Remplacer les valeurs manquantes (NaN) par 0 pour le calcul
df = df.fillna(0) 

# Définition des variables d'entraînement
X = df[FEATURE_COLUMNS] # Les Features (F1, F2, ...)
Y = df[LABEL_COLUMN].astype(int) # Le Label (1 ou 0)

# --- 2. ENTRAÎNEMENT DU MODÈLE ---
print("\nDémarrage de l'entraînement de la Régression Logistique...")

# Création et entraînement du modèle
model = LogisticRegression(solver='liblinear', random_state=42, max_iter=1000)
model.fit(X, Y)

# --- 3. RÉSULTATS : LES POIDS DU CERVEAU ---
print("\n Entraînement terminé.")
print(f"Précision du Modèle (Accuracy) sur le dataset: {model.score(X, Y):.2f}")

# Récupération des Poids et du Biais
weights_array = model.coef_[0]
bias_value = model.intercept_[0]

# --- 4. PRÉPARATION DE L'OUTPUT POUR L'API (À COPIER) ---
print("\n--- COPIEZ LES LIGNES SUIVANTES DANS VOTRE API DE SCORING ---")

api_weights = {}
for i, feature_name in enumerate(FEATURE_COLUMNS):
    # Stocker les poids arrondis
    api_weights[feature_name] = round(weights_array[i], 4)

print("\nPOIDS (Weights) :")
print(api_weights)
print("\nBIAIS (Bias/Intercept) :")
print(f"bias = {round(bias_value, 4)}")