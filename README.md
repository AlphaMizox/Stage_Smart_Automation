# Détection d'Intrusions dans les Réseaux Industriels (ICS/SCADA)

## 📌 Aperçu du Projet

Ce projet vise à détecter les cyberattaques dans les réseaux industriels (ICS/SCADA) en utilisant des techniques de **Machine Learning (ML)** et de **Deep Learning (DL)**. 

L'objectif principal est de développer un système de détection d'intrusions (IDS) capable de :
- Identifier les attaques sur des protocoles industriels comme **Modbus TCP** et **S7comm(+)**.
- Fonctionner en mode **supervisé** (classification) et **non-supervisé** (détection d'anomalies).
- Se généraliser à des environnements industriels variés (évaluation cross-dataset).

---

## 🏗️ Architecture du Pipeline

Le projet est organisé en **3 notebooks** principaux :

| Notebook | Description | Scripts Clés |
|----------|-------------|--------------|
| `1-pre_processing.ipynb` | Prétraitement des données (nettoyage, normalisation, séquençage) | `StandardScaler`, `MinMaxScaler`, Feature Engineering |
| `2-model_trained.ipynb` | Entraînement et évaluation de 3 modèles complémentaires | `IsolationForest`, `XGBoost`, `LSTM-Autoencoder` |
| `3-cross_data_general.ipynb` | Évaluation cross-dataset et analyse de généralisation | `cross_dataset_evaluation.csv`, Heatmaps |

---

## 🤖 Modèles Implémentés

| Modèle | Type | F1-Score | Description |
|--------|------|----------|-------------|
| **Isolation Forest** | Non-Supervisé | **0.992** | Détection d'anomalies par isolation aléatoire |
| **XGBoost** | Supervisé (Multi-Class) | **0.907** | Classification par gradient boosting |
| **XGBoost (Latent)** | Supervisé (Latent) | **0.536** | Classification sur espace latent LSTM |
| **LSTM-Autoencoder** | Non-Supervisé (DL) | **0.165** | Détection d'anomalies par reconstruction |

> **Note :** L'Isolation Forest surpasse les autres modèles, démontrant l'efficacité des approches non-supervisées pour ce type de données tabulaires.




## 🧠 Analyse de l'Échec du LSTM-Autoencoder

Le LSTM-Autoencoder (F1 = 0.165) a montré des performances décevantes pour les raisons suivantes :

- **Inadéquation aux données tabulaires** : Les séquences de 10 paquets sont trop courtes et bruitées pour que le LSTM extraie des motifs temporels pertinents.
- **Déséquilibre des tâches** : Le modèle hybride (reconstruction + classification) optimise principalement la classification, négligeant la reconstruction.
- **Ressources insuffisantes** : les datasets ne sont pas prets et leurs generation prend beaucoup de temps, ainsi la puissance de calcul est faible pour poursuivre les améliorations.
- **Convergence défaillante** : La perte de reconstruction stagne à ~128 dès les premières époques, en raison de l'activation `Tanh` incompatible avec la distribution des données.



## 💡 Perspectives d'Expérience

* **Surface d'Attaque OT** : Maîtrise des vecteurs industriels (ARP Spoofing, Replay, DoS, MITM, forçage Modbus/S7) et de leur impact physique sur une ligne de production.

* **Architecture IDS Découplée** : Ingestion télémétrique basse latence (< 1 s) via Scapy, Redis Pub/Sub et Node-RED.

* **ML Opérationnel** : Du PCAP aux modèles (Isolation Forest, XGBoost) — et constat des limites du Deep Learning (LSTM-AE : F1 = 0.165).

* **AIOps/LLM** : Contextualisation et corrélation automatique des alertes via Ollama/Teleram pour les équipes SOC.

* **Défense en Profondeur** : Complémentarité réseau + intégrité PLC + surveillance IA pour les infrastructures critiques.
