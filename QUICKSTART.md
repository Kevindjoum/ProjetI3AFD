# 🚀 Guide de Démarrage Rapide — RadioAI Pro

## ⏱️ 5 minutes pour démarrer

### Étape 1️⃣ : Installer les dépendances (2 min)

```bash
pip install -r requirements.txt
```

### Étape 2️⃣ : Démarrer Ollama (2 min)

**Terminal 1 — Serveur Ollama :**
```bash
ollama serve
```

**Terminal 2 — Télécharger les modèles :**
```bash
python setup_ollama.py --pull-models
```

Ou manuellement :
```bash
ollama pull biomistral
ollama pull mistral
```

### Étape 3️⃣ : Tester le système (1 min)

```bash
# Mode mock (sans GPU/Ollama requis)
python test_pipeline.py

# Mode réel (avec Ollama)
python test_pipeline.py --real --samples 3
```

✓ Vous devriez voir une sortie verte : `✓ Test complet avec succès !`

---

## 📖 Ensuite : Exécution complète

### Option A : Via Notebook (Recommandé pour exploration)

```bash
jupyter notebook "RadioAI Pro.ipynb"
```

Sections disponibles :
- 📥 Chargement des données
- 🔍 Exploration et EDA
- 🔄 Pipeline sur exemples
- 📊 Évaluation ROUGE/BERTScore
- 🎯 Ablation study

### Option B : Via CLI

```bash
# Évaluation sur 10 rapports
python run_experiment.py --eval --samples 10

# Ablation study (mono vs multi-agents)
python run_experiment.py --ablation --samples 5

# Vérifier la configuration
python run_experiment.py --setup
```

---

## 🎯 Résultats attendus

Après un test réussi, vous devriez voir :

```
✓ Langue détectée : en
✓ Qualité : good
✓ Anomalies trouvées : 1-3
✓ Fidélité clinique : 0.75-0.95
✓ État dégradé : False

Synthèse médicale :
  Mild cardiomegaly without acute cardiopulmonary disease...

Vulgarisation patient :
  Votre coeur est légèrement agrandi mais vos poumons sont...
```

---

## ⚠️ Troubleshooting

### ❌ "Ollama not reachable"
```bash
# Assurez-vous qu'Ollama tourne
ollama serve

# Vérifier la connexion
curl http://localhost:11434/api/tags
```

### ❌ "File not found: data/rapports_clean.csv"
```bash
# Charger les données via notebook (Section 2)
# Ou télécharger manuellement :
python loader.py  # Voir la classe OpenILoader
```

### ❌ "cuda out of memory"
```bash
# Utiliser Ollama au lieu d'HuggingFace
# Ou réduire max_tokens aux agents
```

---

## 📊 Architecture en 30 secondes

```
Rapport brut
    ↓
PreprocessorAgent  (détection langue, qualité)
    ↓
ExtractorAgent     (NER médical)
    ↓
StructurerAgent    (JSON FHIR)
    ↓
VerifierAgent      (fidélité clinique)
    ↓
MedicalSummaryAgent → PatientSummaryAgent
    ↓
Synthèses + scores
```

6 agents = meilleur qualité que 1 LLM monolithe ✓

---

## 📚 Fichiers clés

| Fichier | Rôle |
|---------|------|
| `agents.py` | 6 agents du pipeline |
| `pipeline.py` | Orchestration LangGraph |
| `model_manager.py` | Gestion Ollama + HF |
| `loader.py` | Chargement données |
| `metrics.py` | ROUGE, BERTScore, fidélité |
| `test_pipeline.py` | Test rapide |
| `run_experiment.py` | Exécution complète |
| `README.md` | Documentation détaillée |

---

## 🔗 Ressources

- **README.md** — Documentation complète
- **RadioAI Pro.ipynb** — Notebook interactif
- **config.json** — Configuration
- **setup_ollama.py** — Setup Ollama

---

## 💡 Prochaines étapes après le test

1. **Explorer les résultats** : `python run_experiment.py --eval --samples 10`
2. **Lancer l'ablation** : `python run_experiment.py --ablation --samples 10`
3. **Analyser** : Ouvrir `data/exports/*.json` pour voir les métriques
4. **Customiser** : Modifier les prompts dans `agents.py` pour votre use case

---

**Questions ?** Consultez `README.md` pour la documentation complète.

Happy radiological analysis! 🩺✨
