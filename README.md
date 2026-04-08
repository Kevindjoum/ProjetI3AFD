# RadioAI Pro — Système Agentique d'Analyse de Rapports Radiologiques

## 📋 Vue d'ensemble

**RadioAI Pro** est un système multi-agents basé sur LangGraph et LLMs open-source pour :
- 📖 **Analyser** automatiquement des rapports radiologiques texte brut
- 🔍 **Extraire** les entités médicales (anomalies, mesures, localisation)
- 📊 **Structurer** les données en représentation JSON FHIR-compatible
- ✅ **Vérifier** la fidélité clinique vs rapport source
- 📝 **Synthétiser** en deux versions :
  - Une version **clinique concise** pour les professionnels de santé
  - Une version **vulgarisée** pour les patients

### Architecture

Le pipeline utilise **6 agents spécialisés** orchestrés par LangGraph :

```
Rapport brut
    ↓
1️⃣  PreprocessorAgent     (détection langue, qualité)
    ↓
2️⃣  ExtractorAgent        (NER médical)
    ↓
3️⃣  StructurerAgent       (JSON FHIR)
    ↓
4️⃣  VerifierAgent         (fidélité clinique)
    ↓
5️⃣  MedicalSummaryAgent   (synthèse pro) → 6️⃣ PatientSummaryAgent (vulgarisation)
    ↓
Rapports structurés + synthèses
```

## 🚀 Démarrage rapide

### 1. Installation

```bash
# Cloner et configurer l'environnement
cd /workspaces/ProjetI3AFD

# Installer les dépendances
pip install -r requirements.txt

# Optionnel : pré-télécharger les dépendances nltk
python -c "import nltk; nltk.download('punkt')"
```

### 2. Configurer Ollama (Recommandé — modèles locaux)

Ollama permet d'exécuter les modèles LLM **entièrement en local** sans dépendre de services cloud.

#### Installation d'Ollama

```bash
# macOS/Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Ou visitez https://ollama.ai
```

#### Télécharger les modèles

```bash
# Démarrer le serveur Ollama (une seule fois)
ollama serve

# Dans un autre terminal, télécharger les modèles
python setup_ollama.py --pull-models

# Vérifier l'installation
python setup_ollama.py --health
```

**Modèles utilisés :**
- **BioMistral-7B** (agents extraction, structuration, vérification)
- **OpenBioLLM-8B** (agents médicaux — alternative plus spécialisée)
- **Mistral-7B-Instruct** (vulgarisation patient)

### 3. Préparer les données

```bash
# Télécharger OpenI/IU X-Ray (3996 rapports)
# Le notebook inclut le code de téléchargement automatique

# Ou charger depuis HuggingFace Datasets
python loader.py  # Vérifier la classe OpenILoader

# Fichier attendu : data/rapports_clean.csv
#   - uid : ID du rapport
#   - findings : corps principal
#   - impression : résumé de référence (gold)
```

## 📚 Usage

### Via le notebook (Recommandé pour exploration)

```bash
jupyter notebook "RadioAI Pro.ipynb"
```

Le notebook contient :
- ✅ Vérification des dépendances
- 📥 Chargement des données OpenI
- 🔄 Exécution du pipeline sur des exemples
- 📊 Évaluation ROUGE-L, BERTScore, fidélité clinique
- 🎯 Ablation study (mono-agent vs multi-agents)

### Via scripts CLI

```bash
# Vérification de la configuration
python run_experiment.py --setup

# Évaluation simple (10 rapports)
python run_experiment.py --eval --samples 10 --backend ollama

# Ablation study (comparaison mono vs multi-agents)
python run_experiment.py --ablation --samples 5

# Avec HuggingFace au lieu d'Ollama (nécessite GPU)
python run_experiment.py --eval --samples 10 --backend huggingface
```

### Via Python directement

```python
from loader import OpenILoader
from model_manager import ModelManager
from pipeline import process_report

# 1. Charger un rapport
loader = OpenILoader(source="local", path="data/rapports_clean.csv")
report = next(loader.iter_reports(max_samples=1))

# 2. Initialiser le modèle (Ollama)
mm = ModelManager(backend="ollama", ollama_host="http://localhost:11434")

# 3. Traiter
result = process_report(
    raw_text=report["findings"],
    uid=report["uid"],
    model_manager=mm,
)

# 4. Afficher les résultats
print("Résumé clinique :", result["medical_summary"])
print("Vulgarisation :", result["patient_summary"])
print("Fidélité :", result["verified"]["fidelity_score"])
```

## 📊 Évaluation

### Métriques implémentées

1. **ROUGE-L** (chevauchement séquentiel)
   - Precision, Recall, F1
   - Comparaison avec résumés de référence

2. **BERTScore** (similarité sémantique)
   - Embeddings contextuels (distilbert)
   - F1 global

3. **Fidélité Clinique**
   - Score de fidélité (0-1) vs rapport source
   - Détection d'omissions et hallucinations
   - Taux de réussite (≥ 0.75)

### Ablation Study (Obligatoire)

Compare deux architectures :

| Aspect | Monolithique | Multi-agents |
|--------|-------------|-------------|
| **Approche** | 1 LLM, 1 prompt | 6 agents spécialisés |
| **Qualité** | Baseline | Amélioration attendue |
| **Robustesse** | Hallucinations fréquentes | Vérification intégrée |
| **Temps** | Rapide | Peut être plus lent |
| **Erreurs** | Moins détectables | Gestion dégradée |

**Commande :**
```bash
python run_experiment.py --ablation --samples 10
```

**Résultats sauvegardés en :** `data/exports/ablation_*.json`

## 📂 Structure du projet

```
ProjetI3AFD/
├── agents.py                    # 6 agents du pipeline
├── pipeline.py                  # Orchestration LangGraph
├── model_manager.py             # Gestion modèles (Ollama + HF)
├── loader.py                    # Chargement datasets
├── metrics.py                   # ROUGE-L, BERTScore, fidélité
├── setup_ollama.py             # Config Ollama
├── run_experiment.py           # Scripts d'exécution
├── RadioAI Pro.ipynb           # Notebook complet
├── requirements.txt            # Dépendances
├── README.md                   # Ceci
├── data/
│   ├── rapports_clean.csv      # Dataset (auto-téléchargé)
│   ├── resultats_evaluation.csv # Résultats sauvegardés
│   ├── exports/                # Résultats JSON
│   └── reports/                # Rapports bruts XML OpenI
└── __pycache__/               # Cache Python
```

## 🔧 Configuration avancée

### Mode HuggingFace (Au lieu d'Ollama)

Nécessite un GPU avec au minimum 24GB VRAM pour la quantification 4-bit.

```python
mm = ModelManager(
    backend="huggingface",
    use_4bit=True,
    device="cuda"
)
```

### Mode Mock (Tests sans GPU)

Pour les tests CI/CD :

```python
from model_manager import MockModelManager

mm = MockModelManager()
result = process_report("rapport test", "uid_001", mm)
```

### Configuration personnalisée des modèles

```python
config = {
    "preprocessor": "openbiollm",   # Utiliser OpenBioLLM pour preprocess
    "extractor":    "openbiollm",
    "structurer":   "openbiollm",
    "verifier":     "openbiollm",
    "med_summary":  "biomistral",
    "pat_summary":  "mistral-instruct",
}

result = process_report(raw_text, uid, mm, config=config)
```

## ⚠️ Gestion des cas dégradés

Le pipeline détecte automatiquement et gère les rapports de mauvaise qualité :

- **Rapport trop court** → Préprocesseur arrête
- **Extraction échouée** → Retour à heuristiques
- **Fidélité < 0.4** → Synthèses marquées comme dégradées
- **Erreurs LLM** → Fallback à structures JSON minimales

État dégradé indiqué par `state["degraded"] = True`.

## 📖 Documentation des modules

### `agents.py`

Chaque agent expose une interface `run(state) → state` où `state` est un dict enrichi à chaque étape.

**Agents :**
- `PreprocessorAgent` : Détection langue, qualité, sections
- `ExtractorAgent` : NER médical (anomalies, mesures, organes)
- `StructurerAgent` : Conversion JSON FHIR
- `VerifierAgent` : Vérification fidélité clinique
- `MedicalSummaryAgent` : Synthèse professionnelle
- `PatientSummaryAgent` : Vulgarisation patient

### `pipeline.py`

Orchestre les 6 agents via un graphe LangGraph avec :
- Flux séquentiel principal
- Routage conditionnel après préprocesseur et vérification
- Gestion des rapports dégradés
- Logging détaillé

### `model_manager.py`

Abstraction pour charger/générer avec :
- **Backend Ollama** (recommandé) : appels HTTP au serveur local
- **Backend HuggingFace** : chargement direct avec quantification 4-bit
- **Mode Mock** : réponses fictives pour tests

### `loader.py`

Classe `OpenILoader` pour charger :
- Fichiers JSON locaux
- Datasets HuggingFace (NIH CXR, MIMIC-CXR)
- Nettoyage et normalisation

### `metrics.py`

Évaluation quantitative :
- `compute_rouge_l()` : ROUGE-L sur corpus
- `compute_bert_score()` : BERTScore (F1 sémantique)
- `compute_clinical_fidelity()` : Score fidélité clinique
- `AblationStudy` : Comparaison mono vs multi-agents
- `format_results_table()` : Formatage pour rapports

## 🧪 Tests et Validation

```bash
# Test rapide (mode mock, données factices)
python -c "
from model_manager import MockModelManager
from pipeline import process_report

mm = MockModelManager()
result = process_report('Patient has mild cardiomegaly', 'test_001', mm)
print('✓ Pipeline opérationnel')
print('Résumé médical :', result['medical_summary'][:100])
"

# Vérifier les imports
python -c "import langgraph, langchain, rouge_score, bert_score; print('✓ Dépendances OK')"

# Vérifier Ollama
python setup_ollama.py --health
```

## 📝 Format des données

### Entrée (rapport brut)

```json
{
  "uid": "001",
  "findings": "Chest X-ray shows mild cardiomegaly with no acute cardiopulmonary disease.",
  "impression": "Mild cardiomegaly.",
  "split": "test"
}
```

### Sortie (après pipeline)

```json
{
  "uid": "001",
  "raw_text": "Chest X-ray shows...",
  "preprocessed": {
    "language": "en",
    "quality": "good",
    "completeness": 0.9,
    "sections_found": ["FINDINGS", "IMPRESSION"]
  },
  "extracted": {
    "findings": ["mild cardiomegaly"],
    "anomalies": [{"name": "cardiomegaly", "severity": "mild", "certainty": "definite"}],
    "organs": ["heart", "lungs"]
  },
  "structured": {
    "report_id": "001",
    "status": "final",
    "findings": {"heart": "mildly enlarged"},
    "impression": "Mild cardiomegaly without acute process",
    "anomalies": [...]
  },
  "verified": {
    "fidelity_score": 0.92,
    "missing_findings": [],
    "hallucinations": []
  },
  "medical_summary": "Mild cardiomegaly. Lungs clear. No acute disease.",
  "patient_summary": "Votre coeur est légèrement agrandi mais vos poumons sont en bonne santé.",
  "degraded": false,
  "errors": []
}
```

## 🎯 Résultats attendus

### Sur OpenI/IU X-Ray (test complet)

| Métrique | Baseline Mono | Multi-agents | Amélioration |
|----------|--------------|-------------|--------------|
| ROUGE-L F1 | ~0.35 | ~0.42 | +20% |
| BERTScore F1 | ~0.45 | ~0.53 | +18% |
| Fidélité | ~0.60 | ~0.80 | +33% |
| Taux pass (≥0.75) | ~35% | ~75% | +114% |
| Hallucinations | ~0.8/rapport | ~0.2/rapport | -75% |

*Résultats indicatifs — varient selon dataset et configuration.*

## 🐛 Dépannage

### Ollama ne démarre pas
```bash
# Vérifier l'installation
ollama --version

# Démarrer manuellement
ollama serve

# Vérifier la connexion
curl http://localhost:11434/api/tags
```

### Modèles non trouvés
```bash
# Lister les modèles disponibles
ollama list

# Télécharger manuellement
ollama pull biomistral
ollama pull mistral
```

### Erreur "cuda out of memory"
- Réduire la taille batch ou max_tokens
- Utiliser `--backend ollama` au lieu d'HuggingFace
- Utiliser un modèle plus petit (e.g., phi3-mini)

### Performances lentes
- Augmenter le nombre d'agents en parallèle (LangGraph le supporte)
- Utiliser quantification 4-bit (déjà activée)
- Réduire max_tokens aux agents

## 📚 Références

- **LangGraph** : https://langchain-ai.github.io/langgraph/
- **LangChain** : https://langchain.com/
- **Ollama** : https://ollama.ai/
- **OpenI Dataset** : https://openi.nlm.nih.gov/
- **FHIR DiagnosticReport** : https://hl7.org/fhir/diagnosticreport.html
- **ROUGE** : https://en.wikipedia.org/wiki/ROUGE_(metric)
- **BERTScore** : https://arxiv.org/abs/1904.09675

## 👥 Équipe et Attribution

Développé pour le projet I3AFD (Master Ingénierie 3A, INPT/ENSEEIHT).

**Composants clés :**
- Architecture multi-agents : LangGraph
- Modèles : BioMistral, OpenBioLLM, Mistral (Mistral AI)
- Données : OpenI/IU X-Ray (NIH)
- Évaluation : ROUGE, BERTScore, fidélité clinique

## 📄 Licence

Projet académique. Utilisation libre pour la recherche et l'éducation.

---

**Questions ?** Consultez le notebook `RadioAI Pro.ipynb` pour une démonstration interactive.

Projet 4 : Structuration et synthèse agentique de comptes rendus de radiologie Les comptes rendus de radiologie sont généralement rédigés sous forme de texte libre, ce qui rend leur exploitation automatique difficile. Dans ce projet, nous devrons concevoir un système agentique à base de modèles de langage capable d’analyser un rapport radiologique 
