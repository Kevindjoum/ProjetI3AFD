#!/usr/bin/env python3
"""
Script d'exécution du pipeline RadioAI Pro sur OpenI/IU X-Ray.

Usage:
    # Vérifier la config
    python run_experiment.py --setup
    
    # Lancer une évaluation complète (petit dataset test)
    python run_experiment.py --eval --samples 10
    
    # Ablation study (compare mono vs multi-agents)
    python run_experiment.py --ablation --samples 5
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="RadioAI Pro Pipeline Execution")
    parser.add_argument("--setup", action="store_true", help="Setup and verify configuration")
    parser.add_argument("--eval", action="store_true", help="Run evaluation on dataset")
    parser.add_argument("--ablation", action="store_true", help="Run ablation study")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples to process")
    parser.add_argument("--output", type=str, default="data/exports", help="Output directory")
    parser.add_argument("--backend", type=str, default="ollama", help="Model backend (ollama|huggingface)")
    
    args = parser.parse_args()
    
    if args.setup:
        setup_check(args.backend)
    elif args.eval:
        run_evaluation(args.samples, args.output, args.backend)
    elif args.ablation:
        run_ablation(args.samples, args.output, args.backend)
    else:
        parser.print_help()


def setup_check(backend: str):
    """Vérifier que tout est prêt."""
    logger.info("=== RadioAI Pro Setup Check ===\n")
    
    # 1. Vérifier imports
    logger.info("1. Vérification des dépendances Python…")
    try:
        import langgraph
        import langchain
        import transformers
        import pandas as pd
        import rouge_score
        import bert_score
        logger.info("   ✓ Toutes les dépendances importables")
    except ImportError as e:
        logger.error(f"   ✗ Dépendance manquante: {e}")
        return False
    
    # 2. Vérifier le backend
    logger.info(f"\n2. Vérification du backend {backend}…")
    
    if backend == "ollama":
        try:
            import requests
            resp = requests.get("http://localhost:11434/api/tags", timeout=2)
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                logger.info(f"   ✓ Ollama actif, {len(models)} modèle(s) disponibles:")
                for m in models[:5]:
                    logger.info(f"     - {m['name']}")
            else:
                logger.error("   ✗ Ollama non réactif (http status != 200)")
                return False
        except Exception as e:
            logger.error(f"   ✗ Ollama non disponible: {e}")
            logger.info("   → Lancez: ollama serve")
            return False
    else:
        try:
            import torch
            if torch.cuda.is_available():
                logger.info(f"   ✓ GPU disponible: {torch.cuda.get_device_name(0)}")
            else:
                logger.warning("   ⚠ Pas de GPU détecté (CPU mode)")
        except ImportError:
            logger.error("   ✗ Torch non disponible")
            return False
    
    # 3. Vérifier les données
    logger.info("\n3. Vérification des données…")
    data_path = Path("data")
    if (data_path / "rapports_clean.csv").exists():
        df = pd.read_csv(data_path / "rapports_clean.csv")
        logger.info(f"   ✓ Dataset trouvé: {len(df)} rapports")
    else:
        logger.warning("   ⚠ data/rapports_clean.csv non trouvé")
        logger.info("   → Exécutez le notebook pour télécharger les données")
    
    logger.info("\n✓ Setup complet!")
    return True


def run_evaluation(n_samples: int, output_dir: str, backend: str):
    """Lancer une évaluation du pipeline."""
    logger.info(f"=== Evaluation Pipeline ({n_samples} rapports) ===\n")
    
    try:
        from loader import OpenILoader
        from model_manager import ModelManager, MockModelManager
        from pipeline import process_report
        from metrics import evaluate_corpus, compute_clinical_fidelity
    except ImportError as e:
        logger.error(f"Erreur d'import: {e}")
        return False
    
    # 1. Charger les données
    logger.info(f"1. Chargement de {n_samples} rapports…")
    loader = OpenILoader(source="local", path="data/rapports_clean.csv")
    reports = list(loader.iter_reports(split="test", max_samples=n_samples))
    logger.info(f"   ✓ {len(reports)} rapports chargés")
    
    # 2. Initialiser le modèle
    logger.info("\n2. Initialisation du ModelManager…")
    if backend == "mock":
        mm = MockModelManager()
        logger.info("   ✓ Mode mock (test rapide)")
    else:
        mm = ModelManager(backend=backend, ollama_host="http://localhost:11434")
        logger.info(f"   ✓ Backend: {backend}")
    
    # 3. Traiter les rapports
    logger.info(f"\n3. Traitement des {len(reports)} rapports…")
    results = []
    for i, report in enumerate(reports):
        try:
            logger.info(f"   [{i+1}/{len(reports)}] Traitement uid={report['uid']}")
            result = process_report(
                raw_text=report["findings"],
                uid=report["uid"],
                model_manager=mm,
            )
            results.append(result)
        except Exception as e:
            logger.error(f"   ✗ Erreur sur {report['uid']}: {e}")
            continue
    
    logger.info(f"\n   ✓ {len(results)}/{len(reports)} rapports traités")
    
    # 4. Évaluer
    logger.info("\n4. Évaluation des résultats…")
    references = [r.get("impression", "") for r in reports[:len(results)]]
    
    metrics = evaluate_corpus(results, references, summary_key="medical_summary")
    fidelity = compute_clinical_fidelity(results)
    
    logger.info("\n=== Résultats ===")
    logger.info(f"  ROUGE-L F1:      {metrics.get('rouge_l_f1', 0):.3f}")
    logger.info(f"  BERTScore F1:    {metrics.get('bert_f1', 0):.3f}")
    logger.info(f"  Fidélité:        {fidelity.get('mean_fidelity', 0):.3f}")
    logger.info(f"  Taux pass:       {fidelity.get('pass_rate', 0):.1%}")
    
    # 5. Sauvegarder
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results_file = output_path / f"evaluation_{n_samples}_samples.json"
    with open(results_file, "w") as f:
        json.dump({"metrics": metrics, "fidelity": fidelity, "sample_count": len(results)}, f, indent=2)
    
    logger.info(f"\n✓ Résultats sauvegardés: {results_file}")
    return True


def run_ablation(n_samples: int, output_dir: str, backend: str):
    """Lancer l'ablation study."""
    logger.info(f"=== Ablation Study ({n_samples} rapports) ===\n")
    
    try:
        from loader import OpenILoader
        from model_manager import ModelManager, MockModelManager
        from metrics import AblationStudy, format_results_table
    except ImportError as e:
        logger.error(f"Erreur d'import: {e}")
        return False
    
    # 1. Charger les données
    logger.info(f"1. Chargement de {n_samples} rapports…")
    loader = OpenILoader(source="local", path="data/rapports_clean.csv")
    reports = list(loader.iter_reports(split="test", max_samples=n_samples))
    logger.info(f"   ✓ {len(reports)} rapports chargés")
    
    # 2. Initialiser le modèle
    logger.info("\n2. Initialisation du ModelManager…")
    if backend == "mock":
        mm = MockModelManager()
        logger.info("   ✓ Mode mock (test rapide)")
    else:
        mm = ModelManager(backend=backend, ollama_host="http://localhost:11434")
        logger.info(f"   ✓ Backend: {backend}")
    
    # 3. Exécuter l'ablation
    logger.info(f"\n3. Comparaison monolithique vs multi-agents…")
    ablation = AblationStudy(mm)
    
    references = [r.get("impression", "") for r in reports]
    result = ablation.run(reports, references)
    
    # 4. Afficher les résultats
    logger.info("\n=== Résultats Ablation ===")
    
    df = format_results_table(result)
    logger.info("\n" + str(df.to_string(index=False)))
    
    timing = result["timing"]
    logger.info(f"\nTemps d'exécution:")
    logger.info(f"  Monolithique:   {timing['monolithic_s']:.2f}s")
    logger.info(f"  Multi-agents:   {timing['multi_agent_s']:.2f}s")
    speedup = timing['monolithique_s'] / timing['multi_agent_s'] if timing['multi_agent_s'] > 0 else 0
    logger.info(f"  Ratio:          {speedup:.2f}x")
    
    # 5. Sauvegarder
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results_file = output_path / f"ablation_{n_samples}_samples.json"
    with open(results_file, "w") as f:
        # Sérialisation JSON des résultats
        json_result = {
            "monolithic": result["monolithic"],
            "multi_agent": result["multi_agent"],
            "delta": result["delta"],
            "timing": result["timing"],
        }
        json.dump(json_result, f, indent=2)
    
    logger.info(f"\n✓ Résultats sauvegardés: {results_file}")
    return True


if __name__ == "__main__":
    main()
