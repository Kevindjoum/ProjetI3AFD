#!/usr/bin/env python3
"""
Script de test rapide du pipeline RadioAI Pro.

Usage:
    python test_pipeline.py                    # Test avec données mock
    python test_pipeline.py --real --samples 3 # Test avec vraies données (ollama)
"""

import json
import sys
from pathlib import Path


def test_mock():
    """Test le pipeline en mode mock (aucune dépendance externe)."""
    print("=" * 70)
    print("TEST RAPIDE — Mode Mock (aucun GPU/Ollama requis)")
    print("=" * 70)
    
    from model_manager import MockModelManager
    from pipeline import process_report
    from agents import initial_state
    
    # 1. Créer un rapport test
    raw_report = """
    CHEST X-RAY
    
    FINDINGS:
    The heart size is within normal limits. The lungs are clear bilaterally. 
    No focal consolidation. No pleural effusion. The mediastinum is unremarkable.
    No acute cardiopulmonary disease.
    
    IMPRESSION:
    No acute findings. Normal chest x-ray.
    """
    
    print("\n1. Initialisation du ModelManager (mock)…")
    mm = MockModelManager()
    print("   ✓ ModelManager mock créé")
    
    print("\n2. Traitement du rapport…")
    result = process_report(raw_report, "TEST_001", mm)
    
    print("\n3. Résultats :")
    print(f"   ✓ Langue détectée : {result['preprocessed'].get('language', 'unknown')}")
    print(f"   ✓ Qualité : {result['preprocessed'].get('quality', 'unknown')}")
    print(f"   ✓ Anomalies trouvées : {len(result['extracted'].get('anomalies', []))}")
    print(f"   ✓ Fidélité clinique : {result['verified'].get('fidelity_score', 0):.2f}")
    print(f"   ✓ État dégradé : {result.get('degraded', False)}")
    
    print("\n4. Synthèse médicale :")
    print(f"   {result['medical_summary'][:150]}…")
    
    print("\n5. Vulgarisation patient :")
    print(f"   {result['patient_summary'][:150]}…")
    
    print("\n✓ Test complet avec succès !")
    return True


def test_real(n_samples=3):
    """Test avec vraies données et Ollama."""
    print("=" * 70)
    print(f"TEST RÉEL — {n_samples} rapport(s) avec Ollama")
    print("=" * 70)
    
    # 1. Vérifier Ollama
    print("\n1. Vérification d'Ollama…")
    try:
        import requests
        resp = requests.get("http://localhost:11434/api/tags", timeout=2)
        if resp.status_code != 200:
            print("   ✗ Ollama non réactif")
            return False
        models = resp.json().get("models", [])
        print(f"   ✓ Ollama actif, {len(models)} modèle(s) disponible(s)")
    except Exception as e:
        print(f"   ✗ Erreur Ollama: {e}")
        print("   → Lancez: ollama serve")
        return False
    
    # 2. Charger les données
    print(f"\n2. Chargement de {n_samples} rapport(s)…")
    try:
        from loader import OpenILoader
        loader = OpenILoader(source="local", path="data/rapports_clean.csv")
        reports = list(loader.iter_reports(split="test", max_samples=n_samples))
        print(f"   ✓ {len(reports)} rapport(s) chargés")
    except Exception as e:
        print(f"   ✗ Erreur chargement données: {e}")
        print("   → Assurez-vous que data/rapports_clean.csv existe")
        return False
    
    # 3. Initialiser le modèle
    print("\n3. Initialisation du ModelManager (Ollama)…")
    try:
        from model_manager import ModelManager
        mm = ModelManager(backend="ollama", ollama_host="http://localhost:11434")
        print("   ✓ ModelManager Ollama créé")
    except Exception as e:
        print(f"   ✗ Erreur: {e}")
        return False
    
    # 4. Traiter les rapports
    print(f"\n4. Traitement de {len(reports)} rapport(s)…")
    from pipeline import process_report
    
    results = []
    for i, report in enumerate(reports):
        try:
            print(f"   [{i+1}/{len(reports)}] uid={report['uid']}")
            result = process_report(
                raw_text=report["findings"],
                uid=report["uid"],
                model_manager=mm,
            )
            results.append(result)
        except Exception as e:
            print(f"   ✗ Erreur: {e}")
            continue
    
    print(f"\n   ✓ {len(results)}/{len(reports)} rapport(s) traités")
    
    # 5. Afficher les résultats
    print("\n5. Résultats :")
    for i, result in enumerate(results):
        print(f"\n   [{i+1}] uid={result['uid']}")
        print(f"       Fidélité: {result['verified'].get('fidelity_score', 0):.2f}")
        print(f"       Résumé: {result['medical_summary'][:80]}…")
    
    # 6. Évaluer
    print("\n6. Évaluation…")
    try:
        from metrics import compute_clinical_fidelity
        fidelity = compute_clinical_fidelity(results)
        print(f"   Fidélité moyenne : {fidelity['mean_fidelity']:.2f}")
        print(f"   Taux pass (≥0.75) : {fidelity['pass_rate']:.1%}")
        print(f"   Omissions moyennes : {fidelity['mean_missing']:.1f}")
    except Exception as e:
        print(f"   ✗ Erreur évaluation: {e}")
        return False
    
    print("\n✓ Test complet avec succès !")
    return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test du pipeline RadioAI Pro")
    parser.add_argument("--real", action="store_true", help="Utiliser vraies données + Ollama")
    parser.add_argument("--samples", type=int, default=3, help="Nombre d'échantillons")
    
    args = parser.parse_args()
    
    try:
        if args.real:
            success = test_real(args.samples)
        else:
            success = test_mock()
        
        sys.exit(0 if success else 1)
    
    except KeyboardInterrupt:
        print("\n\n⚠ Test interrompu par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Erreur inattendue: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
