#!/usr/bin/env python3
"""
Script de configuration d'Ollama pour RadioAI Pro.

Usage:
    python setup_ollama.py --pull-models    # Télécharger les modèles
    python setup_ollama.py --check           # Vérifier Ollama
    python setup_ollama.py --health          # Test de santé
"""

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Modèles recommandés
MODELS = {
    "biomistral": "biomistral",  # BioMistral-7B
    "openbiollm": "openbiollm",  # OpenBioLLM-8B
    "mistral": "mistral",        # Mistral-7B-Instruct
}

OLLAMA_HOST = "http://localhost:11434"


def check_ollama_installed() -> bool:
    """Vérifie qu'Ollama est installé."""
    try:
        result = subprocess.run(["ollama", "--version"], capture_output=True)
        if result.returncode == 0:
            logger.info("✓ Ollama trouvé: %s", result.stdout.decode().strip())
            return True
    except FileNotFoundError:
        pass
    return False


def start_ollama_server() -> bool:
    """Démarre le serveur Ollama en arrière-plan."""
    logger.info("Démarrage du serveur Ollama…")
    try:
        # Vérifier d'abord si Ollama écoute déjà
        resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=2)
        if resp.status_code == 200:
            logger.info("✓ Serveur Ollama déjà actif sur %s", OLLAMA_HOST)
            return True
    except requests.exceptions.RequestException:
        pass

    # Lancer Ollama
    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        logger.info("Serveur Ollama lancé. Attente de disponibilité…")
        
        # Attendre le démarrage (max 30s)
        for i in range(30):
            try:
                resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=1)
                if resp.status_code == 200:
                    logger.info("✓ Serveur Ollama prêt !")
                    return True
            except requests.exceptions.RequestException:
                time.sleep(1)
        
        logger.error("✗ Timeout : Ollama n'a pas démarré en 30s")
        return False
        
    except Exception as e:
        logger.error("✗ Erreur au lancement d'Ollama: %s", e)
        return False


def pull_models() -> bool:
    """Télécharge les modèles recommandés."""
    logger.info("Téléchargement des modèles…")
    
    all_ok = True
    for key, model_name in MODELS.items():
        logger.info(f"Téléchargement {key} ({model_name})…")
        try:
            result = subprocess.run(
                ["ollama", "pull", model_name],
                capture_output=True,
                timeout=600,  # 10 min par modèle
            )
            if result.returncode == 0:
                logger.info(f"  ✓ {key} téléchargé")
            else:
                logger.error(f"  ✗ Erreur: {result.stderr.decode()}")
                all_ok = False
        except subprocess.TimeoutExpired:
            logger.error(f"  ✗ Timeout pour {key}")
            all_ok = False
        except Exception as e:
            logger.error(f"  ✗ Exception: {e}")
            all_ok = False
    
    return all_ok


def list_models() -> None:
    """Liste les modèles disponibles."""
    try:
        resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            if models:
                logger.info("Modèles Ollama disponibles :")
                for m in models:
                    logger.info(f"  - {m['name']} ({m['size']} bytes)")
            else:
                logger.info("Aucun modèle disponible. Lancez: python setup_ollama.py --pull-models")
        else:
            logger.error("Erreur lors de la récupération des modèles")
    except Exception as e:
        logger.error("Impossible de joindre Ollama: %s", e)


def health_check() -> bool:
    """Test de santé complet."""
    logger.info("=== Test de santé Ollama ===")
    
    # 1. Vérifier installation
    if not check_ollama_installed():
        logger.error("✗ Ollama n'est pas installé. Visitez https://ollama.ai")
        return False
    
    # 2. Vérifier serveur
    try:
        resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=2)
        if resp.status_code == 200:
            logger.info("✓ Serveur Ollama actif")
        else:
            logger.error("✗ Serveur Ollama non réactif")
            return False
    except requests.exceptions.RequestException:
        logger.error("✗ Impossible de joindre Ollama sur %s", OLLAMA_HOST)
        return False
    
    # 3. Vérifier modèles
    list_models()
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Configuration Ollama pour RadioAI Pro"
    )
    parser.add_argument(
        "--pull-models",
        action="store_true",
        help="Télécharger les modèles recommandés"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Lister les modèles disponibles"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Vérifier qu'Ollama est installé"
    )
    parser.add_argument(
        "--health",
        action="store_true",
        help="Test de santé complet"
    )
    parser.add_argument(
        "--start",
        action="store_true",
        help="Démarrer le serveur Ollama"
    )
    
    args = parser.parse_args()
    
    if args.health or (not any([args.pull_models, args.check, args.list, args.start])):
        # Par défaut, faire un health check
        success = health_check()
        sys.exit(0 if success else 1)
    
    if args.check:
        success = check_ollama_installed()
        sys.exit(0 if success else 1)
    
    if args.start:
        success = start_ollama_server()
        sys.exit(0 if success else 1)
    
    if args.list:
        list_models()
        sys.exit(0)
    
    if args.pull_models:
        if not check_ollama_installed():
            logger.error("Ollama n'est pas installé")
            sys.exit(1)
        
        if not start_ollama_server():
            logger.error("Impossible de démarrer Ollama")
            sys.exit(1)
        
        if pull_models():
            logger.info("\n✓ Tous les modèles ont été téléchargés !")
            sys.exit(0)
        else:
            logger.error("\n✗ Certains modèles n'ont pas pu être téléchargés")
            sys.exit(1)


if __name__ == "__main__":
    main()
