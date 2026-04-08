"""
Gestionnaire de modèles pour Ollama (LOCAL ONLY).

Utilise deux modèles quantifiés (Q3_K_S et IQ3_XS) pour économiser la VRAM :
  1. BioMistral-7B-GGUF:Q3_K_S       → agents médicaux (extraction/structuration)
  2. Mistral-7B-Instruct-GGUF:IQ3_XS → vulgarisation patient

Prérequis : ollama serve doit être lancé en arrière-plan
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import requests

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Registre de modèles Ollama (quantifiés GGUF)
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    # Agent médical : BioMistral-7B Q3_K_S (faible VRAM)
    "biomistral": {
        "ollama_id": "hf.co/MaziyarPanahi/BioMistral-7B-GGUF:Q3_K_S",
        "task":      "medical",
        "context":   4096,
        "instruct":  False,
    },
    # Agent patient : Mistral-7B-Instruct IQ3_XS (ultra-compressé)
    "mistral-instruct": {
        "ollama_id": "hf.co/MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF:IQ3_XS",
        "task":      "patient",
        "context":   8192,
        "instruct":  True,
    },
}


# ---------------------------------------------------------------------------
# Classe principale — Support Ollama + HuggingFace
# ---------------------------------------------------------------------------

@dataclass
class ModelManager:
    """
    Gestionnaire de modèles supportant Ollama et HuggingFace.

    Mode Ollama (par défaut, recommandé) :
        mm = ModelManager(backend="ollama", ollama_host="http://localhost:11434")
        text = mm.generate("biomistral", "Votre prompt ici")

    Mode HuggingFace (nécessite GPU / quantification 4-bit) :
        mm = ModelManager(backend="huggingface", use_4bit=True)
        text = mm.generate("openbiollm", "Votre prompt ici")
    """

    backend: str = "ollama"  # "ollama" ou "huggingface"
    ollama_host: str = "http://localhost:11434"
    use_4bit: bool = True
    device: str = field(default_factory=lambda: "cuda")
    _cache: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    # ------------------------------------------------------------------ #
    # Interface publique
    # ------------------------------------------------------------------ #

    def generate(
        self,
        model_key: str,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
    ) -> str:
        """
        Génère du texte avec le modèle spécifié.
        Utilise Ollama ou HuggingFace selon la configuration.
        """
        if self.backend == "ollama":
            return self._generate_ollama(model_key, prompt, max_new_tokens, temperature)
        else:
            return self._generate_huggingface(model_key, prompt, max_new_tokens, temperature)

    def get_available_models(self) -> list[str]:
        """Liste les modèles disponibles (Ollama ou HF)."""
        if self.backend == "ollama":
            return self._list_ollama_models()
        return list(MODEL_REGISTRY.keys())

    def health_check(self) -> bool:
        """Vérifie que le backend (Ollama ou GPU) est disponible."""
        if self.backend == "ollama":
            try:
                resp = requests.get(f"{self.ollama_host}/api/tags", timeout=2)
                return resp.status_code == 200
            except Exception as e:
                logger.warning("Ollama not reachable: %s", e)
                return False
        else:
            try:
                import torch
                return torch.cuda.is_available() or torch.backends.mps.is_available()
            except ImportError:
                return False

    # ------------------------------------------------------------------ #
    # Ollama backend
    # ------------------------------------------------------------------ #

    def _generate_ollama(
        self,
        model_key: str,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        """Appel à Ollama via HTTP."""
        if model_key not in MODEL_REGISTRY:
            raise ValueError(f"Modèle '{model_key}' inconnu.")

        ollama_name = MODEL_REGISTRY[model_key].get("ollama_name", model_key)
        
        try:
            url = f"{self.ollama_host}/api/generate"
            payload = {
                "model": ollama_name,
                "prompt": prompt,
                "stream": False,
                "temperature": temperature,
                "num_predict": max_new_tokens,
            }
            
            logger.debug("Ollama request: model=%s", ollama_name)
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            
            data = resp.json()
            return data.get("response", "").strip()
            
        except requests.exceptions.ConnectionError as e:
            logger.error("Impossible de joindre Ollama sur %s: %s", self.ollama_host, e)
            raise RuntimeError(
                f"Ollama non disponible. Lancez: ollama serve\n"
                f"URL configurée : {self.ollama_host}"
            )
        except Exception as e:
            logger.error("Erreur Ollama: %s", e)
            raise

    def _list_ollama_models(self) -> list[str]:
        """Liste les modèles disponibles sur le serveur Ollama."""
        try:
            resp = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            resp.raise_for_status()
            data = resp.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception as e:
            logger.warning("Erreur lors de la liste des modèles Ollama: %s", e)
            return []

    # ------------------------------------------------------------------ #
    # HuggingFace backend (quantification 4-bit)
    # ------------------------------------------------------------------ #

    def _generate_huggingface(
        self,
        model_key: str,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        """Génère avec HuggingFace transformers (mode quantifié)."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError("transformers est requis pour le mode HuggingFace")

        if model_key not in self._cache:
            self._cache[model_key] = self._load_hf_model(model_key)

        model, tokenizer = self._cache[model_key]

        # Tokeniser l'input
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        if self.device == "cuda":
            import torch
            inputs = inputs.to("cuda")

        # Générer
        try:
            import torch
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
        except Exception:
            # Fallback sans sampling
            outputs = model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Retirer le prompt original
        return generated_text[len(prompt):].strip()

    def _load_hf_model(self, model_key: str) -> tuple[Any, Any]:
        """Charge un modèle HuggingFace avec quantification 4-bit."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError:
            raise ImportError("transformers et torch requis")

        if model_key not in MODEL_REGISTRY:
            raise ValueError(f"Modèle '{model_key}' inconnu.")

        hf_id = MODEL_REGISTRY[model_key]["hf_id"]
        logger.info("Chargement HF: %s", hf_id)

        tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16,
            "device_map": "auto",
        }

        if self.use_4bit and self.device == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["quantization_config"] = bnb_config

        model = AutoModelForCausalLM.from_pretrained(hf_id, **model_kwargs)
        model.eval()

        logger.info("Modèle '%s' chargé ✓", model_key)
        return model, tokenizer

    def unload(self, model_key: str) -> None:
        """Libère le modèle mis en cache."""
        if model_key in self._cache:
            del self._cache[model_key]
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            logger.info("Modèle '%s' déchargé.", model_key)


# ---------------------------------------------------------------------------
# MockManager pour tests sans GPU
# ---------------------------------------------------------------------------

class MockModelManager(ModelManager):
    """
    Remplace les vrais LLM par des réponses fictives déterministes.
    Utile pour les tests unitaires et CI sans GPU.
    """

    MOCK_RESPONSES: dict[str, str] = {
        "preprocess": '{"language":"en","quality":"good","completeness":0.9,"issues":[]}',
        "extract": (
            '{"findings":["No acute cardiopulmonary disease"],'
            '"anomalies":["Mild cardiomegaly"],'
            '"measurements":{"heart_size":"mildly enlarged"},'
            '"organs":["lungs","heart","mediastinum"]}'
        ),
        "structure": (
            '{"report_id":"mock_001","findings":{"lungs":"clear bilaterally",'
            '"heart":"mildly enlarged","mediastinum":"normal"},'
            '"impression":"Mild cardiomegaly. No acute process.",'
            '"anomalies":[{"name":"cardiomegaly","severity":"mild",'
            '"location":"heart","certainty":"definite"}]}'
        ),
        "verify": '{"fidelity_score":0.92,"missing_findings":[],"hallucinations":[],"pass":true}',
        "medical_summary": (
            "Mild cardiomegaly without acute cardiopulmonary disease. "
            "Lungs clear. No effusion or pneumothorax."
        ),
        "patient_summary": (
            "Votre radiographie montre que votre coeur est légèrement agrandi, "
            "mais vos poumons sont en bonne santé. Il n'y a pas d'infection ni de liquide "
            "autour des poumons. Votre médecin vous expliquera les prochaines étapes."
        ),
    }

    def get_pipeline(self, model_key: str) -> Any:
        return None  # inutilisé en mode mock

    def generate(self, model_key: str, prompt: str, **kwargs) -> str:
        # Inférence du type de réponse à partir du contenu du prompt
        p = prompt.lower()
        if "preprocess" in p or "language" in p or "quality" in p:
            return self.MOCK_RESPONSES["preprocess"]
        if "extract" in p or "anomal" in p or "entit" in p:
            return self.MOCK_RESPONSES["extract"]
        if "json" in p and ("structur" in p or "fhir" in p):
            return self.MOCK_RESPONSES["structure"]
        if "fidel" in p or "verif" in p or "hallucinat" in p:
            return self.MOCK_RESPONSES["verify"]
        if "patient" in p or "vulgarise" in p or "simple" in p:
            return self.MOCK_RESPONSES["patient_summary"]
        return self.MOCK_RESPONSES["medical_summary"]
