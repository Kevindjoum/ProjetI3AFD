"""
Implémentation des six agents du pipeline radiologique.

Chaque agent est une classe autonome exposant une méthode `run(state) -> state`.
L'état (AgentState) est un dict partagé enrichi à chaque étape.

Agents :
    1. PreprocessorAgent  — détection langue, qualité, complétude
    2. ExtractorAgent     — extraction d'entités médicales (NER médical)
    3. StructurerAgent    — production du JSON structuré FHIR-compatible
    4. VerifierAgent      — vérification fidélité clinique
    5. MedicalSummaryAgent — synthèse pour professionnel de santé
    6. PatientSummaryAgent — vulgarisation pour le patient
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schéma de l'état partagé
# ---------------------------------------------------------------------------

def initial_state(raw_text: str, uid: str = "unknown") -> dict:
    """Crée l'état initial à partir d'un rapport brut."""
    return {
        "uid":             uid,
        "raw_text":        raw_text,
        "preprocessed":    None,   # résultat Agent 1
        "extracted":       None,   # résultat Agent 2
        "structured":      None,   # résultat Agent 3
        "verified":        None,   # résultat Agent 4
        "medical_summary": None,   # résultat Agent 5
        "patient_summary": None,   # résultat Agent 6
        "errors":          [],     # journal des erreurs non fatales
        "degraded":        False,  # True si le rapport est incomplet / ambigu
    }


# ---------------------------------------------------------------------------
# Classe de base
# ---------------------------------------------------------------------------

class BaseAgent:
    """Classe de base pour tous les agents."""

    name: str = "base"

    def __init__(self, model_manager, model_key: str):
        self.mm        = model_manager
        self.model_key = model_key

    def run(self, state: dict) -> dict:
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Utilitaires communs                                                  #
    # ------------------------------------------------------------------ #

    def _llm(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.05) -> str:
        return self.mm.generate(
            self.model_key,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

    @staticmethod
    def _parse_json(text: str) -> dict | list | None:
        """
        Extrait le premier bloc JSON valide d'une chaîne de texte.
        Retourne None si aucun JSON valide n'est trouvé.
        """
        # Cherche d'abord un bloc ```json … ```
        match = re.search(r"```json\s*([\s\S]+?)\s*```", text)
        if match:
            candidate = match.group(1)
        else:
            # Cherche le premier { ou [
            start = min(
                (text.find("{") if "{" in text else len(text)),
                (text.find("[") if "[" in text else len(text)),
            )
            candidate = text[start:]

        # Tente de parser en retranchant des caractères depuis la fin si nécessaire
        for end in range(len(candidate), 0, -1):
            try:
                return json.loads(candidate[:end])
            except json.JSONDecodeError:
                continue
        return None

    def _log_error(self, state: dict, message: str) -> None:
        logger.warning("[%s] %s", self.name, message)
        state["errors"].append({"agent": self.name, "message": message})


# ---------------------------------------------------------------------------
# Agent 1 — Préprocesseur
# ---------------------------------------------------------------------------

class PreprocessorAgent(BaseAgent):
    """
    Analyse le rapport brut pour détecter :
      - la langue (en/fr/autre)
      - la qualité générale (bonne / dégradée / très dégradée)
      - les sections présentes (FINDINGS, IMPRESSION, etc.)
      - les anomalies structurelles (rapport tronqué, illisible, etc.)
    """

    name = "PreprocessorAgent"

    PROMPT_TEMPLATE = """\
You are a medical NLP preprocessing specialist.

Analyze the following radiology report and return a JSON object with exactly these fields:
- language: detected language code (e.g. en, fr)
- quality: one of good, degraded, poor
- completeness: float 0.0-1.0 (1.0 = all expected sections present)
- sections: list of detected section labels only (e.g. COMPARISON, INDICATION, FINDINGS, IMPRESSION)

Rules:
- If no sections are detected, return an empty list.
- If the text is empty or unreadable, set quality="poor" and completeness=0.0.
- Return ONLY the JSON object, no explanation.

Examples:

Report:
Chest radiographs XXXX. XXXX-year-old male, chest pain. The cardiomediastinal silhouette is within normal limits for size and contour. The lungs are normally inflated without evidence of focal airspace disease, pleural effusion, or pneumothorax. Stable calcified granuloma within the right upper lung. No acute bone abnormality. No acute cardiopulmonary process.
Output:
{{"language":"en","quality":"good","completeness":1.0,"sections":["COMPARISON","INDICATION","FINDINGS","IMPRESSION"]}}

Report:
 Clear lungs. No acute abnormality.
Output:
{{"language":"en","quality":"degraded","completeness":0.25,"sections":["FINDINGS"]}}

Report:
XXXX XXXX XXXX
Output:
{{"language":"en","quality":"poor","completeness":0.0,"sections":[]}}

Now analyze the following report:

```
{report}
```
"""

    def run(self, state: dict) -> dict:
        raw = state["raw_text"]

        if len(raw.strip()) < 20:
            state["preprocessed"] = {
                "language": "unknown",
                "quality": "poor",
                "completeness": 0.0,
                "sections": [],
            }
            state["degraded"] = True
            return state

        prompt = self.PROMPT_TEMPLATE.format(report=raw[:3000])
        response = self._llm(prompt, max_new_tokens=256, temperature=0.0)
        parsed = self._parse_json(response)

        if parsed is None:
            self._log_error(state, f"JSON invalide : {response[:100]}")
            parsed = self._fallback_preprocess(raw)
            state["degraded"] = True

        state["preprocessed"] = parsed

        if parsed.get("quality") in ("poor",) or parsed.get("completeness", 1.0) < 0.3:
            state["degraded"] = True

        return state

    @staticmethod
    def _fallback_preprocess(text: str) -> dict:
        """Analyse heuristique simple en cas d'échec du LLM."""
        sections = []
        for kw in ["COMPARISON", "INDICATION", "FINDINGS", "IMPRESSION"]:
            if kw.lower() in text.lower():
                sections.append(kw)

        return {
            "language": "en",
            "quality": "degraded",
            "completeness": len(sections) / 4.0,
            "sections": sections,
        }


# ---------------------------------------------------------------------------
# Agent 2 — Extracteur d'entités médicales
# ---------------------------------------------------------------------------

class ExtractorAgent(BaseAgent):
    """
    Extrait les entités médicales clés :
      - findings (observations normales et pathologiques)
      - anomalies (avec localisation et degré de certitude)
      - mesures quantitatives (taille, densité…)
      - organes mentionnés
    """

    name = "ExtractorAgent"

    PROMPT_TEMPLATE = """\
You are a medical information extraction specialist.

Analyze the following radiology report and return a JSON object with exactly these fields:
- "findings": list of key abnormalities or observations (e.g. ["cardiomegaly", "pulmonary opacity"])
- "locations": list of anatomical locations mentioned (e.g. ["right lung", "mediastinum"])
- "measurements": list of any quantitative values (e.g. ["heart size borderline", "artery diameter enlarged"])
- "impression": concise summary of the main conclusion (if present)

Rules:
- Use standardized medical terminology when possible.
- If no abnormalities are found, return an empty list for "findings".
- If no impression is present, set "impression" to "" (empty string).
- Return ONLY the JSON object, no explanation.

Examples:

Report:
Borderline cardiomegaly. Enlarged pulmonary arteries. Clear lungs. No acute pulmonary findings.
Output:
{{
  "findings": ["cardiomegaly", "enlarged pulmonary arteries"],
  "locations": ["heart", "pulmonary arteries", "lungs"],
  "measurements": ["borderline heart size"],
  "impression": "No acute pulmonary findings"
}}

Report:
Clear lungs. No pleural effusion. No pneumothorax.
Output:
{{
  "findings": [],
  "locations": ["lungs", "pleura"],
  "measurements": [],
  "impression": "Normal chest radiograph"
}}

Now analyze the following report:

```
{report}
```
"""

    def run(self, state: dict) -> dict:
        raw = state["raw_text"]

        prompt = self.PROMPT_TEMPLATE.format(report=raw[:3000])
        response = self._llm(prompt, max_new_tokens=512, temperature=0.05)
        parsed   = self._parse_json(response)

        if parsed is None:
            self._log_error(state, "Extraction JSON échouée — retour vide sécurisé")
            parsed = {
                "findings": [],
                "locations": [],
                "measurements": [],
                "impression": "",
            }
            state["degraded"] = True

        state["extracted"] = parsed
        return state


# ---------------------------------------------------------------------------
# Agent 3 — Structureur (JSON FHIR-compatible)
# ---------------------------------------------------------------------------

class StructurerAgent(BaseAgent):
    """
    Convertit les entités extraites en une représentation JSON structurée
    inspirée du profil FHIR DiagnosticReport.
    """

    name = "StructurerAgent"

    PROMPT_TEMPLATE = """\
You are a clinical data structuring specialist.

Using the extracted medical entities provided, produce a structured JSON report
compatible with FHIR DiagnosticReport. Include:
- report_id: the UID
- status: final | preliminary | amended
- category: imaging modality (infer from context, default radiology)
- findings: dict mapping anatomical region -> description
- impression: one-sentence radiological conclusion
- anomalies: list of {{ name, severity, location, certainty, icd10_hint }}
  (add your best ICD-10 category hint, e.g. J18 for pneumonia)
- normal_findings: list
- measurements: dict
- recommendations: list of follow-up recommendations (if any)

Report UID: {uid}
Extracted entities:
{entities}

Original report excerpt:
```
{report_excerpt}
```

Return ONLY the JSON object.
"""

    def run(self, state: dict) -> dict:
        extracted = state.get("extracted", {})
        uid       = state["uid"]
        raw       = state["raw_text"][:1500]

        prompt = self.PROMPT_TEMPLATE.format(
            uid=uid,
            entities=json.dumps(extracted, ensure_ascii=False, indent=2),
            report_excerpt=raw,
        )
        response = self._llm(prompt, max_new_tokens=768, temperature=0.05)
        parsed   = self._parse_json(response)

        if parsed is None:
            self._log_error(state, "Structuration JSON échouée — reconstruction depuis entités")
            parsed = self._reconstruct_from_entities(uid, extracted)

        state["structured"] = parsed
        return state

    @staticmethod
    def _reconstruct_from_entities(uid: str, entities: dict) -> dict:
        """Construit un JSON minimal valide à partir des entités brutes."""
        return {
            "report_id":       uid,
            "status":          "preliminary",
            "category":        "radiology",
            "findings":        {"general": " ".join(entities.get("findings", []))},
            "impression":      "",
            "anomalies":       entities.get("anomalies", []),
            "normal_findings": entities.get("normal_findings", []),
            "measurements":    entities.get("measurements", {}),
            "recommendations": [],
        }


# ---------------------------------------------------------------------------
# Agent 4 — Vérificateur clinique
# ---------------------------------------------------------------------------

class VerifierAgent(BaseAgent):
    """
    Vérifie la fidélité clinique du rapport structuré par rapport au texte source.
    Calcule un score de fidélité et détecte les hallucinations ou omissions.
    """

    name = "VerifierAgent"

    PROMPT_TEMPLATE = """\
You are a senior radiologist performing quality control.

Compare the structured report with the original report.
Identify:
1. Missing findings (present in original, absent in structured)
2. Hallucinations (present in structured, absent in original)
3. Severity errors (wrong severity assigned)
4. Overall fidelity score (0.0 to 1.0)

Return a JSON object:
{{
  "fidelity_score": float,
  "missing_findings": [list of strings],
  "hallucinations": [list of strings],
  "severity_errors": [list of strings],
  "pass": bool  (true if fidelity_score >= 0.75)
}}

ORIGINAL REPORT:
```
{original}
```

STRUCTURED REPORT:
{structured}

Return ONLY the JSON object.
"""

    def run(self, state: dict) -> dict:
        original   = state["raw_text"][:2000]
        structured = json.dumps(state.get("structured", {}), ensure_ascii=False, indent=2)

        prompt = self.PROMPT_TEMPLATE.format(
            original=original,
            structured=structured[:1500],
        )
        response = self._llm(prompt, max_new_tokens=384, temperature=0.0)
        parsed   = self._parse_json(response)

        if parsed is None:
            self._log_error(state, "Vérification JSON échouée — score neutre attribué")
            parsed = {
                "fidelity_score": 0.5,
                "missing_findings": [],
                "hallucinations": [],
                "severity_errors": [],
                "pass": False,
            }

        state["verified"] = parsed

        if not parsed.get("pass", True):
            state["degraded"] = True
            logger.warning(
                "[%s] Rapport %s : fidélité %.2f — marqué dégradé.",
                self.name, state["uid"], parsed.get("fidelity_score", 0),
            )

        return state


# ---------------------------------------------------------------------------
# Agent 5 — Synthèse médicale (professionnel de santé)
# ---------------------------------------------------------------------------

class MedicalSummaryAgent(BaseAgent):
    """
    Génère une synthèse concise destinée à un professionnel de santé :
    terminologie clinique standard, structure SOAP partielle, recommandations.
    """

    name = "MedicalSummaryAgent"

    PROMPT_TEMPLATE = """\
You are a senior radiologist writing a concise structured summary for a referring physician.

Write a clinical summary of maximum 120 words containing:
1. KEY FINDINGS (bullet points, clinical terminology)
2. IMPRESSION (one sentence)
3. RECOMMENDATIONS (if any)

Base your summary STRICTLY on the verified structured report below.
Do NOT add information not present in the source.

Structured report:
{structured}

Verification results:
{verified}

Write the summary now (in the same language as the original report if detectable,
otherwise English):
"""

    def run(self, state: dict) -> dict:
        structured = json.dumps(state.get("structured", {}), ensure_ascii=False, indent=2)
        verified   = json.dumps(state.get("verified", {}),   ensure_ascii=False)

        prompt   = self.PROMPT_TEMPLATE.format(structured=structured[:2000], verified=verified)
        response = self._llm(prompt, max_new_tokens=256, temperature=0.15)

        state["medical_summary"] = response.strip()
        return state


# ---------------------------------------------------------------------------
# Agent 6 — Synthèse patient (vulgarisation)
# ---------------------------------------------------------------------------

class PatientSummaryAgent(BaseAgent):
    """
    Génère une explication accessible pour le patient :
    langage simple (niveau lycée), ton rassurant et factuel,
    en français si le rapport source est en français.
    """

    name = "PatientSummaryAgent"

    PROMPT_TEMPLATE = """\
You are a compassionate doctor explaining a radiology result to a patient with no medical background.

Rules:
- Use simple, everyday language (no jargon)
- Maximum 150 words
- Be factual but reassuring; do not alarm unnecessarily
- Explain what was found and what it means in plain terms
- End with one practical next step ("Your doctor will discuss…")
- Write in FRENCH if the detected language is French, otherwise in English

Detected language: {language}
Key findings: {findings}
Impression: {impression}
Anomalies (simplified): {anomalies}

Write the patient explanation now:
"""

    def run(self, state: dict) -> dict:
        structured  = state.get("structured", {})
        preprocessed = state.get("preprocessed", {})

        findings  = structured.get("findings", {})
        impression = structured.get("impression", "")
        anomalies  = [a.get("name", "") for a in structured.get("anomalies", [])]
        language   = preprocessed.get("language", "en")

        prompt   = self.PROMPT_TEMPLATE.format(
            language=language,
            findings=json.dumps(findings, ensure_ascii=False),
            impression=impression,
            anomalies=", ".join(anomalies) if anomalies else "none identified",
        )
        response = self._llm(prompt, max_new_tokens=256, temperature=0.2)

        state["patient_summary"] = response.strip()
        return state
