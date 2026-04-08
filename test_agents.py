"""
Tests unitaires pour les agents du pipeline radiologique.

Vérifie que chaque agent produit le JSON attendu avec les bons schémas.
"""

import unittest
import json
from unittest.mock import Mock, patch
from agents import (
    initial_state,
    PreprocessorAgent,
    ExtractorAgent,
    StructurerAgent,
    VerifierAgent,
    MedicalSummaryAgent,
    PatientSummaryAgent,
)


# ---------------------------------------------------------------------------
# Données de test
# ---------------------------------------------------------------------------

TEST_REPORTS = {
    "GOOD": """
    Chest radiographs dated XXXX. XXXX-year-old male with chest pain.
    The cardiomediastinal silhouette is within normal limits for size and contour.
    The lungs are normally inflated without evidence of focal airspace disease,
    pleural effusion, or pneumothorax. Stable calcified granuloma within the right
    upper lung. No acute bone abnormality. No acute cardiopulmonary process.
    """,
    
    "DEGRADED": """
    FINDINGS: Clear lungs. No acute abnormality.
    """,
    
    "POOR": """
    XXXX XXXX XXXX XXXX
    """,
    
    "NORMAL": """
    Both lungs are clear and expanded. Heart and mediastinum normal.
    No active disease.
    """,
}


class MockModelManager:
    """Mock du ModelManager pour les tests."""
    
    def generate(self, model_key: str, prompt: str, max_new_tokens: int = 512, 
                 temperature: float = 0.05) -> str:
        """Retourne une réponse JSON appropriée selon le contenu du prompt."""
        
        # Agent 1 - Preprocessor
        if "medical NLP preprocessing" in prompt:
            # Vérifier la longueur et qualité du rapport
            if "XXXX XXXX XXXX" in prompt and len(prompt) < 500:
                return '{"language":"en","quality":"poor","completeness":0.0,"sections":[]}'
            elif "Clear lungs" in prompt and "No acute" in prompt:
                return '{"language":"en","quality":"degraded","completeness":0.25,"sections":["FINDINGS"]}'
            elif len(prompt) > 1500:  # Long report with details
                return '{"language":"en","quality":"good","completeness":1.0,"sections":["COMPARISON","INDICATION","FINDINGS","IMPRESSION"]}'
            else:
                return '{"language":"en","quality":"degraded","completeness":0.5,"sections":["FINDINGS"]}'
        
        # Agent 2 - Extractor
        elif "medical information extraction" in prompt:
            if "clear lungs" in prompt.lower() and "no acute" in prompt.lower():
                return '''{
                    "findings": [],
                    "locations": ["lungs", "heart", "mediastinum"],
                    "measurements": [],
                    "impression": "No acute cardiopulmonary findings"
                }'''
            elif "cardiomegaly" in prompt.lower():
                return '''{
                    "findings": ["cardiomegaly"],
                    "locations": ["heart"],
                    "measurements": ["heart size enlarged"],
                    "impression": "Cardiomegaly present"
                }'''
            else:
                return '''{
                    "findings": [],
                    "locations": [],
                    "measurements": [],
                    "impression": ""
                }'''
        
        # Agent 3 - Structurer
        elif "clinical data structuring" in prompt:
            return '''{
                "report_id": "CXR_001",
                "status": "final",
                "category": "radiology",
                "findings": {"lungs": "clear", "heart": "normal"},
                "impression": "No acute findings",
                "anomalies": [],
                "normal_findings": ["clear lungs", "normal heart"],
                "measurements": {},
                "recommendations": []
            }'''
        
        # Agent 4 - Verifier
        elif "senior radiologist" in prompt and "quality control" in prompt:
            return '''{
                "fidelity_score": 0.95,
                "missing_findings": [],
                "hallucinations": [],
                "severity_errors": [],
                "pass": true
            }'''
        
        # Agent 5 - Medical Summary
        elif "referring physician" in prompt:
            return "KEY FINDINGS: No acute cardiopulmonary abnormality. IMPRESSION: Normal chest radiograph. RECOMMENDATIONS: None."
        
        # Agent 6 - Patient Summary
        elif "compassionate doctor" in prompt:
            return "Your chest X-ray looks normal. The lungs and heart are in good condition. No treatment is needed. Your doctor will discuss the results with you."
        
        return "{}"


class TestPreprocessorAgent(unittest.TestCase):
    """Tests du PreprocessorAgent."""
    
    def setUp(self):
        self.mm = MockModelManager()
        self.agent = PreprocessorAgent(self.mm, "test_model")
    
    def test_good_report(self):
        """Test un rapport de bonne qualité."""
        state = initial_state(TEST_REPORTS["GOOD"], uid="CXR_001")
        result = self.agent.run(state)
        
        # Vérifier la structure du résultat
        self.assertIsNotNone(result["preprocessed"])
        self.assertIn("language", result["preprocessed"])
        self.assertIn("quality", result["preprocessed"])
        self.assertIn("completeness", result["preprocessed"])
        self.assertIn("sections", result["preprocessed"])
        
        # Vérifier les valeurs attendues
        self.assertEqual(result["preprocessed"]["quality"], "good")
        self.assertGreater(result["preprocessed"]["completeness"], 0.5)
        self.assertFalse(result["degraded"])
    
    def test_degraded_report(self):
        """Test un rapport dégradé."""
        state = initial_state(TEST_REPORTS["DEGRADED"], uid="CXR_002")
        result = self.agent.run(state)
        
        self.assertEqual(result["preprocessed"]["quality"], "degraded")
        self.assertLess(result["preprocessed"]["completeness"], 0.5)
    
    def test_poor_report(self):
        """Test un rapport de très mauvaise qualité."""
        state = initial_state(TEST_REPORTS["POOR"], uid="CXR_003")
        result = self.agent.run(state)
        
        self.assertEqual(result["preprocessed"]["quality"], "poor")
        self.assertEqual(result["preprocessed"]["completeness"], 0.0)
        self.assertTrue(result["degraded"])
    
    def test_too_short_report(self):
        """Test un rapport trop court."""
        state = initial_state("Hi", uid="CXR_004")
        result = self.agent.run(state)
        
        self.assertEqual(result["preprocessed"]["quality"], "poor")
        self.assertEqual(result["preprocessed"]["sections"], [])
        self.assertTrue(result["degraded"])
    
    def test_json_schema_consistency(self):
        """Test que le schéma JSON est cohérent (pas de sections_found)."""
        state = initial_state(TEST_REPORTS["GOOD"], uid="CXR_005")
        result = self.agent.run(state)
        
        # Vérifier que "sections" existe et "sections_found" n'existe pas
        self.assertIn("sections", result["preprocessed"])
        self.assertNotIn("sections_found", result["preprocessed"])
        self.assertNotIn("issues", result["preprocessed"])


class TestExtractorAgent(unittest.TestCase):
    """Tests du ExtractorAgent."""
    
    def setUp(self):
        self.mm = MockModelManager()
        self.agent = ExtractorAgent(self.mm, "test_model")
    
    def test_extraction_schema(self):
        """Test que le schéma d'extraction est correct."""
        state = initial_state(TEST_REPORTS["NORMAL"], uid="CXR_006")
        result = self.agent.run(state)
        
        extracted = result["extracted"]
        self.assertIn("findings", extracted)
        self.assertIn("locations", extracted)
        self.assertIn("measurements", extracted)
        self.assertIn("impression", extracted)
        
        # Vérifier les types
        self.assertIsInstance(extracted["findings"], list)
        self.assertIsInstance(extracted["locations"], list)
        self.assertIsInstance(extracted["measurements"], list)
        self.assertIsInstance(extracted["impression"], str)
    
    def test_empty_findings(self):
        """Test un rapport sans anomalies."""
        state = initial_state("Clear lungs. No acute abnormality.", uid="CXR_007")
        result = self.agent.run(state)
        
        self.assertEqual(result["extracted"]["findings"], [])
    
    def test_fallback_structure(self):
        """Test que le fallback produit la bonne structure."""
        state = initial_state(TEST_REPORTS["POOR"], uid="CXR_008")
        # Forcer une réponse invalide
        with patch.object(self.agent, '_parse_json', return_value=None):
            result = self.agent.run(state)
        
        extracted = result["extracted"]
        self.assertEqual(extracted["findings"], [])
        self.assertEqual(extracted["locations"], [])
        self.assertEqual(extracted["measurements"], [])
        self.assertEqual(extracted["impression"], "")


class TestStructurerAgent(unittest.TestCase):
    """Tests du StructurerAgent."""
    
    def setUp(self):
        self.mm = MockModelManager()
        self.agent = StructurerAgent(self.mm, "test_model")
    
    def test_structuring_schema(self):
        """Test que le schéma structuré est correct."""
        state = initial_state(TEST_REPORTS["GOOD"], uid="CXR_009")
        # Pré-remplir l'étape d'extraction
        state["extracted"] = {
            "findings": ["cardiomegaly"],
            "locations": ["heart"],
            "measurements": ["enlarged"],
            "impression": "Cardiomegaly"
        }
        
        result = self.agent.run(state)
        structured = result["structured"]
        
        # Vérifier tous les champs obligatoires
        self.assertIn("report_id", structured)
        self.assertIn("status", structured)
        self.assertIn("category", structured)
        self.assertIn("findings", structured)
        self.assertIn("impression", structured)
        self.assertIn("anomalies", structured)
        self.assertIn("normal_findings", structured)
        self.assertIn("measurements", structured)
        self.assertIn("recommendations", structured)
        
        # Vérifier les valeurs
        self.assertEqual(structured["report_id"], "CXR_009")
        self.assertIn(structured["status"], ["final", "preliminary", "amended"])
        self.assertEqual(structured["category"], "radiology")
    
    def test_fhir_compatibility(self):
        """Test la compatibilité FHIR du schéma."""
        state = initial_state(TEST_REPORTS["NORMAL"], uid="CXR_010")
        state["extracted"] = {
            "findings": [],
            "locations": ["lungs"],
            "measurements": [],
            "impression": "Normal"
        }
        
        result = self.agent.run(state)
        structured = result["structured"]
        
        # Les clés principales doivent être des chaînes ou dicts
        self.assertIsInstance(structured["findings"], dict)
        self.assertIsInstance(structured["recommendations"], list)
        self.assertIsInstance(structured["anomalies"], list)


class TestVerifierAgent(unittest.TestCase):
    """Tests du VerifierAgent."""
    
    def setUp(self):
        self.mm = MockModelManager()
        self.agent = VerifierAgent(self.mm, "test_model")
    
    def test_verification_schema(self):
        """Test que le schéma de vérification est correct."""
        state = initial_state(TEST_REPORTS["GOOD"], uid="CXR_011")
        state["structured"] = {
            "report_id": "CXR_011",
            "status": "final",
            "findings": {"lungs": "clear"},
            "impression": "No acute findings",
            "anomalies": [],
            "normal_findings": [],
            "measurements": {},
            "recommendations": []
        }
        
        result = self.agent.run(state)
        verified = result["verified"]
        
        self.assertIn("fidelity_score", verified)
        self.assertIn("missing_findings", verified)
        self.assertIn("hallucinations", verified)
        self.assertIn("severity_errors", verified)
        self.assertIn("pass", verified)
        
        # Vérifier les types
        self.assertIsInstance(verified["fidelity_score"], (int, float))
        self.assertIsInstance(verified["missing_findings"], list)
        self.assertIsInstance(verified["pass"], bool)
        
        # Vérifier la plage de fidelity_score
        self.assertGreaterEqual(verified["fidelity_score"], 0.0)
        self.assertLessEqual(verified["fidelity_score"], 1.0)


class TestMedicalSummaryAgent(unittest.TestCase):
    """Tests du MedicalSummaryAgent."""
    
    def setUp(self):
        self.mm = MockModelManager()
        self.agent = MedicalSummaryAgent(self.mm, "test_model")
    
    def test_summary_generation(self):
        """Test la génération du résumé médical."""
        state = initial_state(TEST_REPORTS["GOOD"], uid="CXR_012")
        state["structured"] = {
            "findings": {"lungs": "clear"},
            "impression": "No acute findings",
            "anomalies": [],
            "recommendations": []
        }
        state["verified"] = {
            "fidelity_score": 0.95,
            "pass": True
        }
        
        result = self.agent.run(state)
        
        # Le résumé doit être une chaîne non vide
        self.assertIsInstance(result["medical_summary"], str)
        self.assertGreater(len(result["medical_summary"]), 0)


class TestPatientSummaryAgent(unittest.TestCase):
    """Tests du PatientSummaryAgent."""
    
    def setUp(self):
        self.mm = MockModelManager()
        self.agent = PatientSummaryAgent(self.mm, "test_model")
    
    def test_patient_summary_generation(self):
        """Test la génération du résumé patient."""
        state = initial_state(TEST_REPORTS["NORMAL"], uid="CXR_013")
        state["preprocessed"] = {"language": "en"}
        state["structured"] = {
            "findings": {"lungs": "clear"},
            "impression": "Normal",
            "anomalies": []
        }
        
        result = self.agent.run(state)
        
        # Le résumé doit être une chaîne non vide
        self.assertIsInstance(result["patient_summary"], str)
        self.assertGreater(len(result["patient_summary"]), 0)


class TestPipelineIntegration(unittest.TestCase):
    """Tests d'intégration du pipeline complet."""
    
    def setUp(self):
        self.mm = MockModelManager()
        self.agents = {
            1: PreprocessorAgent(self.mm, "test_model"),
            2: ExtractorAgent(self.mm, "test_model"),
            3: StructurerAgent(self.mm, "test_model"),
            4: VerifierAgent(self.mm, "test_model"),
            5: MedicalSummaryAgent(self.mm, "test_model"),
            6: PatientSummaryAgent(self.mm, "test_model"),
        }
    
    def test_full_pipeline(self):
        """Test le pipeline complet du début à la fin."""
        state = initial_state(TEST_REPORTS["NORMAL"], uid="CXR_FULL")
        
        # Exécuter tous les agents
        for i in range(1, 7):
            state = self.agents[i].run(state)
        
        # Vérifier que toutes les étapes ont produit du contenu
        self.assertIsNotNone(state["preprocessed"])
        self.assertIsNotNone(state["extracted"])
        self.assertIsNotNone(state["structured"])
        self.assertIsNotNone(state["verified"])
        self.assertIsNotNone(state["medical_summary"])
        self.assertIsNotNone(state["patient_summary"])
        
        # Vérifier la taille des résumés
        self.assertGreater(len(state["medical_summary"]), 0)
        self.assertGreater(len(state["patient_summary"]), 0)


if __name__ == "__main__":
    unittest.main()
