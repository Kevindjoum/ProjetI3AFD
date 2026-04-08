"""
Tests d'intégration et validation globale du pipeline RadioAI Pro
"""

import unittest
import json
import time
import psutil
import os
from pathlib import Path

from agents import (
    initial_state,
    PreprocessorAgent,
    ExtractorAgent,
    StructurerAgent,
    VerifierAgent,
    MedicalSummaryAgent,
    PatientSummaryAgent,
)
from model_manager import ModelManager
from loader import XMLToCSVConverter


# ============================================================================
# Données de test réalistes
# ============================================================================

REAL_REPORTS = {
    "NORMAL": """
    CLINICAL HISTORY: Chest pain.
    
    COMPARISON: None.
    
    FINDINGS: The heart is normal in size and contour. The mediastinal contour 
    is within normal limits. The lungs are clear bilaterally with no focal 
    consolidation, pleural effusion, or pneumothorax. The osseous structures 
    are intact. No acute abnormality.
    
    IMPRESSION: No acute cardiopulmonary process.
    """,
    
    "PATHOLOGICAL": """
    CLINICAL HISTORY: Shortness of breath.
    
    COMPARISON: Previous chest radiograph from XXXX shows progression of findings.
    
    FINDINGS: There is a moderate bilateral pleural effusion, right greater 
    than left. The cardiac silhouette is enlarged. There are small bilateral 
    opacities in the lung bases consistent with atelectasis. The trachea is 
    midline. The costophrenic angles are blunted bilaterally.
    
    IMPRESSION: Cardiomegaly with moderate bilateral pleural effusions and 
    basilar opacities.
    """,
    
    "DEGRADED": """FINDINGS: No findings noted.""",
}


class TestImportsAndDependencies(unittest.TestCase):
    """Tester que tous les modules s'importent correctement."""
    
    def test_import_agents(self):
        """Test import des agents."""
        from agents import (
            PreprocessorAgent, ExtractorAgent, StructurerAgent,
            VerifierAgent, MedicalSummaryAgent, PatientSummaryAgent
        )
        self.assertIsNotNone(PreprocessorAgent)
    
    def test_import_model_manager(self):
        """Test import du ModelManager."""
        from model_manager import ModelManager
        self.assertIsNotNone(ModelManager)
    
    def test_import_loader(self):
        """Test import du loader."""
        from loader import XMLToCSVConverter
        self.assertIsNotNone(XMLToCSVConverter)
    
    def test_data_files_exist(self):
        """Vérifier que les fichiers de données existent."""
        self.assertTrue(Path("data/rapports_clean.csv").exists())
    
    def test_model_manager_initialization(self):
        """Tester l'initialisation du ModelManager."""
        mm = ModelManager()
        self.assertIsNotNone(mm)


class TestPipelineIntegration(unittest.TestCase):
    """Tester le pipeline complet avec des rapports réalistes."""
    
    @classmethod
    def setUpClass(cls):
        """Initialiser le ModelManager pour tous les tests."""
        cls.mm = ModelManager()
        cls.agents = {
            "preprocessor": PreprocessorAgent(cls.mm, "biomistral"),
            "extractor": ExtractorAgent(cls.mm, "biomistral"),
            "structurer": StructurerAgent(cls.mm, "biomistral"),
            "verifier": VerifierAgent(cls.mm, "biomistral"),
            "medical_summary": MedicalSummaryAgent(cls.mm, "mistral-instruct"),
            "patient_summary": PatientSummaryAgent(cls.mm, "mistral-instruct"),
        }
    
    def test_full_pipeline_normal_report(self):
        """Tester le pipeline complet avec un rapport normal."""
        state = initial_state(REAL_REPORTS["NORMAL"], uid="TEST_NORMAL_001")
        
        # Exécuter tous les agents
        state = self.agents["preprocessor"].run(state)
        state = self.agents["extractor"].run(state)
        state = self.agents["structurer"].run(state)
        state = self.agents["verifier"].run(state)
        state = self.agents["medical_summary"].run(state)
        state = self.agents["patient_summary"].run(state)
        
        # Vérifier que toutes les étapes sont complètes
        self.assertIn("preprocessed", state)
        self.assertIn("extracted", state)
        self.assertIn("structured", state)
        self.assertIn("verified", state)
        self.assertIn("medical_summary", state)
        self.assertIn("patient_summary", state)
        
        # Vérifier les résultats
        self.assertEqual(state["uid"], "TEST_NORMAL_001")
        self.assertIsNotNone(state["preprocessed"]["quality"])
        self.assertGreaterEqual(state["verified"]["fidelity_score"], 0.0)
        self.assertLessEqual(state["verified"]["fidelity_score"], 1.0)
    
    def test_pipeline_pathological_report(self):
        """Tester avec un rapport pathologique."""
        state = initial_state(REAL_REPORTS["PATHOLOGICAL"], uid="TEST_PATHO_001")
        
        state = self.agents["preprocessor"].run(state)
        state = self.agents["extractor"].run(state)
        state = self.agents["structurer"].run(state)
        state = self.agents["verifier"].run(state)
        
        # Pour un rapport pathologique, des anomalies doivent être détectées
        extracted = state["extracted"]
        self.assertIsInstance(extracted["findings"], list)
        self.assertIsInstance(extracted["locations"], list)
    
    def test_pipeline_degraded_report(self):
        """Tester avec un rapport dégradé."""
        state = initial_state(REAL_REPORTS["DEGRADED"], uid="TEST_DEGRAD_001")
        
        state = self.agents["preprocessor"].run(state)
        
        # Un rapport dégradé doit être marqué comme tel
        self.assertIn("degraded", state)


class TestJSONSchemaValidation(unittest.TestCase):
    """Valider que tous les outputs JSON respectent les schémas."""
    
    @classmethod
    def setUpClass(cls):
        cls.mm = ModelManager()
    
    def test_preprocessor_schema(self):
        """Valider le schéma du preprocessor."""
        agent = PreprocessorAgent(self.mm, "biomistral")
        state = initial_state(REAL_REPORTS["NORMAL"], uid="TEST_001")
        result = agent.run(state)
        
        pre = result["preprocessed"]
        self.assertIn("language", pre)
        self.assertIn("quality", pre)
        self.assertIn("completeness", pre)
        self.assertIn("sections", pre)
        
        self.assertIsInstance(pre["language"], str)
        self.assertIsInstance(pre["quality"], str)
        self.assertIsInstance(pre["completeness"], (int, float))
        self.assertIsInstance(pre["sections"], list)
        
        self.assertIn(pre["quality"], ["good", "degraded", "poor"])
    
    def test_extractor_schema(self):
        """Valider le schéma de l'extractor."""
        agent = ExtractorAgent(self.mm, "biomistral")
        state = initial_state(REAL_REPORTS["NORMAL"], uid="TEST_002")
        result = agent.run(state)
        
        ext = result["extracted"]
        self.assertIn("findings", ext)
        self.assertIn("locations", ext)
        self.assertIn("measurements", ext)
        self.assertIn("impression", ext)
        
        self.assertIsInstance(ext["findings"], list)
        self.assertIsInstance(ext["locations"], list)
        self.assertIsInstance(ext["measurements"], list)
        self.assertIsInstance(ext["impression"], str)
    
    def test_structurer_schema(self):
        """Valider le schéma du structurer."""
        agent = StructurerAgent(self.mm, "biomistral")
        state = initial_state(REAL_REPORTS["NORMAL"], uid="TEST_003")
        state["extracted"] = {
            "findings": ["test"],
            "locations": ["test"],
            "measurements": [],
            "impression": "test"
        }
        result = agent.run(state)
        
        struct = result["structured"]
        required_fields = ["report_id", "status", "category", "findings", 
                          "impression", "anomalies", "normal_findings", 
                          "measurements", "recommendations"]
        for field in required_fields:
            self.assertIn(field, struct)
        
        self.assertEqual(struct["report_id"], "TEST_003")
        self.assertIn(struct["status"], ["final", "preliminary", "amended"])
        self.assertEqual(struct["category"], "radiology")
    
    def test_verifier_schema(self):
        """Valider le schéma du vérificateur."""
        agent = VerifierAgent(self.mm, "biomistral")
        state = initial_state(REAL_REPORTS["NORMAL"], uid="TEST_004")
        state["structured"] = {
            "report_id": "TEST_004",
            "status": "final",
            "findings": {},
            "impression": "test",
            "anomalies": [],
            "recommendations": []
        }
        result = agent.run(state)
        
        ver = result["verified"]
        self.assertIn("fidelity_score", ver)
        self.assertIn("missing_findings", ver)
        self.assertIn("hallucinations", ver)
        self.assertIn("severity_errors", ver)
        self.assertIn("pass", ver)
        
        self.assertIsInstance(ver["fidelity_score"], (int, float))
        self.assertIsInstance(ver["missing_findings"], list)
        self.assertIsInstance(ver["pass"], bool)


class TestPerformanceAndMemory(unittest.TestCase):
    """Mesurer les performances et l'utilisation mémoire."""
    
    @classmethod
    def setUpClass(cls):
        cls.mm = ModelManager()
        cls.process = psutil.Process(os.getpid())
    
    def test_memory_usage(self):
        """Tester l'utilisation mémoire."""
        initial_mem = self.process.memory_info().rss / 1024 / 1024  # MB
        
        agent = PreprocessorAgent(self.mm, "biomistral")
        state = initial_state(REAL_REPORTS["NORMAL"], uid="MEM_TEST")
        result = agent.run(state)
        
        final_mem = self.process.memory_info().rss / 1024 / 1024  # MB
        mem_increase = final_mem - initial_mem
        
        # Vérifier que l'augmentation est raisonnable (< 500 MB pour un agent)
        self.assertLess(mem_increase, 500, f"Mémoire augmentée de {mem_increase:.1f} MB")
    
    def test_execution_time(self):
        """Tester le temps d'exécution d'un agent."""
        agent = PreprocessorAgent(self.mm, "biomistral")
        state = initial_state(REAL_REPORTS["NORMAL"], uid="PERF_TEST")
        
        start = time.time()
        result = agent.run(state)
        elapsed = time.time() - start
        
        # Vérifier que l'exécution est raisonnable (< 30 secondes par agent)
        self.assertLess(elapsed, 30, f"Exécution prise {elapsed:.1f} secondes")


class TestErrorHandling(unittest.TestCase):
    """Tester la gestion des erreurs et cas limite."""
    
    @classmethod
    def setUpClass(cls):
        cls.mm = ModelManager()
    
    def test_empty_report(self):
        """Tester avec un rapport vide."""
        agent = PreprocessorAgent(self.mm, "biomistral")
        state = initial_state("", uid="EMPTY_TEST")
        result = agent.run(state)
        
        self.assertTrue(result.get("degraded", False))
        self.assertEqual(result["preprocessed"]["quality"], "poor")
    
    def test_very_long_report(self):
        """Tester avec un rapport très long."""
        long_text = "This is a test report. " * 1000
        agent = PreprocessorAgent(self.mm, "biomistral")
        state = initial_state(long_text, uid="LONG_TEST")
        
        # Ne doit pas lever d'exception
        result = agent.run(state)
        self.assertIn("preprocessed", result)
    
    def test_special_characters(self):
        """Tester avec des caractères spéciaux."""
        special_text = "Rapport avec caractères: é à ù €™ 日本語 العربية"
        agent = PreprocessorAgent(self.mm, "biomistral")
        state = initial_state(special_text, uid="SPECIAL_TEST")
        
        # Ne doit pas lever d'exception
        result = agent.run(state)
        self.assertIn("preprocessed", result)


class TestDataLoading(unittest.TestCase):
    """Tester le chargement des données."""
    
    def test_csv_loading(self):
        """Tester le chargement du CSV."""
        csv_path = Path("data/rapports_clean.csv")
        self.assertTrue(csv_path.exists(), "Le CSV n'existe pas")
        
        # Charger et vérifier le CSV
        import pandas as pd
        df = pd.read_csv(csv_path)
        self.assertGreater(len(df), 0, "Le CSV est vide")
        self.assertIn("id", df.columns)
        self.assertIn("raw_text", df.columns)
        
        # Vérifier que les rapports ne sont pas vides
        for idx, row in df.head(5).iterrows():
            self.assertGreater(len(str(row["raw_text"])), 10)


class TestExportFormats(unittest.TestCase):
    """Tester les formats d'export."""
    
    @classmethod
    def setUpClass(cls):
        cls.mm = ModelManager()
    
    def test_json_export(self):
        """Tester l'export JSON."""
        state = initial_state(REAL_REPORTS["NORMAL"], uid="EXPORT_JSON")
        
        # Vérifier que le state est sérialisable en JSON
        json_str = json.dumps(state, ensure_ascii=False, indent=2, default=str)
        self.assertIsInstance(json_str, str)
        
        # Vérifier qu'on peut le relire
        reloaded = json.loads(json_str)
        self.assertEqual(reloaded["uid"], "EXPORT_JSON")


if __name__ == "__main__":
    # Exécuter les tests avec plus de détails
    unittest.main(verbosity=2)
