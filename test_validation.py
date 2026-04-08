"""
Tests rapides sans dépendance Ollama - Validation de schémas et imports
"""

import unittest
import json
from pathlib import Path

from agents import initial_state


class TestBasicValidation(unittest.TestCase):
    """Tests rapides de validation sans appel Ollama."""
    
    def test_initial_state_creation(self):
        """Tester la création de l'état initial."""
        state = initial_state("Test report", uid="TEST_001")
        
        # Vérifier les clés obligatoires
        required_keys = ["uid", "raw_text", "degraded"]
        for key in required_keys:
            self.assertIn(key, state, f"Clé manquante: {key}")
        
        self.assertEqual(state["uid"], "TEST_001")
        self.assertEqual(state["raw_text"], "Test report")
        self.assertFalse(state["degraded"])
    
    def test_state_with_long_report(self):
        """Tester avec un rapport long."""
        long_report = "This is a test. " * 500
        state = initial_state(long_report, uid="LONG_001")
        
        self.assertEqual(len(state["raw_text"]), len(long_report))
    
    def test_state_with_special_chars(self):
        """Tester avec caractères spéciaux."""
        special_report = "Rapport: é à ù €™ 日本語 العربية"
        state = initial_state(special_report, uid="SPECIAL_001")
        
        self.assertEqual(state["raw_text"], special_report)
    
    def test_json_serializability(self):
        """Tester que l'état peut être sérialisé en JSON."""
        state = initial_state("Test", uid="JSON_001")
        
        # Doit être sérialisable
        json_str = json.dumps(state, ensure_ascii=False, default=str)
        self.assertIsInstance(json_str, str)
        
        # Doit être resérialisable
        reloaded = json.loads(json_str)
        self.assertEqual(reloaded["uid"], "JSON_001")


class TestDataIntegrity(unittest.TestCase):
    """Tester l'intégrité des données."""
    
    def test_csv_file_exists(self):
        """Vérifier que le CSV existe."""
        csv_path = Path("data/rapports_clean.csv")
        self.assertTrue(csv_path.exists())
    
    def test_csv_is_readable(self):
        """Vérifier que le CSV est lisible."""
        import pandas as pd
        csv_path = Path("data/rapports_clean.csv")
        
        df = pd.read_csv(csv_path)
        self.assertGreater(len(df), 0)
        self.assertIn("id", df.columns)
        self.assertIn("raw_report", df.columns)
    
    def test_csv_sample_reports(self):
        """Vérifier quelques rapports du CSV."""
        import pandas as pd
        csv_path = Path("data/rapports_clean.csv")
        
        df = pd.read_csv(csv_path)
        for idx, row in df.head(3).iterrows():
            report_id = row["id"]
            raw_text = row["raw_report"]
            
            # Les IDs doivent être non-vides
            self.assertTrue(str(report_id).strip())
            # Les textes doivent être non-vides
            self.assertGreater(len(str(raw_text).strip()), 0)


class TestFileStructure(unittest.TestCase):
    """Tester la structure des fichiers."""
    
    def test_required_files_exist(self):
        """Vérifier que tous les fichiers requis existent."""
        required_files = [
            "agents.py",
            "model_manager.py",
            "loader.py",
            "metrics.py",
            "pipeline.py",
            "app.py",
            "test_agents.py",
        ]
        
        for filename in required_files:
            filepath = Path(filename)
            self.assertTrue(filepath.exists(), f"Fichier manquant: {filename}")
    
    def test_required_directories_exist(self):
        """Vérifier que tous les répertoires requis existent."""
        required_dirs = [
            "data",
            "data/reports",
            "data/exports",
        ]
        
        for dirname in required_dirs:
            dirpath = Path(dirname)
            self.assertTrue(dirpath.exists(), f"Répertoire manquant: {dirname}")


class TestCodeQuality(unittest.TestCase):
    """Tests de qualité du code."""
    
    def test_agents_file_not_empty(self):
        """Vérifier que agents.py n'est pas vide."""
        with open("agents.py", "r") as f:
            content = f.read()
            self.assertGreater(len(content), 100)
            self.assertIn("class PreprocessorAgent", content)
            self.assertIn("class ExtractorAgent", content)
    
    def test_model_manager_file_not_empty(self):
        """Vérifier que model_manager.py n'est pas vide."""
        with open("model_manager.py", "r") as f:
            content = f.read()
            self.assertGreater(len(content), 100)
            self.assertIn("class ModelManager", content)
    
    def test_python_syntax_valid(self):
        """Vérifier la syntaxe Python de tous les fichiers."""
        import py_compile
        
        py_files = [
            "agents.py",
            "model_manager.py",
            "loader.py",
            "app.py",
        ]
        
        for py_file in py_files:
            try:
                py_compile.compile(py_file, doraise=True)
            except py_compile.PyCompileError as e:
                self.fail(f"Erreur de syntaxe dans {py_file}: {e}")


class TestImportsSafety(unittest.TestCase):
    """Tester les imports de façon sécurisée."""
    
    def test_basic_imports(self):
        """Tester les imports de base."""
        try:
            import json
            import csv
            import pathlib
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Import standard échoué: {e}")
    
    def test_optional_imports(self):
        """Tester les imports optionnels."""
        try:
            import pandas
            import requests
            self.assertTrue(True)
        except ImportError:
            # C'est OK si pandas ou requests ne sont pas disponibles
            pass


class TestDocumentation(unittest.TestCase):
    """Vérifier que la documentation est présente."""
    
    def test_readme_exists(self):
        """Vérifier que le README existe."""
        readme = Path("README.md")
        self.assertTrue(readme.exists())
    
    def test_readme_not_empty(self):
        """Vérifier que le README n'est pas vide."""
        with open("README.md", "r") as f:
            content = f.read()
            self.assertGreater(len(content), 100)
    
    def test_requirements_exists(self):
        """Vérifier que requirements.txt existe."""
        req = Path("requirements.txt")
        self.assertTrue(req.exists())
    
    def test_requirements_has_dependencies(self):
        """Vérifier que requirements.txt a des dépendances."""
        with open("requirements.txt", "r") as f:
            lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
            self.assertGreater(len(lines), 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
