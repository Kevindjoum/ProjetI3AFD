"""
Chargement et génération des rapports radiologiques.

Convertit les fichiers XML (ecgen-radiology/) en CSV (rapports_clean.csv)
avec deux colonnes : id, raw_report
"""

import logging
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterator

import pandas as pd

logger = logging.getLogger(__name__)

# Constantes
XML_DIR = "data/ecgen-radiology"
CSV_PATH = "data/rapports_clean.csv"


# ───────────────────────────────────────────────────────────────
# 1. CONVERTISSEUR XML → CSV
# ───────────────────────────────────────────────────────────────

class XMLToCSVConverter:
    """Convertit les fichiers XML en CSV avec colonnes id, raw_report."""

    def __init__(self, xml_dir: str = XML_DIR):
        self.xml_dir = Path(xml_dir)
        self.valid_count = 0
        self.invalid_count = 0
        self.errors = []

    def load_report(self, xml_file: Path) -> dict | None:
        """
        Parse un fichier XML et extrait id + raw_report.
        
        raw_report = concaténation de tous les textes des balises <AbstractText>
        (sans les labels comme COMPARISON, INDICATION, etc.)
        """
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            # Extraire l'ID
            report_id = xml_file.stem  # ex: "1.xml" → "1"

            # Extraire tous les textes de <AbstractText>
            abstract_texts = []
            for abstract_text in root.findall(".//AbstractText"):
                text = abstract_text.text
                if text and text.strip():
                    abstract_texts.append(text.strip())

            raw_report = " ".join(abstract_texts)

            if not raw_report:
                logger.warning(f"  ⚠ {xml_file.name}: Aucun AbstractText trouvé")
                self.invalid_count += 1
                return None

            self.valid_count += 1
            return {
                "id": report_id,
                "raw_report": raw_report,
            }

        except ET.ParseError as e:
            logger.warning(f"  ✗ {xml_file.name}: XML invalide - {e}")
            self.invalid_count += 1
            self.errors.append((xml_file.name, str(e)))
            return None
        except Exception as e:
            logger.error(f"  ✗ {xml_file.name}: Erreur inattendue - {e}")
            self.invalid_count += 1
            self.errors.append((xml_file.name, str(e)))
            return None

    def convert_all(self, output_path: str = CSV_PATH) -> pd.DataFrame:
        """
        Convertit tous les XML du dossier en DataFrame
        et sauvegarde en CSV.
        """
        logger.info(f"Conversion des XML de {self.xml_dir} → {output_path}")

        xml_files = sorted(self.xml_dir.glob("*.xml"))
        logger.info(f"  Trouvé {len(xml_files)} fichier(s) XML")

        reports = []
        for i, xml_file in enumerate(xml_files):
            report = self.load_report(xml_file)
            if report:
                reports.append(report)

            # Log tous les 100 fichiers
            if (i + 1) % 100 == 0:
                logger.info(f"  Traité {i+1}/{len(xml_files)} fichiers")

        logger.info(f"\n✓ Résultats:")
        logger.info(f"  Valides: {self.valid_count}")
        logger.info(f"  Invalides/corrompus: {self.invalid_count}")

        if self.errors:
            logger.warning(f"\n⚠ Fichiers en erreur (premiers 5):")
            for filename, error in self.errors[:5]:
                logger.warning(f"    - {filename}: {error}")

        # Créer DataFrame
        df = pd.DataFrame(reports)
        logger.info(f"\nDataFrame: {len(df)} rapports")

        # Sauvegarder en CSV
        df.to_csv(output_path, index=False, encoding="utf-8")
        logger.info(f"✓ Sauvegardé: {output_path}")

        return df


# ───────────────────────────────────────────────────────────────
# 2. CHARGEUR CSV
# ───────────────────────────────────────────────────────────────

class ReportLoader:
    """Charge les rapports depuis rapports_clean.csv."""

    def __init__(self, csv_path: str = CSV_PATH):
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV non trouvé: {self.csv_path}")

    def load(self, max_samples: int | None = None) -> pd.DataFrame:
        """Charge le CSV et retourne un DataFrame."""
        df = pd.read_csv(self.csv_path)

        if max_samples:
            df = df.head(max_samples)

        return df.reset_index(drop=True)

    def iter_reports(self, max_samples: int | None = None) -> Iterator[dict]:
        """Itère sur les rapports du CSV."""
        df = self.load(max_samples=max_samples)
        for _, row in df.iterrows():
            yield row.to_dict()

    def get_report(self, report_id: int | str) -> dict | None:
        """Récupère un rapport spécifique par ID."""
        df = self.load()
        match = df[df["id"] == int(report_id)]
        if not match.empty:
            return match.iloc[0].to_dict()
        return None


# ───────────────────────────────────────────────────────────────
# 3. FONCTION PRINCIPALE
# ───────────────────────────────────────────────────────────────

def generate_csv_from_xml(
    xml_dir: str = XML_DIR,
    output: str = CSV_PATH,
) -> pd.DataFrame:
    """
    Crée/met à jour rapports_clean.csv à partir des XML.
    
    Usage:
        df = generate_csv_from_xml()
    """
    converter = XMLToCSVConverter(xml_dir)
    df = converter.convert_all(output)
    return df


# ───────────────────────────────────────────────────────────────
# 4. UTILISATION
# ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Générer le CSV depuis les XML
    df = generate_csv_from_xml()

    # Charger et afficher quelques rapports
    print("\n" + "=" * 70)
    print("Premiers rapports du CSV:")
    print("=" * 70)

    loader = ReportLoader()
    for i, report in enumerate(loader.iter_reports(max_samples=3)):
        print(f"\n[{i+1}] ID: {report['id']}")
        print(f"    Rapport: {report['raw_report'][:150]}...")
