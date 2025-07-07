#!/usr/bin/env python3
"""
Script de préparation des données pour le fine-tuning
"""
import sys
import os
from pathlib import Path

# Ajouter le répertoire parent au PYTHONPATH
sys.path.append(str(Path(__file__).parent.parent))

from src.data_processor import DataProcessor
from src.utils import load_config, setup_logging

def main():
    # Configuration
    input_file = "../data/raw/sample_conversations.json"
    output_dir = "../data/processed"
    
    # Setup logging
    logger = setup_logging()
    
    # Initialiser le processeur de données
    processor = DataProcessor()
    
    try:
        # Préparer les données
        logger.info("Début de la préparation des données")
        dataset_dict = processor.prepare_dataset(
            input_file=input_file,
            output_dir=output_dir,
            data_type="conversational"
        )
        
        # Afficher les statistiques
        stats = processor.get_dataset_stats(dataset_dict)
        logger.info("Statistiques finales:")
        for split_name, split_stats in stats.items():
            logger.info(f"  {split_name}: {split_stats['num_examples']} exemples")
            if 'text_stats' in split_stats:
                text_stats = split_stats['text_stats']
                logger.info(f"    Longueur moyenne: {text_stats['avg_length']:.1f} mots")
                logger.info(f"    Min/Max: {text_stats['min_length']}/{text_stats['max_length']} mots")
        
        logger.info("Préparation des données terminée avec succès")
        
    except Exception as e:
        logger.error(f"Erreur durante la préparation: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
