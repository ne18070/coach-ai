#!/usr/bin/env python3
"""
Script de démarrage rapide pour le fine-tuning
"""
import sys
import os
from pathlib import Path

# Ajouter le répertoire parent au PYTHONPATH
sys.path.append(str(Path(__file__).parent.parent))

from src.utils import load_config, setup_logging, print_training_summary
from src.trainer import LLMTrainer

def main():
    print("🚀 Démarrage rapide du fine-tuning LLM")
    print("=" * 50)
    
    # Configuration par défaut
    project_root = Path(__file__).parent.parent
    config_path = project_root / "config" / "training_config.yaml"
    
    if not config_path.exists():
        print(f"❌ Fichier de configuration non trouvé: {config_path}")
        return 1
    
    try:
        # Charger la configuration
        config = load_config(str(config_path))
        
        # Setup logging
        logger = setup_logging()
        
        # Créer le trainer
        trainer = LLMTrainer(config)
        
        # Lancer l'entraînement
        logger.info("Lancement de l'entraînement")
        trainer.train()
        
        logger.info("Entraînement terminé avec succès!")
        
    except Exception as e:
        print(f"❌ Erreur durante l'entraînement: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
