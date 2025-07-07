#!/usr/bin/env python3
"""
Script principal pour lancer l'entraînement de fine-tuning
"""
import argparse
import sys
import os
from pathlib import Path

# Ajouter le répertoire src au PYTHONPATH
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils import load_config, set_seed, setup_logging
from src.trainer import LLMTrainer

def main():
    parser = argparse.ArgumentParser(description="Fine-tuning de LLM")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Chemin vers le fichier de configuration YAML"
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        help="Chemin vers un checkpoint pour reprendre l'entraînement"
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Effectuer seulement l'évaluation (pas d'entraînement)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Répertoire de sortie (override la config)"
    )
    
    args = parser.parse_args()
    
    # Charger la configuration
    print(f"Chargement de la configuration depuis {args.config}")
    config = load_config(args.config)
    
    # Override le répertoire de sortie si spécifié
    if args.output_dir:
        config['training']['output_dir'] = args.output_dir
    
    # Configurer le logging
    log_dir = config['training'].get('logging_dir', 'logs')
    logger = setup_logging(log_dir)
    
    # Fixer la graine aléatoire
    seed = config.get('system', {}).get('seed', 42)
    set_seed(seed)
    logger.info(f"Graine aléatoire fixée à {seed}")
    
    # Créer le trainer
    trainer = LLMTrainer(config)
    
    try:
        if args.eval_only:
            # Mode évaluation seulement
            logger.info("Mode évaluation seulement")
            results = trainer.evaluate()
            print("Résultats d'évaluation:")
            for metric, value in results.items():
                print(f"  {metric}: {value}")
        else:
            # Mode entraînement
            logger.info("Début de l'entraînement")
            trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
            
            # Évaluation finale si dataset de validation disponible
            if 'validation_file' in config['data']:
                logger.info("Évaluation finale")
                results = trainer.evaluate()
                print("Résultats d'évaluation finale:")
                for metric, value in results.items():
                    print(f"  {metric}: {value}")
    
    except KeyboardInterrupt:
        logger.info("Entraînement interrompu par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Erreur durante l'entraînement: {e}")
        raise
    
    logger.info("Script terminé avec succès")

if __name__ == "__main__":
    main()
