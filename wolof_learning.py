#!/usr/bin/env python3
"""
Script pour démontrer l'apprentissage et l'inférence en wolof.
Ce script sert de démonstration pratique de l'utilisation du système
d'apprentissage adaptatif avec la langue wolof.
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Ajouter le dossier src au path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.adaptive_learner import AdaptiveLearner
import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def train_with_wolof_data(data_path=None, model_name="TheBloke/Mistral-7B-v0.1-GGUF"):
    """Entraîne le modèle avec des données en wolof"""
    
    # Utiliser les données d'exemple si aucun chemin n'est fourni
    if data_path is None:
        data_path = "data/samples/wolof/wolof_conversations.json"
        if not os.path.exists(data_path):
            data_path = "data/samples/wolof/wolof_texte.txt"
            if not os.path.exists(data_path):
                raise FileNotFoundError("Aucune donnée wolof trouvée. Veuillez créer des exemples avec scripts/collect_wolof_data.py")
    
    # Vérifier que le fichier existe
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Fichier non trouvé: {data_path}")
    
    logger.info(f"🌍 Apprentissage avec les données wolof: {data_path}")
    
    # Initialiser le système d'apprentissage adaptatif
    learner = AdaptiveLearner(base_model=model_name)
    
    # Observer et apprendre des données
    results = learner.observe_and_learn(data_path)
    
    # Afficher les résultats
    logger.info("✅ Apprentissage terminé")
    logger.info(f"Résultats: {results}")
    
    return learner

def interactive_wolof_testing(learner=None):
    """Test interactif du modèle en wolof"""
    if learner is None:
        logger.info("⚠️ Aucun modèle entraîné fourni, initialisation d'un nouveau modèle...")
        learner = AdaptiveLearner()
        logger.info("⚠️ Ce modèle n'a pas été entraîné sur des données wolof. Les résultats peuvent être limités.")
    
    logger.info("🎮 Mode interactif de test en wolof")
    logger.info("Tapez 'exit' pour quitter")
    
    while True:
        user_input = input("\n🗣️ Vous (en wolof): ")
        if user_input.lower() == 'exit':
            break
        
        try:
            # Générer une réponse
            response = learner.generate_response(user_input)
            print(f"🤖 Modèle: {response}")
        except Exception as e:
            logger.error(f"Erreur lors de la génération: {e}")

def main():
    parser = argparse.ArgumentParser(description='Apprentissage et test en wolof')
    parser.add_argument('--data', help='Chemin vers les données wolof')
    parser.add_argument('--model', default="TheBloke/Mistral-7B-v0.1-GGUF", 
                       help='Modèle de base à utiliser')
    parser.add_argument('--interactive', action='store_true', 
                       help='Mode interactif pour tester le modèle')
    parser.add_argument('--train-only', action='store_true',
                       help='Entraîner uniquement, sans mode interactif')
    
    args = parser.parse_args()
    
    try:
        if args.interactive and not args.train_only:
            # Mode interactif sans entraînement
            interactive_wolof_testing()
        else:
            # Entraîner le modèle
            learner = train_with_wolof_data(args.data, args.model)
            
            # Mode interactif après entraînement
            if not args.train_only:
                interactive_wolof_testing(learner)
    
    except Exception as e:
        logger.error(f"Erreur: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
