#!/usr/bin/env python3
"""
Script d'apprentissage continu automatique.
Lance un système qui observe et apprend automatiquement de toutes nouvelles données.
"""

import os
import sys
import argparse
import signal
import time
from pathlib import Path

# Ajouter le dossier src au path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.adaptive_learner import AdaptiveLearner
import logging


class ContinuousLearningSystem:
    """Système d'apprentissage continu qui surveille et s'adapte automatiquement"""
    
    def __init__(self, 
                 model_name: str = "microsoft/DialoGPT-small",
                 watch_directory: str = "data/incoming",
                 check_interval: int = 30):
        
        self.learner = AdaptiveLearner(base_model=model_name)
        self.watch_directory = Path(watch_directory)
        self.check_interval = check_interval
        self.running = True
        
        # Créer le dossier de surveillance s'il n'existe pas
        self.watch_directory.mkdir(parents=True, exist_ok=True)
        
        # Configuration du logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/continuous_learning.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Gestionnaire pour arrêt propre
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Gestionnaire pour arrêt propre du système"""
        self.logger.info("🛑 Arrêt du système d'apprentissage continu...")
        self.running = False
    
    def start_learning(self):
        """Démarre le système d'apprentissage continu"""
        self.logger.info("🚀 Démarrage du système d'apprentissage continu")
        self.logger.info(f"📁 Surveillance du dossier: {self.watch_directory}")
        self.logger.info(f"⏰ Intervalle de vérification: {self.check_interval}s")
        
        # Créer un fichier de démonstration si le dossier est vide
        self._create_demo_file()
        
        processed_files = set()
        
        while self.running:
            try:
                # Rechercher de nouveaux fichiers
                current_files = set()
                for pattern in ['*.json', '*.txt', '*.csv']:
                    current_files.update(self.watch_directory.glob(pattern))
                
                new_files = current_files - processed_files
                
                if new_files:
                    self.logger.info(f"📥 {len(new_files)} nouveaux fichiers détectés")
                    
                    for file_path in new_files:
                        self._process_new_file(file_path)
                        processed_files.add(file_path)
                
                # Attendre avant la prochaine vérification
                time.sleep(self.check_interval)
                
            except Exception as e:
                self.logger.error(f"❌ Erreur dans la boucle principale: {e}")
                time.sleep(self.check_interval)
        
        self.logger.info("✅ Système d'apprentissage continu arrêté")
    
    def _process_new_file(self, file_path: Path):
        """Traite un nouveau fichier détecté"""
        try:
            self.logger.info(f"🔍 Traitement du fichier: {file_path.name}")
            
            # Observer et apprendre du nouveau fichier
            result = self.learner.observe_and_learn(str(file_path))
            
            # Afficher les résultats
            for pattern_name, pattern_result in result.items():
                if pattern_result['status'] == 'success':
                    self.logger.info(f"✅ {pattern_name}: Apprentissage réussi ({pattern_result['num_examples']} exemples)")
                else:
                    self.logger.error(f"❌ {pattern_name}: Échec - {pattern_result.get('error', 'Erreur inconnue')}")
            
            # Déplacer le fichier traité vers un dossier d'archives
            archive_dir = self.watch_directory / "processed"
            archive_dir.mkdir(exist_ok=True)
            
            archived_file = archive_dir / f"{file_path.stem}_{int(time.time())}{file_path.suffix}"
            file_path.rename(archived_file)
            
            self.logger.info(f"📂 Fichier archivé: {archived_file.name}")
            
        except Exception as e:
            self.logger.error(f"❌ Erreur lors du traitement de {file_path}: {e}")
    
    def _create_demo_file(self):
        """Crée un fichier de démonstration pour montrer le fonctionnement"""
        demo_file = self.watch_directory / "demo_learning.json"
        
        if not demo_file.exists():
            demo_data = [
                {
                    "input": "Comment fonctionne l'apprentissage automatique ?",
                    "output": "L'apprentissage automatique permet aux machines d'apprendre des patterns dans les données sans être explicitement programmées pour chaque tâche spécifique."
                },
                {
                    "input": "Qu'est-ce que l'apprentissage adaptatif ?",
                    "output": "L'apprentissage adaptatif est la capacité d'un système à modifier son comportement en temps réel en fonction de nouvelles informations ou expériences."
                }
            ]
            
            import json
            with open(demo_file, 'w', encoding='utf-8') as f:
                json.dump(demo_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"📄 Fichier de démonstration créé: {demo_file}")
    
    def test_learning(self, input_text: str):
        """Teste le système d'apprentissage avec une entrée"""
        try:
            response = self.learner.generate_response(input_text)
            print(f"Question: {input_text}")
            print(f"Réponse: {response}")
            return response
        except Exception as e:
            print(f"Erreur lors du test: {e}")
            return None


def main():
    parser = argparse.ArgumentParser(description="Système d'apprentissage continu automatique")
    parser.add_argument('--model', default='microsoft/DialoGPT-small', 
                       help='Modèle de base à utiliser')
    parser.add_argument('--watch-dir', default='data/incoming',
                       help='Dossier à surveiller pour les nouvelles données')
    parser.add_argument('--interval', type=int, default=30,
                       help='Intervalle de vérification en secondes')
    parser.add_argument('--test', type=str,
                       help='Tester le système avec une question')
    
    args = parser.parse_args()
    
    # Créer le système d'apprentissage
    system = ContinuousLearningSystem(
        model_name=args.model,
        watch_directory=args.watch_dir,
        check_interval=args.interval
    )
    
    if args.test:
        # Mode test
        print("🧪 Mode test activé")
        system.test_learning(args.test)
    else:
        # Mode apprentissage continu
        print("🤖 Démarrage du système d'apprentissage continu...")
        print("Déposez des fichiers JSON, TXT ou CSV dans le dossier de surveillance.")
        print("Le système s'adaptera automatiquement aux nouveaux données.")
        print("Appuyez sur Ctrl+C pour arrêter.")
        
        try:
            system.start_learning()
        except KeyboardInterrupt:
            print("\n🛑 Arrêt demandé par l'utilisateur")


if __name__ == "__main__":
    main()
