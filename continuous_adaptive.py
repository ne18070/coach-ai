#!/usr/bin/env python3
"""
Système de surveillance et d'apprentissage continu.
Surveille automatiquement les nouveaux fichiers et s'adapte en temps réel.
"""

import os
import sys
import time
import json
import threading
from pathlib import Path
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Ajouter le dossier src au path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.adaptive_learner import AdaptiveLearner
import logging


class AdaptiveLearningHandler(FileSystemEventHandler):
    """Gestionnaire d'événements pour l'apprentissage adaptatif"""
    
    def __init__(self, learner: AdaptiveLearner, processed_dir: Path):
        self.learner = learner
        self.processed_dir = processed_dir
        self.processed_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Compteurs pour les statistiques
        self.stats = {
            'files_processed': 0,
            'successful_adaptations': 0,
            'failed_adaptations': 0,
            'patterns_learned': [],
            'start_time': datetime.now()
        }
    
    def on_created(self, event):
        """Appelé quand un nouveau fichier est créé"""
        if not event.is_directory:
            file_path = Path(event.src_path)
            
            # Filtrer les fichiers supportés
            if file_path.suffix.lower() in ['.json', '.txt', '.csv']:
                self.logger.info(f"📥 Nouveau fichier détecté: {file_path.name}")
                
                # Attendre un peu pour s'assurer que le fichier est complètement écrit
                time.sleep(1)
                
                # Traiter le fichier
                self._process_file(file_path)
    
    def _process_file(self, file_path: Path):
        """Traite un nouveau fichier et s'adapte"""
        try:
            self.logger.info(f"🔍 Traitement de: {file_path.name}")
            
            # Observer et apprendre
            results = self.learner.observe_and_learn(str(file_path))
            
            # Analyser les résultats
            successful = 0
            failed = 0
            patterns = []
            
            for pattern_name, result in results.items():
                if result['status'] == 'success':
                    successful += 1
                    patterns.append(result['pattern_type'])
                    self.logger.info(f"✅ {pattern_name}: {result['pattern_type']} ({result['num_examples']} exemples)")
                else:
                    failed += 1
                    self.logger.error(f"❌ {pattern_name}: {result.get('error', 'Erreur inconnue')}")
            
            # Mettre à jour les statistiques
            self.stats['files_processed'] += 1
            self.stats['successful_adaptations'] += successful
            self.stats['failed_adaptations'] += failed
            self.stats['patterns_learned'].extend(patterns)
            
            # Archiver le fichier traité
            archived_file = self.processed_dir / f"{file_path.stem}_{int(time.time())}{file_path.suffix}"
            file_path.rename(archived_file)
            
            self.logger.info(f"📂 Fichier archivé: {archived_file.name}")
            
            # Afficher les statistiques mises à jour
            self._print_stats()
            
            # Test optionnel de génération
            if successful > 0:
                self._test_generation()
        
        except Exception as e:
            self.logger.error(f"❌ Erreur lors du traitement de {file_path}: {e}")
            self.stats['failed_adaptations'] += 1
    
    def _print_stats(self):
        """Affiche les statistiques actuelles"""
        runtime = datetime.now() - self.stats['start_time']
        
        print(f"\n📊 Statistiques d'apprentissage adaptatif")
        print(f"⏰ Temps d'exécution: {runtime}")
        print(f"📄 Fichiers traités: {self.stats['files_processed']}")
        print(f"✅ Adaptations réussies: {self.stats['successful_adaptations']}")
        print(f"❌ Adaptations échouées: {self.stats['failed_adaptations']}")
        print(f"🧠 Patterns appris: {set(self.stats['patterns_learned'])}")
        print("-" * 50)
    
    def _test_generation(self):
        """Test rapide de génération"""
        test_prompts = [
            "L'intelligence artificielle",
            "Comment",
            "def"
        ]
        
        for prompt in test_prompts:
            try:
                response = self.learner.generate_response(prompt, max_length=30)
                if response and response.strip():
                    self.logger.info(f"🧪 Test: '{prompt}' → '{response[:30]}...'")
                    break
            except:
                continue


class ContinuousAdaptiveLearner:
    """Système d'apprentissage adaptatif continu"""
    
    def __init__(self, watch_directory: str = "data/incoming", model_name: str = "distilgpt2"):
        self.watch_directory = Path(watch_directory)
        self.processed_directory = Path("data/processed")
        self.model_name = model_name
        
        # Créer les dossiers nécessaires
        self.watch_directory.mkdir(parents=True, exist_ok=True)
        self.processed_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialiser le learner
        self.learner = AdaptiveLearner(
            base_model=model_name,
            max_seq_length=128,
            learning_rate=5e-4
        )
        
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
        
        # Créer le gestionnaire d'événements
        self.handler = AdaptiveLearningHandler(self.learner, self.processed_directory)
        
        # Observer de fichiers
        self.observer = Observer()
        self.observer.schedule(self.handler, str(self.watch_directory), recursive=False)
    
    def start(self):
        """Démarre la surveillance continue"""
        self.logger.info("🚀 Démarrage du système d'apprentissage adaptatif continu")
        self.logger.info(f"👀 Surveillance du dossier: {self.watch_directory}")
        self.logger.info(f"🤖 Modèle utilisé: {self.model_name}")
        
        # Créer un fichier de démonstration
        self._create_initial_demo()
        
        # Démarrer l'observer
        self.observer.start()
        
        print(f"\n🎯 Système actif ! Déposez des fichiers JSON, TXT ou CSV dans: {self.watch_directory}")
        print("Le système s'adaptera automatiquement aux nouveaux données.")
        print("Appuyez sur Ctrl+C pour arrêter.\n")
        
        try:
            # Interface simple pour ajouter des données manuellement
            self._interactive_mode()
        except KeyboardInterrupt:
            self.logger.info("🛑 Arrêt demandé par l'utilisateur")
        finally:
            self.observer.stop()
            self.observer.join()
            self.logger.info("✅ Système d'apprentissage adaptatif arrêté")
    
    def _create_initial_demo(self):
        """Crée un fichier de démonstration initial"""
        demo_file = self.watch_directory / "demo_initial.json"
        
        if not demo_file.exists():
            demo_data = [
                {
                    "input": "Qu'est-ce que l'apprentissage adaptatif ?",
                    "output": "L'apprentissage adaptatif est la capacité d'un système à s'ajuster automatiquement à de nouveaux types de données et patterns."
                }
            ]
            
            with open(demo_file, 'w', encoding='utf-8') as f:
                json.dump(demo_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"📄 Fichier de démonstration créé: {demo_file}")
    
    def _interactive_mode(self):
        """Mode interactif pour ajouter des données"""
        
        def input_thread():
            while True:
                try:
                    command = input().strip().lower()
                    
                    if command in ['quit', 'exit', 'q']:
                        raise KeyboardInterrupt
                    elif command == 'stats':
                        self.handler._print_stats()
                    elif command == 'test':
                        self._create_test_data()
                    elif command.startswith('add '):
                        # Ajouter des données rapidement
                        content = command[4:]
                        self._quick_add_data(content)
                    elif command == 'help':
                        self._print_help()
                        
                except EOFError:
                    break
                except KeyboardInterrupt:
                    raise
        
        # Afficher l'aide au démarrage
        self._print_help()
        
        # Démarrer le thread d'input
        input_thread_obj = threading.Thread(target=input_thread, daemon=True)
        input_thread_obj.start()
        
        # Boucle principale
        while True:
            time.sleep(1)
    
    def _print_help(self):
        """Affiche l'aide"""
        print("\n🎮 Commandes disponibles:")
        print("  stats    - Afficher les statistiques")
        print("  test     - Créer des données de test")
        print("  add <texte> - Ajouter rapidement du texte")
        print("  help     - Afficher cette aide")
        print("  quit     - Quitter le système")
        print()
    
    def _create_test_data(self):
        """Crée des données de test aléatoirement"""
        test_data = [
            {
                "instruction": "Explique l'apprentissage automatique",
                "response": "L'apprentissage automatique permet aux machines d'apprendre des patterns dans les données."
            },
            {
                "question": "Qu'est-ce qu'un neurone artificiel ?",
                "answer": "Un neurone artificiel est une unité de calcul qui traite les informations dans un réseau de neurones."
            }
        ]
        
        test_file = self.watch_directory / f"test_{int(time.time())}.json"
        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, indent=2, ensure_ascii=False)
        
        print(f"📄 Données de test créées: {test_file.name}")
    
    def _quick_add_data(self, content: str):
        """Ajoute rapidement du contenu"""
        quick_data = [content]
        
        quick_file = self.watch_directory / f"quick_{int(time.time())}.json"
        with open(quick_file, 'w', encoding='utf-8') as f:
            json.dump(quick_data, f, indent=2, ensure_ascii=False)
        
        print(f"📄 Contenu ajouté: {quick_file.name}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Système d'apprentissage adaptatif continu")
    parser.add_argument("--watch-dir", default="data/incoming", help="Dossier à surveiller")
    parser.add_argument("--model", default="distilgpt2", help="Modèle de base à utiliser")
    
    args = parser.parse_args()
    
    # Créer et démarrer le système
    system = ContinuousAdaptiveLearner(
        watch_directory=args.watch_dir,
        model_name=args.model
    )
    
    system.start()
