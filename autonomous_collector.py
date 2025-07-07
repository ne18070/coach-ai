#!/usr/bin/env python3
"""
🌟 Collecteur Intelligent Autonome
Lance la collecte automatique de données sur internet
et nourrit l'apprentissage adaptatif en temps réel
"""

import asyncio
import argparse
import signal
import sys
import time
from pathlib import Path
from typing import Optional

from src.data_collector import AutoDataCollector, IntelligentDataCollector, DataSource
from src.adaptive_learner import AdaptiveLearner


class AutonomousLearningSystem:
    """
    Système d'apprentissage autonome qui collecte des données
    sur internet et s'adapte automatiquement
    """
    
    def __init__(self, storage_dir: str = "data/collected"):
        self.collector = AutoDataCollector(storage_dir)
        self.learner = AdaptiveLearner()
        self.running = False
        
    async def start_autonomous_learning(self, 
                                       collection_interval: int = 6,
                                       learning_interval: int = 12):
        """
        Lance l'apprentissage autonome :
        - Collecte des données sur internet
        - Apprentissage adaptatif automatique
        """
        
        print("🚀 === SYSTÈME D'APPRENTISSAGE AUTONOME ===")
        print("🤖 L'IA va maintenant apprendre de façon autonome...")
        print("🌐 Collecte de données depuis internet...")
        print("🧠 Adaptation automatique du modèle...")
        print()
        
        self.running = True
        
        # Gestion de l'arrêt propre
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Collecte initiale
        await self._initial_data_collection()
        
        # Boucle principal d'apprentissage autonome
        last_learning = time.time()
        
        while self.running:
            try:
                current_time = time.time()
                
                # Collecte de nouvelles données
                print(f"🌐 Collecte de données - {time.strftime('%H:%M:%S')}")
                data = await self.collector.collector.collect_all_sources()
                
                if data:
                    # Sauvegarde des données collectées
                    self.collector.collector.save_collected_data(data)
                    print(f"📥 {len(data)} nouveaux éléments collectés")
                    
                    # Déclenchement de l'apprentissage si interval écoulé
                    if current_time - last_learning >= learning_interval * 3600:
                        await self._trigger_adaptive_learning()
                        last_learning = current_time
                    
                    # Affichage des statistiques
                    await self._display_progress_stats()
                
                # Attente avant la prochaine collecte
                await asyncio.sleep(collection_interval * 3600)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"❌ Erreur dans le cycle d'apprentissage: {e}")
                await asyncio.sleep(300)  # Attente 5 min avant retry
        
        print("🛑 Arrêt du système d'apprentissage autonome")
    
    async def _initial_data_collection(self):
        """Collecte initiale de données pour démarrer l'apprentissage"""
        print("🎯 Collecte initiale de données de qualité...")
        
        # Collecte depuis les meilleures sources
        data = await self.collector.collector.collect_all_sources()
        
        if data:
            self.collector.collector.save_collected_data(data)
            print(f"✅ Collecte initiale: {len(data)} éléments")
            
            # Apprentissage initial
            await self._trigger_adaptive_learning()
        else:
            print("⚠️ Aucune donnée collectée lors de l'initialisation")
    
    async def _trigger_adaptive_learning(self):
        """Déclenche l'apprentissage adaptatif sur les nouvelles données"""
        print("🧠 Déclenchement de l'apprentissage adaptatif...")
        
        collected_dir = Path("data/collected")
        if not collected_dir.exists():
            return
            
        # Recherche des fichiers de données récents
        json_files = list(collected_dir.glob("*.json"))
        
        if json_files:
            # Tri par date de modification (plus récents en premier)
            json_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            
            # Apprentissage sur les fichiers les plus récents
            for data_file in json_files[:3]:  # 3 fichiers les plus récents
                try:
                    print(f"📚 Apprentissage depuis: {data_file.name}")
                    result = self.learner.observe_and_learn(str(data_file))
                    
                    if result:
                        print(f"✅ Apprentissage réussi: {result.get('patterns_learned', 0)} patterns")
                    
                except Exception as e:
                    print(f"❌ Erreur d'apprentissage sur {data_file.name}: {e}")
        
        print("🎯 Cycle d'apprentissage terminé")
    
    async def _display_progress_stats(self):
        """Affiche les statistiques de progression"""
        stats = self.collector.collector.get_collection_stats()
        
        print("📊 === STATISTIQUES D'APPRENTISSAGE ===")
        print(f"🌐 Sources actives: {stats['active_sources']}/{stats['total_sources']}")
        print(f"📈 Types de données: {', '.join(stats['sources_by_type'].keys())}")
        if stats['last_collection']:
            print(f"⏰ Dernière collecte: {stats['last_collection'].strftime('%H:%M:%S')}")
        
        # Statistiques d'apprentissage
        memory_file = Path("models/adaptive_memory.json")
        if memory_file.exists():
            import json
            with open(memory_file, 'r', encoding='utf-8') as f:
                memory = json.load(f)
                print(f"🧠 Patterns appris: {len(memory.get('learned_patterns', []))}")
                print(f"🎯 Expériences: {len(memory.get('learning_history', []))}")
        
        print("=" * 45)
        print()
    
    def _signal_handler(self, signum, frame):
        """Gestion de l'arrêt propre"""
        print(f"\n🛑 Signal reçu ({signum}), arrêt en cours...")
        self.running = False
    
    def add_specialized_sources(self, domain: str):
        """Ajoute des sources spécialisées selon le domaine"""
        domain_sources = {
            'ai': [
                DataSource(
                    name="OpenAI_Blog",
                    url="https://openai.com/blog/rss.xml",
                    data_type="text",
                    extraction_method="rss",
                    update_frequency=24
                ),
                DataSource(
                    name="DeepMind_Publications",
                    url="https://deepmind.com/research",
                    data_type="instruction",
                    extraction_method="scraping",
                    update_frequency=48
                )
            ],
            'code': [
                DataSource(
                    name="Python_Tutorials",
                    url="https://realpython.com/atom.xml",
                    data_type="code",
                    extraction_method="rss",
                    update_frequency=12
                ),
                DataSource(
                    name="GitHub_Trending",
                    url="https://github.com/trending/python",
                    data_type="code",
                    extraction_method="scraping",
                    update_frequency=6
                )
            ],
            'science': [
                DataSource(
                    name="Nature_AI",
                    url="https://www.nature.com/subjects/machine-learning.rss",
                    data_type="text",
                    extraction_method="rss",
                    update_frequency=24
                )
            ]
        }
        
        if domain in domain_sources:
            for source in domain_sources[domain]:
                self.collector.collector.add_custom_source(source)
                print(f"➕ Source spécialisée ajoutée: {source.name}")


async def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(description="Collecteur Intelligent Autonome")
    parser.add_argument("--collection-interval", type=int, default=6,
                       help="Intervalle de collecte en heures (défaut: 6)")
    parser.add_argument("--learning-interval", type=int, default=12,
                       help="Intervalle d'apprentissage en heures (défaut: 12)")
    parser.add_argument("--domain", type=str, choices=['ai', 'code', 'science'],
                       help="Ajouter des sources spécialisées pour un domaine")
    parser.add_argument("--test", action="store_true",
                       help="Mode test - collecte unique")
    parser.add_argument("--stats", action="store_true",
                       help="Afficher les statistiques uniquement")
    
    args = parser.parse_args()
    
    # Création du système
    system = AutonomousLearningSystem()
    
    # Ajout de sources spécialisées si demandé
    if args.domain:
        system.add_specialized_sources(args.domain)
    
    if args.test:
        # Mode test - collecte unique
        print("🧪 Mode test - collecte unique")
        data = await system.collector.collector.collect_all_sources()
        system.collector.collector.save_collected_data(data)
        print(f"✅ Test terminé: {len(data)} éléments collectés")
        
    elif args.stats:
        # Affichage des statistiques uniquement
        await system._display_progress_stats()
        
    else:
        # Mode apprentissage autonome
        await system.start_autonomous_learning(
            collection_interval=args.collection_interval,
            learning_interval=args.learning_interval
        )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Au revoir !")
    except Exception as e:
        print(f"❌ Erreur fatale: {e}")
        sys.exit(1)
