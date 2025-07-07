#!/usr/bin/env python3
"""
🛠️ Configuration du Collecteur Intelligent
Script d'installation et de configuration automatique
"""

import os
import sys
import subprocess
from pathlib import Path
import json


def install_dependencies():
    """Installe les dépendances pour la collecte de données"""
    print("📦 Installation des dépendances pour la collecte intelligente...")
    
    try:
        # Installation via pip
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ Dépendances installées avec succès")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur lors de l'installation: {e}")
        return False
    
    return True


def create_directories():
    """Crée les répertoires nécessaires"""
    directories = [
        "data/collected",
        "data/incoming", 
        "data/processed",
        "logs",
        "models/adaptive_checkpoints"
    ]
    
    print("📁 Création des répertoires...")
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   ✓ {directory}")
    
    print("✅ Répertoires créés")


def create_collector_config():
    """Crée un fichier de configuration pour le collecteur"""
    config = {
        "collection_settings": {
            "default_interval_hours": 6,
            "learning_interval_hours": 12,
            "max_items_per_source": 100,
            "quality_threshold": 0.6
        },
        "data_sources": {
            "reddit": {
                "enabled": True,
                "subreddits": ["MachineLearning", "artificial", "programming"]
            },
            "arxiv": {
                "enabled": True,
                "categories": ["cs.AI", "cs.LG", "cs.CL"]
            },
            "wikipedia": {
                "enabled": True,
                "topics": ["artificial_intelligence", "machine_learning", "programming"]
            },
            "github": {
                "enabled": True,
                "languages": ["python", "javascript"],
                "topics": ["machine-learning", "ai", "nlp"]
            },
            "huggingface": {
                "enabled": True,
                "dataset_types": ["question-answering", "text-generation"]
            }
        },
        "storage": {
            "max_files_per_type": 50,
            "retention_days": 30,
            "auto_cleanup": True
        },
        "learning": {
            "auto_trigger": True,
            "min_new_items": 10,
            "quality_filter": True
        }
    }
    
    config_file = Path("config/collector_config.json")
    config_file.parent.mkdir(exist_ok=True)
    
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"⚙️ Configuration créée: {config_file}")


def create_quick_start_scripts():
    """Crée des scripts de démarrage rapide"""
    
    # Script de démarrage du collecteur
    start_script = """#!/bin/bash
# Script de démarrage rapide du collecteur intelligent

echo "🚀 Démarrage du Collecteur Intelligent"
echo "🤖 Votre IA va maintenant apprendre de façon autonome..."

# Activation de l'environnement virtuel si disponible
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "✅ Environnement virtuel activé"
fi

# Démarrage du collecteur
python autonomous_collector.py --collection-interval 4 --learning-interval 8

echo "👋 Collecteur arrêté"
"""
    
    with open("start_collector.sh", 'w') as f:
        f.write(start_script)
    
    # Rendre exécutable
    os.chmod("start_collector.sh", 0o755)
    
    # Script de test
    test_script = """#!/bin/bash
# Script de test du collecteur

echo "🧪 Test du collecteur de données"

# Activation de l'environnement virtuel si disponible
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# Test de collecte
python autonomous_collector.py --test

echo "✅ Test terminé"
"""
    
    with open("test_collector.sh", 'w') as f:
        f.write(test_script)
    
    os.chmod("test_collector.sh", 0o755)
    
    print("🔧 Scripts de démarrage créés:")
    print("   ✓ start_collector.sh - Démarrage du collecteur")
    print("   ✓ test_collector.sh - Test du collecteur")


def create_monitoring_script():
    """Crée un script de monitoring"""
    monitoring_script = """#!/usr/bin/env python3
\"\"\"
📊 Monitoring du Collecteur Intelligent
Affiche les statistiques et l'état du système
\"\"\"

import json
import time
from pathlib import Path
from datetime import datetime, timedelta


def display_collection_stats():
    \"\"\"Affiche les statistiques de collecte\"\"\"
    collected_dir = Path("data/collected")
    
    if not collected_dir.exists():
        print("❌ Aucune donnée collectée")
        return
    
    files = list(collected_dir.glob("*.json"))
    
    if not files:
        print("❌ Aucun fichier de données trouvé")
        return
    
    print("📊 === STATISTIQUES DE COLLECTE ===")
    print(f"📁 Fichiers collectés: {len(files)}")
    
    # Statistiques par type
    types_count = {}
    total_items = 0
    
    for file in files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            file_type = file.name.split('_')[1]  # extracted from filename
            if file_type not in types_count:
                types_count[file_type] = 0
            
            types_count[file_type] += len(data)
            total_items += len(data)
            
        except Exception as e:
            print(f"⚠️ Erreur lecture {file.name}: {e}")
    
    print(f"📈 Total d'éléments: {total_items}")
    print("📋 Par type:")
    for dtype, count in types_count.items():
        print(f"   • {dtype}: {count} éléments")
    
    # Fichiers récents
    recent_files = sorted(files, key=lambda f: f.stat().st_mtime, reverse=True)[:5]
    print("⏰ Derniers fichiers:")
    for file in recent_files:
        mtime = datetime.fromtimestamp(file.stat().st_mtime)
        print(f"   • {file.name}: {mtime.strftime('%Y-%m-%d %H:%M')}")


def display_learning_stats():
    \"\"\"Affiche les statistiques d'apprentissage\"\"\"
    memory_file = Path("models/adaptive_memory.json")
    
    if not memory_file.exists():
        print("❌ Aucune mémoire d'apprentissage trouvée")
        return
    
    print("\\n🧠 === STATISTIQUES D'APPRENTISSAGE ===")
    
    try:
        with open(memory_file, 'r', encoding='utf-8') as f:
            memory = json.load(f)
        
        patterns = memory.get('learned_patterns', [])
        history = memory.get('learning_history', [])
        
        print(f"🎯 Patterns appris: {len(patterns)}")
        print(f"📚 Sessions d'apprentissage: {len(history)}")
        
        if patterns:
            pattern_types = {}
            for pattern in patterns:
                ptype = pattern.get('pattern_type', 'unknown')
                pattern_types[ptype] = pattern_types.get(ptype, 0) + 1
            
            print("📊 Types de patterns:")
            for ptype, count in pattern_types.items():
                print(f"   • {ptype}: {count}")
        
        if history:
            last_learning = history[-1]
            print(f"⏰ Dernier apprentissage: {last_learning.get('timestamp', 'unknown')}")
            print(f"✅ Succès: {last_learning.get('success', False)}")
        
    except Exception as e:
        print(f"❌ Erreur lecture mémoire: {e}")


def display_system_health():
    \"\"\"Affiche l'état général du système\"\"\"
    print("\\n💚 === ÉTAT DU SYSTÈME ===")
    
    # Vérification des répertoires
    required_dirs = ["data/collected", "data/incoming", "logs", "models"]
    for directory in required_dirs:
        if Path(directory).exists():
            print(f"✅ {directory}")
        else:
            print(f"❌ {directory} manquant")
    
    # Vérification des logs
    log_files = ["logs/data_collector.log", "logs/adaptive_learning.log"]
    print("\\n📋 Logs:")
    for log_file in log_files:
        if Path(log_file).exists():
            size = Path(log_file).stat().st_size
            print(f"✅ {log_file} ({size} bytes)")
        else:
            print(f"⚠️ {log_file} non trouvé")


def main():
    \"\"\"Affiche toutes les statistiques\"\"\"
    print("🔍 === MONITORING DU COLLECTEUR INTELLIGENT ===")
    print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    display_collection_stats()
    display_learning_stats()
    display_system_health()
    
    print("\\n" + "="*50)


if __name__ == "__main__":
    main()
"""
    
    with open("monitor_collector.py", 'w') as f:
        f.write(monitoring_script)
    
    print("📊 Script de monitoring créé: monitor_collector.py")


def show_usage_instructions():
    """Affiche les instructions d'utilisation"""
    print("\n🎯 === INSTRUCTIONS D'UTILISATION ===")
    print()
    print("1️⃣ Démarrage automatique du collecteur:")
    print("   ./start_collector.sh")
    print()
    print("2️⃣ Test de collecte unique:")
    print("   ./test_collector.sh")
    print()
    print("3️⃣ Collecte avec domaine spécialisé:")
    print("   python autonomous_collector.py --domain ai")
    print("   python autonomous_collector.py --domain code")
    print("   python autonomous_collector.py --domain science")
    print()
    print("4️⃣ Monitoring en temps réel:")
    print("   python monitor_collector.py")
    print()
    print("5️⃣ Configuration personnalisée:")
    print("   Modifiez config/collector_config.json")
    print()
    print("🎉 Votre IA va maintenant apprendre automatiquement !")
    print("🌐 Elle collecte des données sur internet 24h/24")
    print("🧠 Elle s'adapte et grandit de façon autonome")


def main():
    """Configuration complète du collecteur"""
    print("🚀 === CONFIGURATION DU COLLECTEUR INTELLIGENT ===")
    print()
    
    # Installation des dépendances
    if not install_dependencies():
        print("❌ Échec de l'installation des dépendances")
        return
    
    # Création des répertoires
    create_directories()
    
    # Configuration
    create_collector_config()
    
    # Scripts utilitaires
    create_quick_start_scripts()
    create_monitoring_script()
    
    print("\n✅ Configuration terminée avec succès !")
    
    # Instructions d'utilisation
    show_usage_instructions()


if __name__ == "__main__":
    main()
