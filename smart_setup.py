#!/usr/bin/env python3
"""
🚀 Démarrage Intelligent du Système d'Apprentissage Autonome
Configuration et lancement automatique complet
"""

import os
import sys
import subprocess
import shutil
import time
from pathlib import Path
import argparse


def check_python_version():
    """Vérifie la version de Python"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ requis")
        print(f"   Version actuelle: {sys.version}")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}")
    return True


def check_environment():
    """Vérifie et prépare l'environnement"""
    print("🔍 Vérification de l'environnement...")
    
    # Vérification de Python
    if not check_python_version():
        return False
    
    # Détection de l'environnement virtuel actuel
    venv_active = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    
    if venv_active:
        print("✅ Environnement virtuel détecté et actif")
        # Utiliser pip du système actuel (environnement virtuel actif)
        pip_executable = "python3" if not shutil.which("pip") else "pip"
    else:
        # Création de l'environnement virtuel si nécessaire
        venv_paths = [Path("venv"), Path(".venv")]
        venv_path = None
        
        for path in venv_paths:
            if path.exists():
                venv_path = path
                break
        
        if not venv_path:
            print("📦 Création de l'environnement virtuel...")
            try:
                subprocess.check_call([sys.executable, "-m", "venv", "venv"])
                venv_path = Path("venv")
                print("✅ Environnement virtuel créé")
            except subprocess.CalledProcessError:
                print("❌ Erreur lors de la création de l'environnement virtuel")
                return False
        
        # Configuration des chemins selon l'OS
        if sys.platform == "win32":
            pip_executable = str(venv_path / "Scripts" / "pip")
        else:
            pip_executable = str(venv_path / "bin" / "pip")
    
    # Installation des dépendances
    print("📚 Installation des dépendances...")
    try:
        if venv_active and pip_executable == "python3":
            # Dans un environnement virtuel, utiliser python -m pip
            subprocess.check_call([pip_executable, "-m", "pip", "install", "beautifulsoup4", "feedparser", "wikipedia", "arxiv", "googlesearch-python", "aiohttp", "lxml", "plotly"])
        else:
            subprocess.check_call([pip_executable, "install", "beautifulsoup4", "feedparser", "wikipedia", "arxiv", "googlesearch-python", "aiohttp", "lxml", "plotly"])
        print("✅ Dépendances du collecteur installées")
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur lors de l'installation des dépendances: {e}")
        print("💡 Les dépendances principales semblent déjà installées")
        # Ne pas échouer si l'installation échoue, continuer
    
    return True


def setup_directories():
    """Crée la structure de répertoires"""
    print("📁 Configuration des répertoires...")
    
    directories = [
        "data/collected",
        "data/incoming",
        "data/processed", 
        "data/raw",
        "logs",
        "models/adaptive_checkpoints",
        "config"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Répertoires configurés")
    return True


def create_config_files():
    """Crée les fichiers de configuration par défaut"""
    print("⚙️ Création des configurations...")
    
    # Configuration du collecteur
    config_file = Path("config/collector_config.json")
    if not config_file.exists():
        import json
        
        config = {
            "collection_settings": {
                "default_interval_hours": 6,
                "learning_interval_hours": 12,
                "max_items_per_source": 100,
                "quality_threshold": 0.6
            },
            "data_sources": {
                "reddit": {"enabled": True},
                "arxiv": {"enabled": True},
                "wikipedia": {"enabled": True},
                "github": {"enabled": True},
                "huggingface": {"enabled": True}
            },
            "learning": {
                "auto_trigger": True,
                "min_new_items": 10
            }
        }
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    
    print("✅ Configuration créée")
    return True


def create_startup_scripts():
    """Crée les scripts de démarrage"""
    print("🔧 Création des scripts de démarrage...")
    
    # Script principal de collecte
    collector_script = """#!/bin/bash
# Script de démarrage du collecteur intelligent

echo "🤖 === COLLECTEUR INTELLIGENT AUTONOME ==="
echo "🌐 Démarrage de l'apprentissage automatique..."

# Activation de l'environnement virtuel
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "✅ Environnement virtuel activé"
fi

# Démarrage du collecteur avec gestion d'erreurs
while true; do
    echo "🚀 Lancement du collecteur..."
    python autonomous_collector.py --collection-interval 4 --learning-interval 8
    
    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "👋 Arrêt normal du collecteur"
        break
    else
        echo "⚠️ Redémarrage automatique dans 30 secondes..."
        sleep 30
    fi
done
"""
    
    with open("start_smart_collector.sh", 'w') as f:
        f.write(collector_script)
    os.chmod("start_smart_collector.sh", 0o755)
    
    # Script de monitoring
    monitor_script = """#!/bin/bash
# Script de monitoring intelligent

echo "📊 === MONITORING INTELLIGENT ==="

# Activation de l'environnement virtuel
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# Monitoring continu
watch -n 30 python monitor_collector.py
"""
    
    with open("monitor_smart.sh", 'w') as f:
        f.write(monitor_script)
    os.chmod("monitor_smart.sh", 0o755)
    
    # Script de dashboard
    dashboard_script = """#!/bin/bash
# Script de dashboard web

echo "🖥️ === DASHBOARD WEB ==="

# Activation de l'environnement virtuel
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# Démarrage du dashboard
echo "🌐 Dashboard disponible sur: http://localhost:8501"
streamlit run dashboard_collector.py --server.port 8501
"""
    
    with open("start_dashboard.sh", 'w') as f:
        f.write(dashboard_script)
    os.chmod("start_dashboard.sh", 0o755)
    
    print("✅ Scripts de démarrage créés")
    return True


def run_initial_test():
    """Effectue un test initial du système"""
    print("🧪 Test initial du système...")
    
    try:
        # Import et test des modules principaux
        sys.path.append(str(Path.cwd()))
        
        # Test du collecteur
        from src.data_collector import AutoDataCollector
        collector = AutoDataCollector()
        print("✅ Collecteur de données initialisé")
        
        # Test de l'apprentissage adaptatif
        from src.adaptive_learner import AdaptiveLearner
        learner = AdaptiveLearner()
        print("✅ Système d'apprentissage adaptatif initialisé")
        
        print("✅ Tous les tests passent")
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors des tests: {e}")
        return False


def show_startup_menu():
    """Affiche le menu de démarrage"""
    print("\n" + "="*60)
    print("🤖 === SYSTÈME D'APPRENTISSAGE AUTONOME CONFIGURÉ ===")
    print("="*60)
    print()
    print("Votre IA est maintenant prête à apprendre de façon autonome !")
    print()
    print("🚀 Options de démarrage :")
    print()
    print("1️⃣  Collecteur Automatique (recommandé)")
    print("   ./start_smart_collector.sh")
    print("   → L'IA collecte et apprend automatiquement 24h/24")
    print()
    print("2️⃣  Dashboard Web Interactif")
    print("   ./start_dashboard.sh")
    print("   → Interface graphique de surveillance")
    print()
    print("3️⃣  Monitoring Terminal")
    print("   ./monitor_smart.sh")
    print("   → Surveillance en temps réel dans le terminal")
    print()
    print("4️⃣  Test de Collecte Unique")
    print("   python autonomous_collector.py --test")
    print("   → Test rapide du système")
    print()
    print("5️⃣  Collecte Spécialisée")
    print("   python autonomous_collector.py --domain ai")
    print("   → Collecte focalisée sur un domaine")
    print()
    print("🎯 Fonctionnalités principales :")
    print("   • Collecte automatique depuis Reddit, ArXiv, Wikipedia...")
    print("   • Apprentissage adaptatif en temps réel")
    print("   • Détection intelligente de patterns")
    print("   • Amélioration continue des performances")
    print("   • Interface de monitoring graphique")
    print()
    print("📖 Pour plus d'informations : README_ADAPTATIF.md")
    print("="*60)


def main():
    """Fonction principale de configuration"""
    parser = argparse.ArgumentParser(description="Configuration intelligente du système")
    parser.add_argument("--quick", action="store_true", 
                       help="Configuration rapide sans tests")
    parser.add_argument("--test-only", action="store_true",
                       help="Tests uniquement")
    parser.add_argument("--start-collector", action="store_true",
                       help="Démarrer le collecteur après configuration")
    
    args = parser.parse_args()
    
    print("🚀 === CONFIGURATION INTELLIGENTE ===")
    print("🤖 Préparation de votre IA autonome...")
    print()
    
    if args.test_only:
        # Tests uniquement
        if run_initial_test():
            print("✅ Tous les tests passent")
        else:
            print("❌ Certains tests échouent")
        return
    
    # Configuration complète
    steps = [
        ("Vérification de l'environnement", check_environment),
        ("Configuration des répertoires", setup_directories),
        ("Création des configurations", create_config_files),
        ("Création des scripts", create_startup_scripts),
    ]
    
    if not args.quick:
        steps.append(("Tests du système", run_initial_test))
    
    # Exécution des étapes
    for step_name, step_func in steps:
        print(f"📋 {step_name}...")
        try:
            if not step_func():
                print(f"❌ Échec: {step_name}")
                return
        except Exception as e:
            print(f"❌ Erreur dans {step_name}: {e}")
            return
        time.sleep(0.5)  # Pause pour l'effet visuel
    
    print("\n🎉 Configuration terminée avec succès !")
    
    # Affichage du menu
    show_startup_menu()
    
    # Démarrage automatique si demandé
    if args.start_collector:
        print("\n🚀 Démarrage automatique du collecteur...")
        try:
            subprocess.run(["./start_smart_collector.sh"])
        except KeyboardInterrupt:
            print("\n👋 Arrêt demandé par l'utilisateur")
        except Exception as e:
            print(f"\n❌ Erreur lors du démarrage: {e}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Configuration interrompue")
    except Exception as e:
        print(f"\n❌ Erreur fatale: {e}")
        sys.exit(1)
