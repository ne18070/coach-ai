#!/usr/bin/env python3
"""
Script de nettoyage pour le système d'apprentissage adaptatif.
Permet de supprimer les fichiers non essentiels et les données temporaires.
"""

import os
import shutil
import argparse
from pathlib import Path
from datetime import datetime, timedelta


def cleanup_logs(logs_dir="logs", days_old=7):
    """Nettoie les fichiers de logs plus anciens qu'un certain nombre de jours"""
    logs_path = Path(logs_dir)
    if not logs_path.exists():
        print(f"📁 Dossier {logs_dir} n'existe pas, rien à nettoyer.")
        return 0
    
    count = 0
    cutoff_date = datetime.now() - timedelta(days=days_old)
    
    for log_file in logs_path.glob("*.log"):
        file_time = datetime.fromtimestamp(log_file.stat().st_mtime)
        if file_time < cutoff_date:
            log_file.unlink()
            count += 1
    
    print(f"🧹 {count} anciens fichiers de logs supprimés.")
    return count


def cleanup_processed_data(processed_dir="data/processed", keep_count=10):
    """Ne conserve que les N derniers fichiers traités par type"""
    processed_path = Path(processed_dir)
    if not processed_path.exists():
        print(f"📁 Dossier {processed_dir} n'existe pas, rien à nettoyer.")
        return 0
    
    # Grouper les fichiers par extension
    file_groups = {}
    for file_path in processed_path.iterdir():
        if file_path.is_file():
            extension = file_path.suffix
            if extension not in file_groups:
                file_groups[extension] = []
            file_groups[extension].append(file_path)
    
    # Trier chaque groupe par date de modification et supprimer les plus anciens
    count = 0
    for extension, files in file_groups.items():
        files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        if len(files) > keep_count:
            for old_file in files[keep_count:]:
                old_file.unlink()
                count += 1
    
    print(f"🧹 {count} anciens fichiers de données traités supprimés.")
    return count


def cleanup_demo_files(demo_dir="data/demo"):
    """Nettoie les fichiers de démo générés automatiquement"""
    demo_path = Path(demo_dir)
    if not demo_path.exists():
        print(f"📁 Dossier {demo_dir} n'existe pas, rien à nettoyer.")
        return 0
    
    # Conserver une sauvegarde
    backup_dir = demo_path.parent / "demo_backup"
    backup_dir.mkdir(exist_ok=True)
    
    count = 0
    for demo_file in demo_path.iterdir():
        if demo_file.is_file():
            # Copier dans la sauvegarde avant suppression
            shutil.copy2(demo_file, backup_dir / demo_file.name)
            demo_file.unlink()
            count += 1
    
    print(f"🧹 {count} fichiers de démo nettoyés (sauvegardés dans {backup_dir}).")
    return count


def remove_redundant_files(confirm=False):
    """Supprime les fichiers redondants du projet"""
    redundant_files = [
        "continuous_learning.py",  # Remplacé par continuous_adaptive.py
        "__pycache__",             # Fichiers compilés temporaires
    ]
    
    count = 0
    for file_name in redundant_files:
        file_path = Path(file_name)
        if file_path.exists():
            if file_path.is_dir():
                if confirm:
                    shutil.rmtree(file_path)
                    count += 1
                    print(f"🗑️ Dossier supprimé: {file_path}")
                else:
                    print(f"🔍 Dossier redondant trouvé: {file_path} (utilisez --confirm pour supprimer)")
            else:
                if confirm:
                    file_path.unlink()
                    count += 1
                    print(f"🗑️ Fichier supprimé: {file_path}")
                else:
                    print(f"🔍 Fichier redondant trouvé: {file_path} (utilisez --confirm pour supprimer)")
    
    if count > 0:
        print(f"🧹 {count} fichiers/dossiers redondants supprimés.")
    return count


def cleanup_checkpoints(checkpoints_dir="models/adaptive_checkpoints", keep_count=3):
    """Ne conserve que les N derniers checkpoints par type de pattern"""
    checkpoints_path = Path(checkpoints_dir)
    if not checkpoints_path.exists():
        print(f"📁 Dossier {checkpoints_dir} n'existe pas, rien à nettoyer.")
        return 0
    
    # Grouper les checkpoints par pattern
    pattern_groups = {}
    for checkpoint_dir in checkpoints_path.iterdir():
        if checkpoint_dir.is_dir():
            pattern_type = checkpoint_dir.name.split('_')[0]
            if pattern_type not in pattern_groups:
                pattern_groups[pattern_type] = []
            pattern_groups[pattern_type].append(checkpoint_dir)
    
    # Trier chaque groupe par date de modification et supprimer les plus anciens
    count = 0
    for pattern_type, directories in pattern_groups.items():
        directories.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        if len(directories) > keep_count:
            for old_dir in directories[keep_count:]:
                shutil.rmtree(old_dir)
                count += 1
    
    print(f"🧹 {count} anciens checkpoints supprimés.")
    return count


def main():
    parser = argparse.ArgumentParser(description="Nettoie les fichiers non essentiels du système adaptatif")
    parser.add_argument("--logs", action="store_true", help="Nettoyer les anciens fichiers de logs")
    parser.add_argument("--processed", action="store_true", help="Nettoyer les fichiers traités")
    parser.add_argument("--demo", action="store_true", help="Nettoyer les fichiers de démo")
    parser.add_argument("--checkpoints", action="store_true", help="Nettoyer les checkpoints anciens")
    parser.add_argument("--redundant", action="store_true", help="Identifier les fichiers redondants")
    parser.add_argument("--confirm", action="store_true", help="Confirmer la suppression des fichiers redondants")
    parser.add_argument("--all", action="store_true", help="Nettoyer tous les types de fichiers")
    
    args = parser.parse_args()
    
    total_cleaned = 0
    
    # Si aucun argument spécifique n'est fourni, afficher l'aide
    if not any([args.logs, args.processed, args.demo, args.checkpoints, args.redundant, args.all]):
        parser.print_help()
        return
    
    # Nettoyer selon les options
    if args.logs or args.all:
        total_cleaned += cleanup_logs()
    
    if args.processed or args.all:
        total_cleaned += cleanup_processed_data()
    
    if args.demo or args.all:
        total_cleaned += cleanup_demo_files()
    
    if args.checkpoints or args.all:
        total_cleaned += cleanup_checkpoints()
    
    if args.redundant or args.all:
        total_cleaned += remove_redundant_files(args.confirm)
    
    print(f"✅ Nettoyage terminé! {total_cleaned} éléments nettoyés au total.")
    print("🎯 Système d'apprentissage adaptatif optimisé!")


if __name__ == "__main__":
    main()
