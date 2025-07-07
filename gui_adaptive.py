#!/usr/bin/env python3
"""
Interface graphique simple pour montrer l'apprentissage adaptatif en temps réel.
Permet de voir le système apprendre et s'adapter visuellement.
"""

import sys
import os
import json
import time
import threading
from pathlib import Path
from datetime import datetime

# Ajouter le dossier src au path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    import tkinter as tk
    from tkinter import ttk, scrolledtext, filedialog, messagebox
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False

from src.adaptive_learner import AdaptiveLearner


class AdaptiveLearningGUI:
    """Interface graphique pour l'apprentissage adaptatif"""
    
    def __init__(self):
        if not GUI_AVAILABLE:
            raise ImportError("Interface graphique non disponible (tkinter non installé)")
        
        self.root = tk.Tk()
        self.root.title("🧠 Système d'Apprentissage Adaptatif")
        self.root.geometry("1000x700")
        
        # Variables
        self.learner = None
        self.is_learning = False
        self.learning_stats = {
            'patterns_learned': 0,
            'files_processed': 0,
            'successful_adaptations': 0,
            'start_time': None
        }
        
        self.setup_ui()
    
    def setup_ui(self):
        """Configure l'interface utilisateur"""
        
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configuration de la grille
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(3, weight=1)
        
        # Titre
        title_label = ttk.Label(main_frame, text="🧠 Système d'Apprentissage Adaptatif", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Frame de contrôle
        control_frame = ttk.LabelFrame(main_frame, text="Contrôles", padding="10")
        control_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Boutons de contrôle
        self.start_button = ttk.Button(control_frame, text="🚀 Démarrer l'Apprentissage", 
                                      command=self.start_learning)
        self.start_button.grid(row=0, column=0, padx=(0, 10))
        
        self.stop_button = ttk.Button(control_frame, text="⏹ Arrêter", 
                                     command=self.stop_learning, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=1, padx=(0, 10))
        
        self.add_data_button = ttk.Button(control_frame, text="📄 Ajouter des Données", 
                                         command=self.add_data_file)
        self.add_data_button.grid(row=0, column=2, padx=(0, 10))
        
        self.test_button = ttk.Button(control_frame, text="🧪 Tester", 
                                     command=self.test_generation)
        self.test_button.grid(row=0, column=3)
        
        # Frame de statistiques
        stats_frame = ttk.LabelFrame(main_frame, text="Statistiques d'Apprentissage", padding="10")
        stats_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Statistiques
        self.stats_text = tk.StringVar(value="En attente de démarrage...")
        stats_label = ttk.Label(stats_frame, textvariable=self.stats_text)
        stats_label.grid(row=0, column=0, sticky=tk.W)
        
        # Progress bar
        self.progress = ttk.Progressbar(stats_frame, mode='indeterminate')
        self.progress.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        stats_frame.columnconfigure(0, weight=1)
        
        # Frame de logs
        logs_frame = ttk.LabelFrame(main_frame, text="Journal d'Apprentissage", padding="10")
        logs_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        logs_frame.columnconfigure(0, weight=1)
        logs_frame.rowconfigure(0, weight=1)
        
        # Zone de texte pour les logs
        self.log_text = scrolledtext.ScrolledText(logs_frame, height=15, width=60)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Frame de test
        test_frame = ttk.LabelFrame(main_frame, text="Test de Génération", padding="10")
        test_frame.grid(row=3, column=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(10, 0))
        test_frame.columnconfigure(0, weight=1)
        test_frame.rowconfigure(1, weight=1)
        
        # Zone de saisie pour test
        ttk.Label(test_frame, text="Entrez votre question:").grid(row=0, column=0, sticky=tk.W)
        self.test_input = tk.Entry(test_frame, width=30)
        self.test_input.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        self.test_input.bind('<Return>', lambda e: self.test_generation())
        
        # Zone de réponse
        ttk.Label(test_frame, text="Réponse générée:").grid(row=2, column=0, sticky=tk.W, pady=(10, 0))
        self.response_text = scrolledtext.ScrolledText(test_frame, height=10, width=30)
        self.response_text.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(5, 0))
    
    def log_message(self, message):
        """Ajoute un message au journal"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        full_message = f"[{timestamp}] {message}\n"
        
        self.log_text.insert(tk.END, full_message)
        self.log_text.see(tk.END)
        self.root.update_idletasks()
    
    def update_stats(self):
        """Met à jour les statistiques affichées"""
        if self.learning_stats['start_time']:
            elapsed = datetime.now() - self.learning_stats['start_time']
            elapsed_str = str(elapsed).split('.')[0]  # Enlever les microsecondes
        else:
            elapsed_str = "00:00:00"
        
        stats = (f"⏰ Temps: {elapsed_str} | "
                f"📄 Fichiers: {self.learning_stats['files_processed']} | "
                f"✅ Adaptations: {self.learning_stats['successful_adaptations']} | "
                f"🧠 Patterns: {self.learning_stats['patterns_learned']}")
        
        self.stats_text.set(stats)
    
    def start_learning(self):
        """Démarre le système d'apprentissage"""
        try:
            self.log_message("🚀 Initialisation du système d'apprentissage...")
            
            # Initialiser le learner
            self.learner = AdaptiveLearner(
                base_model='distilgpt2',
                max_seq_length=128,
                learning_rate=5e-4
            )
            
            self.log_message("✅ Système initialisé avec succès")
            
            # Mettre à jour l'interface
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.progress.start()
            
            # Démarrer les statistiques
            self.learning_stats['start_time'] = datetime.now()
            self.is_learning = True
            
            # Créer des dossiers
            Path("data/incoming").mkdir(parents=True, exist_ok=True)
            Path("data/processed").mkdir(parents=True, exist_ok=True)
            
            self.log_message("📁 Surveillance du dossier data/incoming/")
            self.log_message("💡 Astuce: Utilisez le bouton 'Ajouter des Données' pour tester")
            
            # Démarrer la surveillance dans un thread séparé
            self.monitoring_thread = threading.Thread(target=self.monitor_folder, daemon=True)
            self.monitoring_thread.start()
            
        except Exception as e:
            self.log_message(f"❌ Erreur lors de l'initialisation: {e}")
            messagebox.showerror("Erreur", f"Erreur lors de l'initialisation: {e}")
    
    def stop_learning(self):
        """Arrête le système d'apprentissage"""
        self.is_learning = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.progress.stop()
        
        self.log_message("🛑 Système d'apprentissage arrêté")
    
    def monitor_folder(self):
        """Surveille le dossier pour les nouveaux fichiers"""
        watch_dir = Path("data/incoming")
        processed_files = set()
        
        while self.is_learning:
            try:
                # Vérifier les nouveaux fichiers
                current_files = set(watch_dir.glob("*.json")) | set(watch_dir.glob("*.txt"))
                new_files = current_files - processed_files
                
                for file_path in new_files:
                    self.process_new_file(file_path)
                    processed_files.add(file_path)
                
                # Mettre à jour les stats
                self.update_stats()
                
                time.sleep(2)  # Vérifier toutes les 2 secondes
                
            except Exception as e:
                self.log_message(f"❌ Erreur de surveillance: {e}")
                time.sleep(5)
    
    def process_new_file(self, file_path):
        """Traite un nouveau fichier"""
        try:
            self.log_message(f"📥 Nouveau fichier: {file_path.name}")
            
            # Observer et apprendre
            results = self.learner.observe_and_learn(str(file_path))
            
            # Analyser les résultats
            successful = 0
            patterns = []
            
            for pattern_name, result in results.items():
                if result['status'] == 'success':
                    successful += 1
                    patterns.append(result['pattern_type'])
                    self.log_message(f"✅ {result['pattern_type']}: {result['num_examples']} exemples")
                else:
                    self.log_message(f"❌ Échec: {result.get('error', 'Erreur inconnue')}")
            
            # Mettre à jour les statistiques
            self.learning_stats['files_processed'] += 1
            self.learning_stats['successful_adaptations'] += successful
            self.learning_stats['patterns_learned'] = len(set(patterns))
            
            # Archiver le fichier
            archive_dir = Path("data/processed")
            archived_file = archive_dir / f"{file_path.stem}_{int(time.time())}{file_path.suffix}"
            file_path.rename(archived_file)
            
            self.log_message(f"📂 Archivé: {archived_file.name}")
            
        except Exception as e:
            self.log_message(f"❌ Erreur traitement {file_path.name}: {e}")
    
    def add_data_file(self):
        """Permet d'ajouter manuellement un fichier de données"""
        file_path = filedialog.askopenfilename(
            title="Sélectionner un fichier de données",
            filetypes=[
                ("Fichiers JSON", "*.json"),
                ("Fichiers texte", "*.txt"),
                ("Tous les fichiers", "*.*")
            ]
        )
        
        if file_path:
            # Copier le fichier vers le dossier surveillé
            import shutil
            source = Path(file_path)
            destination = Path("data/incoming") / f"manual_{int(time.time())}_{source.name}"
            
            shutil.copy2(source, destination)
            self.log_message(f"📄 Fichier ajouté: {source.name}")
    
    def test_generation(self):
        """Teste la génération de texte"""
        if not self.learner:
            messagebox.showwarning("Attention", "Veuillez d'abord démarrer l'apprentissage")
            return
        
        prompt = self.test_input.get().strip()
        if not prompt:
            messagebox.showwarning("Attention", "Veuillez entrer une question")
            return
        
        try:
            self.log_message(f"🧪 Test: '{prompt}'")
            response = self.learner.generate_response(prompt, max_length=100)
            
            # Afficher la réponse
            self.response_text.delete(1.0, tk.END)
            self.response_text.insert(tk.END, response)
            
            self.log_message(f"💬 Réponse générée")
            
        except Exception as e:
            self.log_message(f"❌ Erreur génération: {e}")
            messagebox.showerror("Erreur", f"Erreur lors de la génération: {e}")
    
    def run(self):
        """Lance l'interface graphique"""
        self.root.mainloop()


def create_demo_files():
    """Crée des fichiers de démonstration pour tester l'interface"""
    demo_dir = Path("data/demo_gui")
    demo_dir.mkdir(parents=True, exist_ok=True)
    
    # Données de conversation
    conversations = [
        {
            "input": "Comment ça va ?",
            "output": "Ça va très bien, merci ! Et vous ?"
        },
        {
            "input": "Qu'est-ce que l'apprentissage adaptatif ?",
            "output": "L'apprentissage adaptatif permet à un système de s'ajuster automatiquement à de nouveaux types de données."
        }
    ]
    
    # Données de code
    code_data = [
        "def hello_world():\n    print('Hello, World!')",
        "class AdaptiveAI:\n    def __init__(self):\n        self.knowledge = {}"
    ]
    
    # Sauvegarder
    with open(demo_dir / "conversations_demo.json", "w", encoding="utf-8") as f:
        json.dump(conversations, f, indent=2, ensure_ascii=False)
    
    with open(demo_dir / "code_demo.json", "w", encoding="utf-8") as f:
        json.dump(code_data, f, indent=2, ensure_ascii=False)
    
    print(f"📄 Fichiers de démonstration créés dans {demo_dir}")
    return demo_dir


def main():
    if not GUI_AVAILABLE:
        print("❌ Interface graphique non disponible")
        print("💡 Alternative: utilisez 'python demo_adaptive.py' ou 'python continuous_adaptive.py'")
        return
    
    print("🧠 Lancement de l'interface d'apprentissage adaptatif...")
    
    # Créer des fichiers de démo
    demo_dir = create_demo_files()
    
    try:
        # Lancer l'interface
        gui = AdaptiveLearningGUI()
        
        print(f"📱 Interface lancée !")
        print(f"💡 Fichiers de test disponibles dans: {demo_dir}")
        print("🎯 Utilisez le bouton 'Ajouter des Données' pour tester")
        
        gui.run()
        
    except Exception as e:
        print(f"❌ Erreur: {e}")


if __name__ == "__main__":
    main()
