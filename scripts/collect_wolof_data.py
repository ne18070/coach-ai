#!/usr/bin/env python3
"""
Script pour collecter interactivement des données en wolof ou toute autre langue
pour entraîner le système d'apprentissage adaptatif.
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime

def collect_conversations(output_file, lang="wolof"):
    """Collecte des conversations de manière interactive."""
    conversations = []
    
    print(f"\n===== COLLECTE DE DONNÉES EN {lang.upper()} =====")
    print("Entrez vos paires de questions-réponses.")
    print("Pour terminer, laissez la question vide et appuyez sur Entrée.")
    
    while True:
        print("\n--- Nouvelle conversation ---")
        question = input("Question: ")
        
        if not question:
            break
            
        answer = input("Réponse: ")
        
        conversations.append({
            "input": question,
            "output": answer
        })
        
        print(f"Conversation enregistrée! ({len(conversations)} au total)")
    
    # Créer le dossier si nécessaire
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Vérifier si le fichier existe déjà
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
            conversations = existing_data + conversations
    
    # Sauvegarder les données
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(conversations, f, ensure_ascii=False, indent=2)
    
    print(f"\nDonnées sauvegardées dans {output_file}")
    print(f"Total: {len(conversations)} conversations")

def collect_instructions(output_file, lang="wolof"):
    """Collecte des instructions et réponses de manière interactive."""
    instructions = []
    
    print(f"\n===== COLLECTE D'INSTRUCTIONS EN {lang.upper()} =====")
    print("Entrez vos paires d'instruction-réponse.")
    print("Pour terminer, laissez l'instruction vide et appuyez sur Entrée.")
    
    while True:
        print("\n--- Nouvelle instruction ---")
        instruction = input("Instruction: ")
        
        if not instruction:
            break
            
        response = input("Réponse: ")
        
        instructions.append({
            "instruction": instruction,
            "response": response
        })
        
        print(f"Instruction enregistrée! ({len(instructions)} au total)")
    
    # Créer le dossier si nécessaire
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Vérifier si le fichier existe déjà
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
            instructions = existing_data + instructions
    
    # Sauvegarder les données
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(instructions, f, ensure_ascii=False, indent=2)
    
    print(f"\nDonnées sauvegardées dans {output_file}")
    print(f"Total: {len(instructions)} instructions")

def collect_text(output_file, lang="wolof"):
    """Collecte du texte brut de manière interactive."""
    texts = []
    
    print(f"\n===== COLLECTE DE TEXTE EN {lang.upper()} =====")
    print("Entrez vos textes ligne par ligne.")
    print("Pour terminer un texte, entrez une ligne vide.")
    print("Pour quitter complètement, entrez 'q' sur une ligne vide.")
    
    while True:
        print("\n--- Nouveau texte ---")
        print("(Entrez une ligne vide pour terminer ce texte, 'q' pour quitter)")
        
        current_text = []
        while True:
            line = input("> ")
            if not line:
                confirm = input("Texte terminé? (Entrez 'q' pour quitter ou appuyez sur Entrée pour continuer): ")
                if confirm.lower() == 'q':
                    break
                else:
                    texts.append("\n".join(current_text))
                    current_text = []
                    print("Texte enregistré! Commencez-en un nouveau ou entrez 'q' pour quitter.")
                    break
            current_text.append(line)
        
        if not current_text and confirm.lower() == 'q':
            break
    
    # Créer le dossier si nécessaire
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Vérifier si le fichier existe déjà
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            existing_text = f.read()
            full_text = existing_text + "\n\n" + "\n\n".join(texts)
    else:
        full_text = "\n\n".join(texts)
    
    # Sauvegarder les données
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(full_text)
    
    print(f"\nDonnées sauvegardées dans {output_file}")
    print(f"Total: {len(texts)} paragraphes")

def main():
    parser = argparse.ArgumentParser(description='Collecte de données pour entraînement LLM')
    parser.add_argument('--type', choices=['conversation', 'instruction', 'text'], 
                       default='conversation', help='Type de données à collecter')
    parser.add_argument('--lang', default='wolof', help='Langue des données')
    parser.add_argument('--output', help='Fichier de sortie')
    
    args = parser.parse_args()
    
    # Générer un nom de fichier par défaut si non spécifié
    if not args.output:
        data_dir = Path("data/collected")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if args.type == 'conversation':
            output_file = data_dir / f"{args.lang}_conversations_{timestamp}.json"
        elif args.type == 'instruction':
            output_file = data_dir / f"{args.lang}_instructions_{timestamp}.json"
        else:
            output_file = data_dir / f"{args.lang}_text_{timestamp}.txt"
    else:
        output_file = Path(args.output)
    
    # Collecter les données selon le type
    if args.type == 'conversation':
        collect_conversations(output_file, args.lang)
    elif args.type == 'instruction':
        collect_instructions(output_file, args.lang)
    else:
        collect_text(output_file, args.lang)

if __name__ == "__main__":
    main()
