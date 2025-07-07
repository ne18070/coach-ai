#!/usr/bin/env python3
"""
Script pour convertir et structurer des données existantes en formats compatibles
avec le système d'apprentissage adaptatif.
"""

import os
import json
import argparse
import pandas as pd
from pathlib import Path
import re

def convert_text_to_completions(input_file, output_file, chunk_size=512):
    """
    Convertit un fichier texte en format de complétion pour l'apprentissage.
    Découpe le texte en morceaux de longueur chunk_size.
    """
    # Lire le fichier texte
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Découper en paragraphes
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    
    # Fusionner les paragraphes en morceaux de longueur chunk_size
    chunks = []
    current_chunk = []
    current_length = 0
    
    for paragraph in paragraphs:
        paragraph_length = len(paragraph)
        
        if current_length + paragraph_length > chunk_size and current_chunk:
            # Ajouter le chunk actuel et en commencer un nouveau
            chunks.append(' '.join(current_chunk))
            current_chunk = [paragraph]
            current_length = paragraph_length
        else:
            # Ajouter le paragraphe au chunk actuel
            current_chunk.append(paragraph)
            current_length += paragraph_length
    
    # Ajouter le dernier chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    # Créer les données de complétion
    completions = [{"text": chunk} for chunk in chunks]
    
    # Sauvegarder au format JSON
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(completions, f, ensure_ascii=False, indent=2)
    
    print(f"Conversion terminée: {len(completions)} morceaux de texte créés dans {output_file}")

def convert_csv_to_conversations(input_file, output_file, question_col='question', answer_col='answer'):
    """
    Convertit un fichier CSV en format de conversation pour l'apprentissage.
    """
    # Lire le fichier CSV
    df = pd.read_csv(input_file)
    
    # Vérifier que les colonnes existent
    if question_col not in df.columns or answer_col not in df.columns:
        raise ValueError(f"Les colonnes {question_col} et/ou {answer_col} n'existent pas dans le fichier CSV")
    
    # Créer les conversations
    conversations = []
    for _, row in df.iterrows():
        conversations.append({
            "input": str(row[question_col]),
            "output": str(row[answer_col])
        })
    
    # Sauvegarder au format JSON
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(conversations, f, ensure_ascii=False, indent=2)
    
    print(f"Conversion terminée: {len(conversations)} conversations créées dans {output_file}")

def convert_qa_to_instructions(input_file, output_file):
    """
    Convertit un fichier JSON de questions-réponses en format d'instruction.
    """
    # Lire le fichier JSON
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convertir en format d'instruction
    instructions = []
    
    # Détecter le format d'entrée
    if isinstance(data, list):
        if data and isinstance(data[0], dict):
            if "input" in data[0] and "output" in data[0]:
                # Format conversation
                for item in data:
                    instructions.append({
                        "instruction": item["input"],
                        "response": item["output"]
                    })
            elif "question" in data[0] and "answer" in data[0]:
                # Format QA
                for item in data:
                    instructions.append({
                        "instruction": item["question"],
                        "response": item["answer"]
                    })
            else:
                raise ValueError("Format JSON non reconnu")
    else:
        raise ValueError("Le fichier JSON doit contenir une liste d'objets")
    
    # Sauvegarder au format JSON
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(instructions, f, ensure_ascii=False, indent=2)
    
    print(f"Conversion terminée: {len(instructions)} instructions créées dans {output_file}")

def create_sample_wolof_data(output_dir):
    """
    Crée des exemples de données en wolof pour démontrer les différents formats.
    """
    # Créer le répertoire de sortie
    os.makedirs(output_dir, exist_ok=True)
    
    # Exemples de conversations
    conversations = [
        {"input": "Naka nga def?", "output": "Mangi fi rekk, yow?"},
        {"input": "Lan mooy intelligence artificielle?", 
         "output": "Intelligence artificielle dafay teknolosi bou nit yi defar ngir saytu ak jëfandikoo xel mu nit."},
        {"input": "Foo joge?", "output": "Dakar laa joge."},
        {"input": "Naka temps bi?", "output": "Temps bi dafa tang, waaye ngelaw li dafa sedd."}
    ]
    
    # Exemples d'instructions
    instructions = [
        {"instruction": "Tektal ma lan mooy wolof bi", 
         "response": "Wolof mooy lakk bu ñuy wax ci Sénégal, Gambie ak Mauritanie. Mooy lakk bu am doole ci Sénégal."},
        {"instruction": "Jox ma benn recette ceebujën", 
         "response": "Ceebujën nii lañ koy togg: danga wara am jën, ceeb, nététou ak xaalis. Da ngay togg jën bi..."},
        {"instruction": "Wadial ci mbir mi aju ci climat bi", 
         "response": "Climat bi dafa sopiku ci addina bi. Températures yi dañuy yokku, glaciers yi di rey..."}
    ]
    
    # Exemple de texte
    text = """
Sénégal mooy réew mi nga xam ne capitale bi mooy Dakar. Mooy benn pays bu nekk ci Afrique de l'Ouest.
Waa Sénégal ñoo leen di wax Sénégalais. Ay langues amna fii: wolof, sérère, pulaar, diola ak yeneen.
Wolof bi mooy langue nationale bu ñuñu gën a wax.

Cuisine sénégalaise bi dafa am: ceebujën, mafé, yassa, thiéboudiène ak yeneen ñam yu neex.
Musique bi itam dafa siiw: mbalax, afrobeat, rap galsen.

Sénégal dafa am histoire bu yaatou. Colonisation française amoon na fi, waaye léegi Sénégal dafa am boppam.
"""
    
    # Sauvegarder les exemples
    with open(os.path.join(output_dir, "wolof_conversations_exemple.json"), 'w', encoding='utf-8') as f:
        json.dump(conversations, f, ensure_ascii=False, indent=2)
    
    with open(os.path.join(output_dir, "wolof_instructions_exemple.json"), 'w', encoding='utf-8') as f:
        json.dump(instructions, f, ensure_ascii=False, indent=2)
    
    with open(os.path.join(output_dir, "wolof_texte_exemple.txt"), 'w', encoding='utf-8') as f:
        f.write(text)
    
    print(f"Exemples de données wolof créés dans le répertoire {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Conversion de données pour entraînement LLM')
    parser.add_argument('--action', choices=['text_to_completion', 'csv_to_conversation', 
                                           'qa_to_instruction', 'create_samples'],
                       required=True, help='Action à effectuer')
    parser.add_argument('--input', help='Fichier d\'entrée')
    parser.add_argument('--output', help='Fichier ou répertoire de sortie')
    parser.add_argument('--question_col', default='question', help='Nom de la colonne des questions (CSV)')
    parser.add_argument('--answer_col', default='answer', help='Nom de la colonne des réponses (CSV)')
    
    args = parser.parse_args()
    
    if args.action == 'create_samples':
        output_dir = args.output or 'data/samples/wolof'
        create_sample_wolof_data(output_dir)
    else:
        # Vérifier que les fichiers d'entrée et de sortie sont spécifiés
        if not args.input or not args.output:
            parser.error("--input et --output sont requis pour cette action")
            
        # Exécuter l'action demandée
        if args.action == 'text_to_completion':
            convert_text_to_completions(args.input, args.output)
        elif args.action == 'csv_to_conversation':
            convert_csv_to_conversations(args.input, args.output, args.question_col, args.answer_col)
        elif args.action == 'qa_to_instruction':
            convert_qa_to_instructions(args.input, args.output)

if __name__ == "__main__":
    main()
