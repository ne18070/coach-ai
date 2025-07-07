#!/usr/bin/env python3
"""
Script pour tester l'inférence avec un modèle fine-tuné
"""
import argparse
import sys
from pathlib import Path

# Ajouter le répertoire src au PYTHONPATH
sys.path.append(str(Path(__file__).parent / "src"))

from src.inference import ModelInference
from src.utils import load_config

def main():
    parser = argparse.ArgumentParser(description="Test d'inférence avec un modèle fine-tuné")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Chemin vers le modèle fine-tuné"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Chemin vers le fichier de configuration (optionnel)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Prompt pour la génération"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Mode interactif"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Effectuer un benchmark du modèle"
    )
    
    args = parser.parse_args()
    
    # Charger la configuration si fournie
    config = None
    if args.config:
        config = load_config(args.config)
    
    # Initialiser le modèle d'inférence
    print(f"Chargement du modèle depuis {args.model_path}")
    model = ModelInference(args.model_path, config)
    
    # Afficher les informations du modèle
    info = model.get_model_info()
    print("\nInformations du modèle:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    print()
    
    if args.benchmark:
        # Mode benchmark
        print("Début du benchmark...")
        stats = model.benchmark()
        print("\nRésultats du benchmark:")
        for metric, value in stats.items():
            print(f"  {metric}: {value:.3f}")
    
    elif args.interactive:
        # Mode interactif
        print("Mode interactif activé. Tapez 'quit' pour quitter.")
        print("Vous pouvez utiliser des commandes spéciales:")
        print("  /clear - Effacer l'historique")
        print("  /info - Afficher les informations du modèle")
        print("  /quit ou /exit - Quitter")
        print()
        
        conversation_history = []
        
        while True:
            try:
                user_input = input("Vous: ").strip()
                
                if user_input.lower() in ['quit', 'exit', '/quit', '/exit']:
                    break
                elif user_input == '/clear':
                    conversation_history = []
                    print("Historique effacé.")
                    continue
                elif user_input == '/info':
                    for key, value in info.items():
                        print(f"  {key}: {value}")
                    continue
                elif not user_input:
                    continue
                
                # Ajouter le message de l'utilisateur à l'historique
                conversation_history.append({"role": "user", "content": user_input})
                
                # Générer la réponse
                response = model.chat(conversation_history)
                print(f"Assistant: {response}")
                
                # Ajouter la réponse à l'historique
                conversation_history.append({"role": "assistant", "content": response})
                
            except KeyboardInterrupt:
                print("\nAu revoir!")
                break
            except Exception as e:
                print(f"Erreur: {e}")
    
    elif args.prompt:
        # Mode prompt unique
        print(f"Prompt: {args.prompt}")
        response = model.generate(args.prompt)
        print(f"Réponse: {response}")
    
    else:
        # Exemple par défaut
        default_prompt = "Expliquez ce qu'est l'intelligence artificielle en quelques phrases."
        print(f"Prompt par défaut: {default_prompt}")
        response = model.generate(default_prompt)
        print(f"Réponse: {response}")

if __name__ == "__main__":
    main()
