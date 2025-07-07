#!/usr/bin/env python3
"""
Démonstration du système d'apprentissage adaptatif.
Montre comment le système s'adapte automatiquement à différents types de données.
"""

import os
import sys
import json
import time
from pathlib import Path

# Ajouter le dossier src au path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.adaptive_learner import AdaptiveLearner


def create_demo_data():
    """Crée différents types de données pour la démonstration"""
    
    # Données de conversation
    conversation_data = [
        {
            "input": "Comment fonctionne l'apprentissage automatique ?",
            "output": "L'apprentissage automatique utilise des algorithmes pour identifier des patterns dans les données et faire des prédictions."
        },
        {
            "input": "Qu'est-ce qu'un réseau de neurones ?",
            "output": "Un réseau de neurones est un modèle computationnel inspiré du cerveau humain, composé de neurones artificiels interconnectés."
        }
    ]
    
    # Données d'instruction
    instruction_data = [
        {
            "instruction": "Explique ce qu'est l'intelligence artificielle",
            "response": "L'intelligence artificielle est un domaine de l'informatique qui vise à créer des machines capables de simuler l'intelligence humaine."
        },
        {
            "instruction": "Décris le processus d'entraînement d'un modèle ML",
            "response": "L'entraînement consiste à ajuster les paramètres d'un modèle en utilisant des données d'exemple pour qu'il puisse faire des prédictions précises."
        }
    ]
    
    # Données de code
    code_examples = [
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
        "class NeuralNetwork:\n    def __init__(self, layers):\n        self.layers = layers\n    def forward(self, x):\n        return x",
        "import numpy as np\ndef sigmoid(x):\n    return 1 / (1 + np.exp(-x))"
    ]
    
    # Sauvegarder les données
    demo_dir = Path("data/demo")
    demo_dir.mkdir(exist_ok=True)
    
    with open(demo_dir / "conversations.json", "w", encoding="utf-8") as f:
        json.dump(conversation_data, f, indent=2, ensure_ascii=False)
    
    with open(demo_dir / "instructions.json", "w", encoding="utf-8") as f:
        json.dump(instruction_data, f, indent=2, ensure_ascii=False)
    
    with open(demo_dir / "code_examples.json", "w", encoding="utf-8") as f:
        json.dump(code_examples, f, indent=2, ensure_ascii=False)
    
    return [
        demo_dir / "conversations.json",
        demo_dir / "instructions.json", 
        demo_dir / "code_examples.json"
    ]


def demo_adaptive_learning():
    """Démonstration complète du système d'apprentissage adaptatif"""
    
    print("🤖 Démonstration du Système d'Apprentissage Adaptatif")
    print("=" * 60)
    
    print("\n📁 Création des données de démonstration...")
    demo_files = create_demo_data()
    print(f"✅ {len(demo_files)} fichiers créés")
    
    print("\n🧠 Initialisation du système d'apprentissage...")
    learner = AdaptiveLearner(
        base_model='distilgpt2',
        max_seq_length=128,
        learning_rate=5e-4
    )
    print("✅ Système initialisé")
    
    print("\n🔄 Test d'apprentissage adaptatif sur différents types de données...")
    
    for i, file_path in enumerate(demo_files, 1):
        print(f"\n--- Test {i}: {file_path.name} ---")
        
        try:
            # Observer et apprendre
            print(f"🔍 Analyse du fichier: {file_path.name}")
            result = learner.observe_and_learn(str(file_path))
            
            # Afficher les résultats
            for pattern_name, pattern_result in result.items():
                status = "✅" if pattern_result['status'] == 'success' else "❌"
                print(f"{status} {pattern_name}: {pattern_result['pattern_type']}")
                if pattern_result['status'] == 'success':
                    print(f"   📊 Exemples traités: {pattern_result['num_examples']}")
                    print(f"   📉 Loss: {pattern_result['training_loss']:.4f}")
                else:
                    print(f"   ❌ Erreur: {pattern_result.get('error', 'Inconnue')}")
            
            # Test de génération
            if any(r['status'] == 'success' for r in result.values()):
                print(f"🧪 Test de génération...")
                test_prompts = [
                    "L'intelligence artificielle",
                    "Un réseau de neurones",
                    "def fonction"
                ]
                
                for prompt in test_prompts:
                    try:
                        response = learner.generate_response(prompt, max_length=60)
                        if response and response.strip():
                            print(f"   Prompt: '{prompt}' → Réponse: '{response[:50]}{'...' if len(response) > 50 else ''}'")
                            break
                    except Exception as e:
                        continue
            
        except Exception as e:
            print(f"❌ Erreur lors du traitement de {file_path.name}: {e}")
        
        print()
    
    print("🎯 Démonstration terminée !")
    print("\nLe système a montré sa capacité à :")
    print("• 🔍 Détecter automatiquement le type de données")
    print("• 🧠 S'adapter à différents formats (conversation, instruction, code)")
    print("• 📈 Apprendre de nouveaux patterns")
    print("• 🤖 Générer du contenu basé sur l'apprentissage")


def interactive_demo():
    """Démonstration interactive"""
    print("\n🎮 Mode interactif")
    print("Vous pouvez maintenant tester le système avec vos propres données.")
    print("Déposez un fichier JSON dans 'data/incoming/' et le système l'analysera automatiquement.")
    
    learner = AdaptiveLearner(base_model='distilgpt2', max_seq_length=128)
    
    while True:
        try:
            user_input = input("\n💬 Entrez votre question (ou 'quit' pour quitter): ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if user_input:
                response = learner.generate_response(user_input, max_length=100)
                print(f"🤖 Réponse: {response}")
        
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Erreur: {e}")
    
    print("\n👋 À bientôt !")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Démonstration du système d'apprentissage adaptatif")
    parser.add_argument("--interactive", action="store_true", help="Mode interactif")
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_demo()
    else:
        demo_adaptive_learning()
