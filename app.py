#!/usr/bin/env python3
"""
Interface web Gradio pour tester le modèle fine-tuné
"""
import sys
import os
from pathlib import Path
import gradio as gr

# Ajouter le répertoire parent au PYTHONPATH
sys.path.append(str(Path(__file__).parent))

from src.inference import ModelInference
from src.utils import load_config

class ChatInterface:
    def __init__(self, model_path, config_path=None):
        """Initialise l'interface de chat"""
        self.config = None
        if config_path and os.path.exists(config_path):
            self.config = load_config(config_path)
        
        print(f"🔄 Chargement du modèle depuis {model_path}")
        self.model = ModelInference(model_path, self.config)
        print("✅ Modèle chargé avec succès")
        
        # Historique de conversation
        self.conversation_history = []
    
    def chat(self, message, history):
        """Fonction de chat pour Gradio"""
        if not message.strip():
            return history, ""
        
        # Ajouter le message de l'utilisateur à l'historique
        self.conversation_history.append({"role": "user", "content": message})
        
        try:
            # Générer la réponse
            response = self.model.chat(
                self.conversation_history,
                max_new_tokens=256,
                temperature=0.7
            )
            
            # Ajouter la réponse à l'historique
            self.conversation_history.append({"role": "assistant", "content": response})
            
            # Mettre à jour l'historique Gradio
            history.append([message, response])
            
        except Exception as e:
            error_msg = f"Erreur: {str(e)}"
            history.append([message, error_msg])
        
        return history, ""
    
    def clear_history(self):
        """Efface l'historique de conversation"""
        self.conversation_history = []
        return []
    
    def get_model_info(self):
        """Retourne les informations du modèle"""
        info = self.model.get_model_info()
        info_text = "🤖 **Informations du Modèle**\n\n"
        for key, value in info.items():
            info_text += f"- **{key}**: {value}\n"
        return info_text

def create_interface(model_path, config_path=None):
    """Crée l'interface Gradio"""
    
    chat_interface = ChatInterface(model_path, config_path)
    
    with gr.Blocks(title="Fine-tuned LLM Chat", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🤖 Chat avec votre LLM Fine-tuné")
        gr.Markdown("Testez votre modèle fine-tuné dans cette interface interactive.")
        
        with gr.Tab("💬 Chat"):
            chatbot = gr.Chatbot(
                value=[],
                height=400,
                bubble_full_width=False
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Tapez votre message ici...",
                    label="Message",
                    scale=4
                )
                send_btn = gr.Button("Envoyer", scale=1, variant="primary")
            
            with gr.Row():
                clear_btn = gr.Button("🗑️ Effacer l'historique", variant="secondary")
            
            # Actions
            msg.submit(
                chat_interface.chat,
                inputs=[msg, chatbot],
                outputs=[chatbot, msg]
            )
            
            send_btn.click(
                chat_interface.chat,
                inputs=[msg, chatbot],
                outputs=[chatbot, msg]
            )
            
            clear_btn.click(
                chat_interface.clear_history,
                outputs=[chatbot]
            )
        
        with gr.Tab("ℹ️ Informations"):
            info_display = gr.Markdown(chat_interface.get_model_info())
            
            with gr.Row():
                benchmark_btn = gr.Button("🏃 Lancer un Benchmark", variant="secondary")
            
            benchmark_output = gr.Textbox(
                label="Résultats du Benchmark",
                lines=10,
                interactive=False
            )
            
            def run_benchmark():
                try:
                    stats = chat_interface.model.benchmark(num_runs=3)
                    result = "📊 **Résultats du Benchmark**\n\n"
                    for metric, value in stats.items():
                        result += f"- **{metric}**: {value:.3f}\n"
                    return result
                except Exception as e:
                    return f"❌ Erreur durante le benchmark: {e}"
            
            benchmark_btn.click(
                run_benchmark,
                outputs=[benchmark_output]
            )
        
        with gr.Tab("⚙️ Configuration"):
            gr.Markdown("### Paramètres de génération")
            
            with gr.Row():
                temp_slider = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=0.7,
                    step=0.1,
                    label="Température"
                )
                
                max_tokens_slider = gr.Slider(
                    minimum=50,
                    maximum=500,
                    value=256,
                    step=50,
                    label="Max tokens"
                )
            
            gr.Markdown("### Chemins")
            gr.Textbox(value=model_path, label="Chemin du modèle", interactive=False)
            if config_path:
                gr.Textbox(value=config_path, label="Fichier de configuration", interactive=False)
    
    return interface

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Interface web pour tester le modèle fine-tuné")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/my_finetuned_model",
        help="Chemin vers le modèle fine-tuné"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Chemin vers le fichier de configuration (optionnel)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port pour l'interface web"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Créer un lien public partageable"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"❌ Modèle non trouvé: {args.model_path}")
        print("Assurez-vous d'avoir entraîné un modèle d'abord.")
        return 1
    
    try:
        # Créer l'interface
        interface = create_interface(args.model_path, args.config)
        
        # Lancer l'interface
        print(f"🌐 Lancement de l'interface web sur le port {args.port}")
        print(f"📱 Accédez à l'interface via: http://localhost:{args.port}")
        
        interface.launch(
            server_port=args.port,
            share=args.share,
            show_error=True
        )
        
    except Exception as e:
        print(f"❌ Erreur au lancement de l'interface: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
