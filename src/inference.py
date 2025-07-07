"""
Module d'inférence pour les modèles fine-tunés
"""
import torch
from typing import Dict, Any, List, Optional, Union
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList
)
from peft import PeftModel
import logging

from .utils import setup_logging, get_device
from .model_manager import ModelManager

logger = setup_logging()

class StopOnTokens(StoppingCriteria):
    """Critère d'arrêt personnalisé basé sur des tokens spécifiques."""
    
    def __init__(self, stop_token_ids: List[int]):
        self.stop_token_ids = stop_token_ids
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_id in self.stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

class ModelInference:
    """Classe pour l'inférence avec des modèles fine-tunés."""
    
    def __init__(self, model_path: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialise le module d'inférence.
        
        Args:
            model_path: Chemin vers le modèle fine-tuné
            config: Configuration d'inférence (optionnel)
        """
        self.model_path = model_path
        self.config = config or {}
        self.device = get_device()
        self.logger = logging.getLogger(__name__)
        
        self.tokenizer = None
        self.model = None
        
        # Charger le modèle
        self.load_model()
    
    def load_model(self) -> None:
        """Charge le modèle et le tokenizer."""
        self.logger.info(f"Chargement du modèle depuis {self.model_path}")
        
        # Charger le tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # Vérifier si c'est un modèle PEFT (LoRA)
        if self._is_peft_model():
            self._load_peft_model()
        else:
            self._load_standard_model()
        
        self.logger.info("Modèle chargé avec succès")
    
    def _is_peft_model(self) -> bool:
        """Vérifie si le modèle est un modèle PEFT."""
        import os
        adapter_config_path = os.path.join(self.model_path, "adapter_config.json")
        return os.path.exists(adapter_config_path)
    
    def _load_peft_model(self) -> None:
        """Charge un modèle PEFT (LoRA)."""
        import json
        import os
        
        # Lire la configuration de l'adaptateur
        adapter_config_path = os.path.join(self.model_path, "adapter_config.json")
        with open(adapter_config_path, 'r') as f:
            adapter_config = json.load(f)
        
        base_model_name = adapter_config['base_model_name_or_path']
        
        # Charger le modèle de base
        self.logger.info(f"Chargement du modèle de base: {base_model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Charger les adaptateurs PEFT
        self.model = PeftModel.from_pretrained(base_model, self.model_path)
        self.model = self.model.merge_and_unload()  # Fusionner pour l'inférence
    
    def _load_standard_model(self) -> None:
        """Charge un modèle standard (non-PEFT)."""
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
    
    def generate(self, 
                prompt: str,
                max_new_tokens: int = None,
                temperature: float = None,
                top_p: float = None,
                top_k: int = None,
                do_sample: bool = None,
                stop_sequences: List[str] = None,
                **kwargs) -> str:
        """
        Génère une réponse à partir d'un prompt.
        
        Args:
            prompt: Prompt d'entrée
            max_new_tokens: Nombre maximum de nouveaux tokens
            temperature: Température pour le sampling
            top_p: Paramètre top-p pour le sampling
            top_k: Paramètre top-k pour le sampling
            do_sample: Utiliser le sampling ou non
            stop_sequences: Séquences d'arrêt
            **kwargs: Autres paramètres pour la génération
        
        Returns:
            Texte généré
        """
        # Utiliser les paramètres de la config par défaut
        inference_config = self.config.get('inference', {})
        
        max_new_tokens = max_new_tokens or inference_config.get('max_new_tokens', 256)
        temperature = temperature or inference_config.get('temperature', 0.7)
        top_p = top_p or inference_config.get('top_p', 0.9)
        top_k = top_k or inference_config.get('top_k', 50)
        do_sample = do_sample if do_sample is not None else inference_config.get('do_sample', True)
        
        # Tokeniser le prompt
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # Configurer les critères d'arrêt
        stopping_criteria = self._create_stopping_criteria(stop_sequences)
        
        # Paramètres de génération
        generation_kwargs = {
            'input_ids': inputs,
            'max_new_tokens': max_new_tokens,
            'temperature': temperature,
            'top_p': top_p,
            'top_k': top_k,
            'do_sample': do_sample,
            'pad_token_id': self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            'stopping_criteria': stopping_criteria,
            **kwargs
        }
        
        # Générer
        with torch.no_grad():
            outputs = self.model.generate(**generation_kwargs)
        
        # Décoder seulement les nouveaux tokens
        new_tokens = outputs[0][inputs.shape[-1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        # Nettoyer la réponse
        response = self._clean_response(response, stop_sequences)
        
        return response
    
    def _create_stopping_criteria(self, stop_sequences: List[str] = None) -> StoppingCriteriaList:
        """Crée les critères d'arrêt pour la génération."""
        stopping_criteria = StoppingCriteriaList()
        
        if stop_sequences:
            # Convertir les séquences d'arrêt en IDs de tokens
            stop_token_ids = []
            for seq in stop_sequences:
                tokens = self.tokenizer.encode(seq, add_special_tokens=False)
                stop_token_ids.extend(tokens)
            
            if stop_token_ids:
                stopping_criteria.append(StopOnTokens(stop_token_ids))
        
        return stopping_criteria
    
    def _clean_response(self, response: str, stop_sequences: List[str] = None) -> str:
        """Nettoie la réponse générée."""
        # Supprimer les séquences d'arrêt de la réponse
        if stop_sequences:
            for seq in stop_sequences:
                if seq in response:
                    response = response.split(seq)[0]
        
        # Supprimer les espaces en début/fin
        response = response.strip()
        
        return response
    
    def generate_batch(self, 
                      prompts: List[str],
                      batch_size: int = 4,
                      **generation_kwargs) -> List[str]:
        """
        Génère des réponses pour une liste de prompts.
        
        Args:
            prompts: Liste des prompts
            batch_size: Taille du batch
            **generation_kwargs: Paramètres pour la génération
        
        Returns:
            Liste des réponses générées
        """
        responses = []
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_responses = []
            
            for prompt in batch_prompts:
                response = self.generate(prompt, **generation_kwargs)
                batch_responses.append(response)
            
            responses.extend(batch_responses)
            
            if i + batch_size < len(prompts):
                self.logger.info(f"Traité {i + batch_size}/{len(prompts)} prompts")
        
        return responses
    
    def chat(self, 
             messages: List[Dict[str, str]], 
             system_message: Optional[str] = None,
             **generation_kwargs) -> str:
        """
        Interface de chat avec historique de conversation.
        
        Args:
            messages: Liste des messages [{"role": "user"/"assistant", "content": "..."}]
            system_message: Message système optionnel
            **generation_kwargs: Paramètres pour la génération
        
        Returns:
            Réponse de l'assistant
        """
        # Construire le prompt de conversation
        conversation = ""
        
        # Ajouter le message système
        if system_message:
            conversation += f"<|system|>\n{system_message}\n\n"
        
        # Ajouter l'historique
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "user":
                conversation += f"<|user|>\n{content}\n\n"
            elif role == "assistant":
                conversation += f"<|assistant|>\n{content}\n\n"
        
        # Ajouter le marqueur pour la réponse de l'assistant
        conversation += "<|assistant|>\n"
        
        # Générer la réponse
        response = self.generate(
            conversation,
            stop_sequences=["<|user|>", "<|endoftext|>"],
            **generation_kwargs
        )
        
        return response
    
    def get_model_info(self) -> Dict[str, Any]:
        """Retourne des informations sur le modèle chargé."""
        info = {
            "model_path": self.model_path,
            "model_type": type(self.model).__name__,
            "device": str(self.device),
            "vocab_size": len(self.tokenizer) if self.tokenizer else None
        }
        
        if self.tokenizer:
            info.update({
                "pad_token": self.tokenizer.pad_token,
                "eos_token": self.tokenizer.eos_token,
                "bos_token": self.tokenizer.bos_token
            })
        
        return info
    
    def benchmark(self, 
                 prompt: str = "Expliquez ce qu'est l'intelligence artificielle.",
                 num_runs: int = 5) -> Dict[str, float]:
        """
        Effectue un benchmark simple du modèle.
        
        Args:
            prompt: Prompt à utiliser pour le benchmark
            num_runs: Nombre d'exécutions
        
        Returns:
            Statistiques de performance
        """
        import time
        
        times = []
        token_counts = []
        
        self.logger.info(f"Début du benchmark ({num_runs} exécutions)")
        
        for i in range(num_runs):
            start_time = time.time()
            response = self.generate(prompt, max_new_tokens=100)
            end_time = time.time()
            
            generation_time = end_time - start_time
            token_count = len(self.tokenizer.encode(response))
            
            times.append(generation_time)
            token_counts.append(token_count)
            
            self.logger.info(f"Exécution {i+1}/{num_runs}: {generation_time:.2f}s, {token_count} tokens")
        
        # Calculer les statistiques
        avg_time = sum(times) / len(times)
        avg_tokens = sum(token_counts) / len(token_counts)
        tokens_per_second = avg_tokens / avg_time
        
        stats = {
            "avg_generation_time": avg_time,
            "avg_tokens_generated": avg_tokens,
            "tokens_per_second": tokens_per_second,
            "min_time": min(times),
            "max_time": max(times)
        }
        
        self.logger.info(f"Benchmark terminé. Tokens/seconde: {tokens_per_second:.2f}")
        
        return stats
