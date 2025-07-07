"""
Gestionnaire de modèles pour le fine-tuning
"""
import os
import torch
from typing import Dict, Any, Optional
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoConfig,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    TaskType,
    PeftModel
)
import logging

from .utils import setup_logging, count_parameters, get_device

logger = setup_logging()

class ModelManager:
    """Gestionnaire pour charger et configurer les modèles."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_config = config['model']
        self.lora_config = config['lora']
        self.logger = logging.getLogger(__name__)
        
        self.tokenizer = None
        self.model = None
        self.device = get_device()
    
    def load_tokenizer(self) -> AutoTokenizer:
        """Charge le tokenizer."""
        self.logger.info(f"Chargement du tokenizer: {self.model_config['name']}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config['name'],
            cache_dir=self.model_config.get('cache_dir'),
            trust_remote_code=True
        )
        
        # Ajouter un token de padding si nécessaire
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        self.logger.info(f"Tokenizer chargé. Vocabulaire: {len(self.tokenizer)} tokens")
        return self.tokenizer
    
    def load_model(self, load_in_8bit: bool = False, load_in_4bit: bool = False) -> AutoModelForCausalLM:
        """
        Charge le modèle de base.
        
        Args:
            load_in_8bit: Charger en quantification 8-bit
            load_in_4bit: Charger en quantification 4-bit
        """
        self.logger.info(f"Chargement du modèle: {self.model_config['name']}")
        
        # Configuration pour la quantification
        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
        elif load_in_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
        
        # Charger le modèle
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_config['name'],
            cache_dir=self.model_config.get('cache_dir'),
            quantization_config=quantization_config,
            device_map="auto" if quantization_config else None,
            torch_dtype=torch.float16 if not quantization_config else None,
            trust_remote_code=True
        )
        
        # Déplacer sur le device si pas de quantification
        if quantization_config is None:
            self.model = self.model.to(self.device)
        
        # Redimensionner les embeddings si nécessaire
        if len(self.tokenizer) > self.model.config.vocab_size:
            self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Informations sur le modèle
        params_info = count_parameters(self.model)
        self.logger.info(f"Modèle chargé:")
        self.logger.info(f"  Paramètres totaux: {params_info['total_parameters']:,}")
        self.logger.info(f"  Device: {self.device}")
        
        return self.model
    
    def setup_lora(self) -> None:
        """Configure LoRA pour le fine-tuning efficace."""
        if not self.lora_config['enabled']:
            self.logger.info("LoRA désactivé")
            return
        
        self.logger.info("Configuration de LoRA")
        
        # Configuration LoRA
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.lora_config['r'],
            lora_alpha=self.lora_config['alpha'],
            lora_dropout=self.lora_config['dropout'],
            target_modules=self.lora_config['target_modules'],
            bias="none"
        )
        
        # Appliquer LoRA au modèle
        self.model = get_peft_model(self.model, peft_config)
        
        # Informations après LoRA
        params_info = count_parameters(self.model)
        self.logger.info(f"LoRA configuré:")
        self.logger.info(f"  Paramètres totaux: {params_info['total_parameters']:,}")
        self.logger.info(f"  Paramètres entraînables: {params_info['trainable_parameters']:,}")
        self.logger.info(f"  Pourcentage entraînable: {params_info['percentage_trainable']:.2f}%")
    
    def prepare_model_for_training(self, load_in_8bit: bool = False, load_in_4bit: bool = False):
        """
        Prépare le modèle complet pour l'entraînement.
        
        Args:
            load_in_8bit: Utiliser la quantification 8-bit
            load_in_4bit: Utiliser la quantification 4-bit
        """
        # 1. Charger le tokenizer
        self.load_tokenizer()
        
        # 2. Charger le modèle
        self.load_model(load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit)
        
        # 3. Configurer LoRA
        self.setup_lora()
        
        # 4. Préparer pour l'entraînement
        if hasattr(self.model, 'enable_input_require_grads'):
            self.model.enable_input_require_grads()
        
        return self.model, self.tokenizer
    
    def save_model(self, output_dir: str, save_tokenizer: bool = True) -> None:
        """
        Sauvegarde le modèle fine-tuné.
        
        Args:
            output_dir: Répertoire de sortie
            save_tokenizer: Sauvegarder aussi le tokenizer
        """
        self.logger.info(f"Sauvegarde du modèle dans {output_dir}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Sauvegarder le modèle
        if hasattr(self.model, 'save_pretrained'):
            self.model.save_pretrained(output_dir)
        else:
            # Pour les modèles PEFT
            self.model.save_pretrained(output_dir)
        
        # Sauvegarder le tokenizer
        if save_tokenizer and self.tokenizer:
            self.tokenizer.save_pretrained(output_dir)
        
        # Sauvegarder la configuration
        config_path = os.path.join(output_dir, "training_config.yaml")
        from .utils import save_config
        save_config(self.config, config_path)
        
        self.logger.info("Modèle sauvegardé avec succès")
    
    def load_finetuned_model(self, model_path: str, base_model_name: Optional[str] = None):
        """
        Charge un modèle fine-tuné.
        
        Args:
            model_path: Chemin vers le modèle fine-tuné
            base_model_name: Nom du modèle de base (si différent de la config)
        """
        if base_model_name is None:
            base_model_name = self.model_config['name']
        
        self.logger.info(f"Chargement du modèle fine-tuné depuis {model_path}")
        
        # Charger le tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Charger le modèle de base
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Charger les adaptateurs LoRA
        self.model = PeftModel.from_pretrained(base_model, model_path)
        
        return self.model, self.tokenizer
    
    def get_model_info(self) -> Dict[str, Any]:
        """Retourne des informations sur le modèle actuel."""
        if self.model is None:
            return {"error": "Aucun modèle chargé"}
        
        info = {
            "model_name": self.model_config['name'],
            "model_type": type(self.model).__name__,
            "device": str(self.device)
        }
        
        # Informations sur les paramètres
        params_info = count_parameters(self.model)
        info.update(params_info)
        
        # Informations LoRA
        if hasattr(self.model, 'peft_config'):
            info["lora_enabled"] = True
            info["lora_config"] = self.lora_config
        else:
            info["lora_enabled"] = False
        
        # Informations sur le tokenizer
        if self.tokenizer:
            info["vocab_size"] = len(self.tokenizer)
            info["pad_token"] = self.tokenizer.pad_token
            info["eos_token"] = self.tokenizer.eos_token
        
        return info
