"""
Utilitaires pour le fine-tuning de LLM
"""
import os
import json
import yaml
import random
import numpy as np
import torch
from typing import Dict, Any, List, Optional
import logging
from pathlib import Path

def setup_logging(log_dir: str = "logs", level: int = logging.INFO) -> logging.Logger:
    """Configure le logging pour le projet."""
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """Charge la configuration depuis un fichier YAML."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def save_config(config: Dict[str, Any], config_path: str) -> None:
    """Sauvegarde la configuration dans un fichier YAML."""
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

def set_seed(seed: int = 42) -> None:
    """Fixe la graine aléatoire pour la reproductibilité."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_device() -> torch.device:
    """Retourne le device optimal disponible."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():  # Pour les Mac avec Apple Silicon
        return torch.device("mps")
    else:
        return torch.device("cpu")

def count_parameters(model) -> Dict[str, int]:
    """Compte les paramètres du modèle."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "percentage_trainable": (trainable_params / total_params) * 100 if total_params > 0 else 0
    }

def format_bytes(bytes_value: int) -> str:
    """Formate une valeur en bytes en unité lisible."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"

def get_model_size(model) -> str:
    """Estime la taille du modèle en mémoire."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_size = param_size + buffer_size
    return format_bytes(total_size)

def create_directories(directories: List[str]) -> None:
    """Crée une liste de répertoires."""
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def load_json(file_path: str) -> Any:
    """Charge un fichier JSON."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data: Any, file_path: str) -> None:
    """Sauvegarde des données dans un fichier JSON."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def get_gpu_memory_usage() -> Optional[Dict[str, float]]:
    """Retourne l'utilisation mémoire GPU si disponible."""
    if not torch.cuda.is_available():
        return None
    
    memory_stats = {}
    for i in range(torch.cuda.device_count()):
        device = f"cuda:{i}"
        allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(i) / 1024**3   # GB
        memory_stats[device] = {
            "allocated_gb": allocated,
            "reserved_gb": reserved
        }
    
    return memory_stats

def print_training_summary(config: Dict[str, Any], model, tokenizer) -> None:
    """Affiche un résumé de la configuration d'entraînement."""
    print("=" * 50)
    print("RÉSUMÉ DE L'ENTRAÎNEMENT")
    print("=" * 50)
    
    print(f"Modèle: {config['model']['name']}")
    print(f"Tokenizer: {tokenizer.__class__.__name__}")
    print(f"Taille du vocabulaire: {len(tokenizer)}")
    
    params_info = count_parameters(model)
    print(f"Paramètres totaux: {params_info['total_parameters']:,}")
    print(f"Paramètres entraînables: {params_info['trainable_parameters']:,}")
    print(f"Pourcentage entraînable: {params_info['percentage_trainable']:.2f}%")
    print(f"Taille estimée du modèle: {get_model_size(model)}")
    
    print(f"Device: {get_device()}")
    
    gpu_memory = get_gpu_memory_usage()
    if gpu_memory:
        print("Mémoire GPU:")
        for device, stats in gpu_memory.items():
            print(f"  {device}: {stats['allocated_gb']:.2f}GB allouée, {stats['reserved_gb']:.2f}GB réservée")
    
    training_config = config['training']
    print(f"Époques: {training_config['num_train_epochs']}")
    print(f"Batch size (train): {training_config['per_device_train_batch_size']}")
    print(f"Batch size (eval): {training_config['per_device_eval_batch_size']}")
    print(f"Learning rate: {training_config['learning_rate']}")
    print(f"Répertoire de sortie: {training_config['output_dir']}")
    
    if config['lora']['enabled']:
        print("LoRA activé:")
        print(f"  Rang: {config['lora']['r']}")
        print(f"  Alpha: {config['lora']['alpha']}")
        print(f"  Dropout: {config['lora']['dropout']}")
    
    print("=" * 50)

class EarlyStoppingCallback:
    """Callback pour l'arrêt précoce."""
    
    def __init__(self, patience: int = 3, threshold: float = 0.001, metric: str = "eval_loss", greater_is_better: bool = False):
        self.patience = patience
        self.threshold = threshold
        self.metric = metric
        self.greater_is_better = greater_is_better
        self.best_score = None
        self.counter = 0
        self.should_stop = False
    
    def __call__(self, logs: Dict[str, float]) -> bool:
        """Retourne True si l'entraînement doit s'arrêter."""
        current_score = logs.get(self.metric)
        if current_score is None:
            return False
        
        if self.best_score is None:
            self.best_score = current_score
        elif self._is_improvement(current_score):
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True
        
        return False
    
    def _is_improvement(self, current_score: float) -> bool:
        """Vérifie si le score actuel est une amélioration."""
        if self.greater_is_better:
            return current_score > self.best_score + self.threshold
        else:
            return current_score < self.best_score - self.threshold
