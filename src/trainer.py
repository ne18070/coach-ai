"""
Classe d'entraînement pour le fine-tuning de LLM
"""
import os
import torch
from typing import Dict, Any, Optional
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from datasets import DatasetDict
import wandb
import logging

from .utils import setup_logging, EarlyStoppingCallback as CustomEarlyStoppingCallback
from .model_manager import ModelManager
from .data_processor import DataProcessor

logger = setup_logging()

class LLMTrainer:
    """Classe principale pour l'entraînement de LLM."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.training_config = config['training']
        self.logger = logging.getLogger(__name__)
        
        # Initialiser les composants
        self.model_manager = ModelManager(config)
        self.data_processor = None
        self.trainer = None
        self.model = None
        self.tokenizer = None
    
    def setup_wandb(self) -> None:
        """Configure Weights & Biases pour le monitoring."""
        wandb_config = self.config.get('wandb', {})
        
        if 'wandb' in self.training_config.get('report_to', []):
            wandb.init(
                project=wandb_config.get('project', 'llm-finetuning'),
                entity=wandb_config.get('entity'),
                name=wandb_config.get('name'),
                tags=wandb_config.get('tags', []),
                config=self.config
            )
            self.logger.info("W&B initialisé")
    
    def prepare_data(self) -> DatasetDict:
        """Prépare les données d'entraînement."""
        self.logger.info("Préparation des données")
        
        # Charger les données depuis les fichiers configurés
        data_config = self.config['data']
        
        # Initialiser le processeur de données avec le tokenizer
        self.data_processor = DataProcessor(self.tokenizer)
        
        # Charger les datasets
        datasets = {}
        
        if 'train_file' in data_config:
            train_dataset = self.data_processor.load_data(data_config['train_file'])
            datasets['train'] = train_dataset
        
        if 'validation_file' in data_config:
            val_dataset = self.data_processor.load_data(data_config['validation_file'])
            datasets['validation'] = val_dataset
        
        if 'test_file' in data_config:
            test_dataset = self.data_processor.load_data(data_config['test_file'])
            datasets['test'] = test_dataset
        
        dataset_dict = DatasetDict(datasets)
        
        # Tokeniser les datasets
        max_length = self.config['model'].get('max_length', 512)
        for split_name in dataset_dict:
            dataset_dict[split_name] = self.data_processor.tokenize_dataset(
                dataset_dict[split_name],
                max_length=max_length
            )
        
        # Limiter le nombre d'échantillons si spécifié
        if data_config.get('max_train_samples') and 'train' in dataset_dict:
            dataset_dict['train'] = dataset_dict['train'].select(
                range(min(len(dataset_dict['train']), data_config['max_train_samples']))
            )
        
        if data_config.get('max_eval_samples') and 'validation' in dataset_dict:
            dataset_dict['validation'] = dataset_dict['validation'].select(
                range(min(len(dataset_dict['validation']), data_config['max_eval_samples']))
            )
        
        self.logger.info("Données préparées:")
        for split_name, dataset in dataset_dict.items():
            self.logger.info(f"  {split_name}: {len(dataset)} exemples")
        
        return dataset_dict
    
    def create_training_arguments(self) -> TrainingArguments:
        """Crée les arguments d'entraînement."""
        args = TrainingArguments(
            output_dir=self.training_config['output_dir'],
            num_train_epochs=self.training_config['num_train_epochs'],
            per_device_train_batch_size=self.training_config['per_device_train_batch_size'],
            per_device_eval_batch_size=self.training_config['per_device_eval_batch_size'],
            gradient_accumulation_steps=self.training_config['gradient_accumulation_steps'],
            learning_rate=self.training_config['learning_rate'],
            weight_decay=self.training_config['weight_decay'],
            warmup_ratio=self.training_config['warmup_ratio'],
            lr_scheduler_type=self.training_config['lr_scheduler_type'],
            
            # Sauvegarde
            save_strategy=self.training_config['save_strategy'],
            save_steps=self.training_config['save_steps'],
            save_total_limit=self.training_config['save_total_limit'],
            
            # Évaluation
            evaluation_strategy=self.training_config['evaluation_strategy'],
            eval_steps=self.training_config['eval_steps'],
            load_best_model_at_end=self.training_config['load_best_model_at_end'],
            metric_for_best_model=self.training_config['metric_for_best_model'],
            greater_is_better=self.training_config['greater_is_better'],
            
            # Logging
            logging_dir=self.training_config['logging_dir'],
            logging_steps=self.training_config['logging_steps'],
            report_to=self.training_config.get('report_to', []),
            
            # Optimisations
            fp16=self.training_config.get('fp16', False),
            bf16=self.training_config.get('bf16', False),
            dataloader_num_workers=self.training_config.get('dataloader_num_workers', 0),
            remove_unused_columns=self.training_config.get('remove_unused_columns', False),
            
            # Autres
            seed=self.config.get('system', {}).get('seed', 42),
            data_seed=self.config.get('system', {}).get('seed', 42),
        )
        
        return args
    
    def create_data_collator(self):
        """Crée le data collator pour l'entraînement."""
        return DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Pour les modèles causaux
            pad_to_multiple_of=8 if self.training_config.get('fp16') else None
        )
    
    def setup_callbacks(self):
        """Configure les callbacks d'entraînement."""
        callbacks = []
        
        # Early stopping
        early_stopping_config = self.training_config.get('early_stopping', {})
        if early_stopping_config.get('enabled', False):
            callback = EarlyStoppingCallback(
                early_stopping_patience=early_stopping_config.get('patience', 3),
                early_stopping_threshold=early_stopping_config.get('threshold', 0.001)
            )
            callbacks.append(callback)
        
        return callbacks
    
    def train(self, resume_from_checkpoint: Optional[str] = None) -> None:
        """
        Lance l'entraînement.
        
        Args:
            resume_from_checkpoint: Chemin vers un checkpoint pour reprendre l'entraînement
        """
        self.logger.info("Début de l'entraînement")
        
        # 1. Préparer le modèle
        quantization = self.config.get('system', {}).get('mixed_precision') in ['8bit', '4bit']
        load_in_8bit = self.config.get('system', {}).get('mixed_precision') == '8bit'
        load_in_4bit = self.config.get('system', {}).get('mixed_precision') == '4bit'
        
        self.model, self.tokenizer = self.model_manager.prepare_model_for_training(
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit
        )
        
        # 2. Préparer les données
        dataset_dict = self.prepare_data()
        
        # 3. Configurer W&B
        self.setup_wandb()
        
        # 4. Créer les arguments d'entraînement
        training_args = self.create_training_arguments()
        
        # 5. Créer le data collator
        data_collator = self.create_data_collator()
        
        # 6. Configurer les callbacks
        callbacks = self.setup_callbacks()
        
        # 7. Créer le trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset_dict.get('train'),
            eval_dataset=dataset_dict.get('validation'),
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            callbacks=callbacks,
        )
        
        # 8. Afficher le résumé
        from .utils import print_training_summary
        print_training_summary(self.config, self.model, self.tokenizer)
        
        # 9. Lancer l'entraînement
        try:
            self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
            self.logger.info("Entraînement terminé avec succès")
        except Exception as e:
            self.logger.error(f"Erreur durante l'entraînement: {e}")
            raise
        
        # 10. Sauvegarder le modèle final
        self.save_model()
        
        # 11. Finaliser W&B
        if 'wandb' in self.training_config.get('report_to', []):
            wandb.finish()
    
    def evaluate(self, dataset_name: str = 'validation') -> Dict[str, float]:
        """
        Évalue le modèle sur un dataset.
        
        Args:
            dataset_name: Nom du dataset à utiliser pour l'évaluation
        
        Returns:
            Métriques d'évaluation
        """
        if self.trainer is None:
            raise ValueError("Le trainer n'est pas initialisé. Lancez d'abord l'entraînement.")
        
        self.logger.info(f"Évaluation sur le dataset {dataset_name}")
        
        # Préparer les données si nécessaire
        if not hasattr(self, 'dataset_dict'):
            self.dataset_dict = self.prepare_data()
        
        # Évaluer
        if dataset_name in self.dataset_dict:
            eval_results = self.trainer.evaluate(eval_dataset=self.dataset_dict[dataset_name])
            self.logger.info(f"Résultats d'évaluation: {eval_results}")
            return eval_results
        else:
            raise ValueError(f"Dataset '{dataset_name}' non trouvé")
    
    def save_model(self, output_dir: Optional[str] = None) -> None:
        """
        Sauvegarde le modèle entraîné.
        
        Args:
            output_dir: Répertoire de sortie (utilise la config par défaut si None)
        """
        if output_dir is None:
            output_dir = self.training_config['output_dir']
        
        self.model_manager.save_model(output_dir)
        
        # Sauvegarder les métriques d'entraînement si disponibles
        if self.trainer and hasattr(self.trainer.state, 'log_history'):
            import json
            metrics_path = os.path.join(output_dir, 'training_metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(self.trainer.state.log_history, f, indent=2)
    
    def get_training_state(self) -> Dict[str, Any]:
        """Retourne l'état actuel de l'entraînement."""
        if self.trainer is None:
            return {"status": "not_started"}
        
        state = {
            "status": "completed" if self.trainer.state.epoch >= self.training_config['num_train_epochs'] else "in_progress",
            "current_epoch": self.trainer.state.epoch,
            "total_epochs": self.training_config['num_train_epochs'],
            "global_step": self.trainer.state.global_step,
        }
        
        if hasattr(self.trainer.state, 'log_history') and self.trainer.state.log_history:
            latest_log = self.trainer.state.log_history[-1]
            state["latest_metrics"] = latest_log
        
        return state
