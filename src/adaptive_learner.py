"""
Système d'apprentissage adaptatif qui s'entraîne automatiquement
sur tout type de données, inspiré de l'apprentissage humain.
"""

import os
import json
import torch
import logging
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import numpy as np
from datetime import datetime
from dataclasses import dataclass

from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    Trainer, TrainingArguments, DataCollatorForLanguageModeling
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType


@dataclass
class LearningPattern:
    """Structure pour capturer les patterns d'apprentissage"""
    pattern_type: str  # 'conversation', 'instruction', 'completion', 'q_a'
    input_format: str
    output_format: str
    confidence: float
    examples: List[Dict]


class DataPatternDetector:
    """Détecte automatiquement les patterns dans les données"""
    
    def __init__(self):
        self.detected_patterns = []
        self.logger = logging.getLogger(__name__)
    
    def analyze_data(self, data: Union[List, Dict, str]) -> List[LearningPattern]:
        """Analyse les données pour détecter les patterns d'apprentissage"""
        patterns = []
        
        if isinstance(data, str):
            data = self._load_data_from_path(data)
        
        if isinstance(data, list):
            patterns.extend(self._detect_list_patterns(data))
        elif isinstance(data, dict):
            patterns.extend(self._detect_dict_patterns(data))
        
        self.detected_patterns = patterns
        return patterns
    
    def _load_data_from_path(self, path: str) -> Any:
        """Charge les données depuis un fichier"""
        path = Path(path)
        
        if path.suffix == '.json':
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif path.suffix == '.txt':
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            raise ValueError(f"Format de fichier non supporté: {path.suffix}")
    
    def _detect_list_patterns(self, data: List) -> List[LearningPattern]:
        """Détecte les patterns dans une liste de données"""
        patterns = []
        
        if not data:
            return patterns
        
        sample = data[0]
        
        # Pattern conversation/dialogue
        if isinstance(sample, dict) and 'input' in sample and 'output' in sample:
            pattern = LearningPattern(
                pattern_type='conversation',
                input_format='input',
                output_format='output',
                confidence=0.9,
                examples=data[:3]
            )
            patterns.append(pattern)
        
        # Pattern instruction-following
        elif isinstance(sample, dict) and 'instruction' in sample and 'response' in sample:
            pattern = LearningPattern(
                pattern_type='instruction',
                input_format='instruction',
                output_format='response',
                confidence=0.9,
                examples=data[:3]
            )
            patterns.append(pattern)
        
        # Pattern question-answer
        elif isinstance(sample, dict) and 'question' in sample and 'answer' in sample:
            pattern = LearningPattern(
                pattern_type='q_a',
                input_format='question',
                output_format='answer',
                confidence=0.9,
                examples=data[:3]
            )
            patterns.append(pattern)
        
        # Pattern texte libre
        elif isinstance(sample, str):
            pattern = LearningPattern(
                pattern_type='completion',
                input_format='text',
                output_format='continuation',
                confidence=0.7,
                examples=data[:3]
            )
            patterns.append(pattern)
        
        return patterns
    
    def _detect_dict_patterns(self, data: Dict) -> List[LearningPattern]:
        """Détecte les patterns dans un dictionnaire"""
        patterns = []
        
        # Analyser la structure du dictionnaire
        keys = list(data.keys())
        
        if 'conversations' in keys or 'dialogues' in keys:
            conv_key = 'conversations' if 'conversations' in keys else 'dialogues'
            pattern = LearningPattern(
                pattern_type='conversation',
                input_format='nested_conversations',
                output_format='response',
                confidence=0.8,
                examples=[data[conv_key][:3]] if isinstance(data[conv_key], list) else [data[conv_key]]
            )
            patterns.append(pattern)
        
        return patterns


class AdaptiveLearner:
    """
    Système d'apprentissage adaptatif qui s'entraîne automatiquement
    sur tout type de données qu'on lui fournit.
    """
    
    def __init__(self, 
                 base_model: str = "microsoft/DialoGPT-small",
                 learning_rate: float = 2e-4,
                 max_seq_length: int = 512):
        self.base_model = base_model
        self.learning_rate = learning_rate
        self.max_seq_length = max_seq_length
        
        # Composants principaux
        self.pattern_detector = DataPatternDetector()
        self.tokenizer = None
        self.model = None
        self.learned_patterns = []
        
        # Configuration LoRA sera définie dynamiquement selon le modèle
        self.lora_config = None
        
        # Mémorisation des expériences
        self.memory = {
            'successful_patterns': [],
            'failed_patterns': [],
            'adaptation_history': []
        }
        
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure le logging pour suivre l'apprentissage"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/adaptive_learning.log'),
                logging.StreamHandler()
            ]
        )
    
    def observe_and_learn(self, data_source: Union[str, List, Dict]) -> Dict[str, Any]:
        """
        Observe de nouvelles données et s'adapte automatiquement.
        Comme un enfant qui observe et imite.
        """
        self.logger.info("🔍 Observation de nouvelles données...")
        
        # 1. Observer et analyser les patterns
        patterns = self.pattern_detector.analyze_data(data_source)
        self.logger.info(f"📊 {len(patterns)} patterns détectés")
        
        # 2. Initialiser ou charger le modèle si nécessaire
        if self.model is None:
            self._initialize_model()
        
        # 3. Pour chaque pattern détecté, s'adapter
        results = {}
        for i, pattern in enumerate(patterns):
            self.logger.info(f"🎯 Adaptation au pattern {i+1}: {pattern.pattern_type}")
            
            # Préparer les données d'entraînement
            training_data = self._prepare_training_data(pattern, data_source)
            
            # S'entraîner sur ce pattern
            training_result = self._adapt_to_pattern(pattern, training_data)
            results[f"pattern_{i+1}"] = training_result
            
            # Mémoriser l'expérience
            self._memorize_experience(pattern, training_result)
        
        # 4. Sauvegarder les adaptations
        self._save_adaptations()
        
        return results
    
    def _get_target_modules(self, model):
        """Détecte automatiquement les modules cibles pour LoRA selon le modèle"""
        target_modules = []
        
        # Parcourir les modules du modèle pour trouver les couches linéaires
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                # Prendre seulement le nom du module (pas le chemin complet)
                module_name = name.split('.')[-1]
                if module_name not in target_modules:
                    target_modules.append(module_name)
        
        # Configurations par défaut selon le type de modèle
        model_name = self.base_model.lower()
        
        if 'gpt' in model_name or 'distilgpt' in model_name:
            # Pour les modèles GPT et DistilGPT
            target_modules = ['c_attn', 'c_proj']
        elif 'llama' in model_name:
            target_modules = ['q_proj', 'v_proj', 'k_proj', 'o_proj']
        elif 'mistral' in model_name:
            target_modules = ['q_proj', 'v_proj', 'k_proj', 'o_proj']
        elif 'bert' in model_name:
            target_modules = ['query', 'value', 'key']
        else:
            # Fallback : chercher les modules linéaires courants
            common_modules = ['c_attn', 'c_proj', 'q_proj', 'v_proj', 'k_proj', 'o_proj', 'query', 'value', 'key']
            found_modules = []
            
            for name, module in model.named_modules():
                module_name = name.split('.')[-1]
                if module_name in common_modules and module_name not in found_modules:
                    found_modules.append(module_name)
            
            target_modules = found_modules if found_modules else ['c_attn']
        
        self.logger.info(f"🎯 Modules cibles détectés pour LoRA: {target_modules}")
        return target_modules

    def _initialize_model(self):
        """Initialise le modèle de base"""
        self.logger.info(f"🚀 Initialisation du modèle: {self.base_model}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
            
            # Utiliser CPU pour éviter les problèmes MPS lors de l'inférence
            device_map = None
            torch_dtype = torch.float32  # Utiliser float32 pour la compatibilité
            
            if torch.cuda.is_available():
                device_map = "auto"
                torch_dtype = torch.float16
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                torch_dtype=torch_dtype,
                device_map=device_map
            )
            
            # Ajouter un token de padding si nécessaire
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Détecter automatiquement les modules cibles pour LoRA
            target_modules = self._get_target_modules(self.model)
            
            # Créer la configuration LoRA adaptée au modèle
            self.lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=8,
                lora_alpha=16,
                lora_dropout=0.1,
                target_modules=target_modules
            )
            
            # Appliquer LoRA pour l'adaptation efficace
            self.model = get_peft_model(self.model, self.lora_config)
            
            self.logger.info("✅ Modèle initialisé avec succès")
            
        except Exception as e:
            self.logger.error(f"❌ Erreur lors de l'initialisation: {e}")
            raise
    
    def _prepare_training_data(self, pattern: LearningPattern, data_source: Any) -> Dataset:
        """Prépare les données d'entraînement selon le pattern détecté"""
        
        if isinstance(data_source, str):
            data_source = self.pattern_detector._load_data_from_path(data_source)
        
        # Convertir selon le type de pattern
        if pattern.pattern_type == 'conversation':
            texts = self._format_conversation_data(data_source, pattern)
        elif pattern.pattern_type == 'instruction':
            texts = self._format_instruction_data(data_source, pattern)
        elif pattern.pattern_type == 'q_a':
            texts = self._format_qa_data(data_source, pattern)
        elif pattern.pattern_type == 'completion':
            texts = self._format_completion_data(data_source, pattern)
        else:
            texts = self._format_generic_data(data_source, pattern)
        
        # Tokenizer les textes
        tokenized_data = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_seq_length,
            return_tensors="pt"
        )
        
        return Dataset.from_dict(tokenized_data)
    
    def _format_conversation_data(self, data: List[Dict], pattern: LearningPattern) -> List[str]:
        """Formate les données de conversation pour l'entraînement"""
        texts = []
        for item in data:
            if isinstance(item, dict) and pattern.input_format in item and pattern.output_format in item:
                # Format: Human: input\nAssistant: output
                text = f"Human: {item[pattern.input_format]}\nAssistant: {item[pattern.output_format]}"
                texts.append(text)
        return texts
    
    def _format_instruction_data(self, data: List[Dict], pattern: LearningPattern) -> List[str]:
        """Formate les données d'instruction pour l'entraînement"""
        texts = []
        for item in data:
            if isinstance(item, dict) and pattern.input_format in item and pattern.output_format in item:
                text = f"Instruction: {item[pattern.input_format]}\nRéponse: {item[pattern.output_format]}"
                texts.append(text)
        return texts
    
    def _format_qa_data(self, data: List[Dict], pattern: LearningPattern) -> List[str]:
        """Formate les données de Q&A pour l'entraînement"""
        texts = []
        for item in data:
            if isinstance(item, dict) and pattern.input_format in item and pattern.output_format in item:
                text = f"Question: {item[pattern.input_format]}\nRéponse: {item[pattern.output_format]}"
                texts.append(text)
        return texts
    
    def _format_completion_data(self, data: List[str], pattern: LearningPattern) -> List[str]:
        """Formate les données de complétion pour l'entraînement"""
        return data if isinstance(data, list) else [str(data)]
    
    def _format_generic_data(self, data: Any, pattern: LearningPattern) -> List[str]:
        """Formate les données génériques"""
        if isinstance(data, list):
            return [str(item) for item in data]
        else:
            return [str(data)]
    
    def _adapt_to_pattern(self, pattern: LearningPattern, training_data: Dataset) -> Dict[str, Any]:
        """S'adapte à un pattern spécifique par entraînement"""
        
        # Configuration d'entraînement adaptative
        training_args = TrainingArguments(
            output_dir=f"./models/adaptive_checkpoints/{pattern.pattern_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=100,
            learning_rate=self.learning_rate,
            save_steps=500,
            save_total_limit=2,
            logging_steps=50,
            logging_dir='./logs',
            remove_unused_columns=False,
            push_to_hub=False,
            report_to=None,
        )
        
        # Data collator pour le language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Créer le trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=training_data,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        
        try:
            # Entraînement adaptatif
            self.logger.info(f"🎓 Début de l'apprentissage du pattern {pattern.pattern_type}")
            train_result = trainer.train()
            
            # Sauvegarder l'adaptation
            trainer.save_model()
            
            self.logger.info(f"✅ Apprentissage terminé pour {pattern.pattern_type}")
            
            return {
                'pattern_type': pattern.pattern_type,
                'training_loss': train_result.training_loss,
                'num_examples': len(training_data),
                'status': 'success'
            }
            
        except Exception as e:
            self.logger.error(f"❌ Erreur lors de l'apprentissage: {e}")
            return {
                'pattern_type': pattern.pattern_type,
                'error': str(e),
                'status': 'failed'
            }
    
    def _memorize_experience(self, pattern: LearningPattern, result: Dict[str, Any]):
        """Mémorise l'expérience d'apprentissage pour améliorer les futures adaptations"""
        experience = {
            'timestamp': datetime.now().isoformat(),
            'pattern': pattern,
            'result': result
        }
        
        if result['status'] == 'success':
            self.memory['successful_patterns'].append(experience)
        else:
            self.memory['failed_patterns'].append(experience)
        
        self.memory['adaptation_history'].append(experience)
    
    def _save_adaptations(self):
        """Sauvegarde les adaptations et la mémoire"""
        memory_path = "models/adaptive_memory.json"
        os.makedirs(os.path.dirname(memory_path), exist_ok=True)
        
        # Convertir les objets non-sérialisables
        serializable_memory = {
            'successful_patterns': len(self.memory['successful_patterns']),
            'failed_patterns': len(self.memory['failed_patterns']),
            'total_adaptations': len(self.memory['adaptation_history']),
            'last_adaptation': datetime.now().isoformat()
        }
        
        with open(memory_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_memory, f, indent=2, ensure_ascii=False)
    
    def generate_response(self, input_text: str, max_length: int = 200) -> str:
        """Génère une réponse en utilisant les adaptations apprises"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Le modèle n'est pas initialisé. Utilisez observe_and_learn() d'abord.")
        
        try:
            # Préparer l'entrée
            inputs = self.tokenizer.encode(input_text, return_tensors="pt")
            
            # S'assurer que les inputs sont sur le même device que le modèle
            device = next(self.model.parameters()).device
            inputs = inputs.to(device)
            
            # Créer un attention mask
            attention_mask = torch.ones_like(inputs)
            
            # Générer
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    attention_mask=attention_mask,
                    max_length=min(max_length, inputs.shape[1] + 50),  # Limite raisonnable
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Décoder la réponse
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Retourner seulement la partie générée
            return response[len(input_text):].strip()
            
        except Exception as e:
            self.logger.error(f"❌ Erreur lors de la génération: {e}")
            return f"Erreur lors de la génération: {str(e)}"
    
    def continuous_learning_mode(self, data_directory: str, check_interval: int = 300):
        """
        Mode d'apprentissage continu - surveille un dossier pour de nouvelles données
        et s'adapte automatiquement
        """
        import time
        
        self.logger.info(f"🔄 Mode d'apprentissage continu activé - surveillance: {data_directory}")
        
        processed_files = set()
        
        while True:
            try:
                # Vérifier les nouveaux fichiers
                data_path = Path(data_directory)
                if not data_path.exists():
                    data_path.mkdir(parents=True, exist_ok=True)
                
                current_files = set(data_path.glob("*.json")) | set(data_path.glob("*.txt"))
                new_files = current_files - processed_files
                
                for file_path in new_files:
                    self.logger.info(f"📥 Nouveau fichier détecté: {file_path}")
                    try:
                        self.observe_and_learn(str(file_path))
                        processed_files.add(file_path)
                        self.logger.info(f"✅ Apprentissage terminé pour: {file_path}")
                    except Exception as e:
                        self.logger.error(f"❌ Erreur lors du traitement de {file_path}: {e}")
                
                time.sleep(check_interval)
                
            except KeyboardInterrupt:
                self.logger.info("🛑 Arrêt du mode d'apprentissage continu")
                break
            except Exception as e:
                self.logger.error(f"❌ Erreur dans l'apprentissage continu: {e}")
                time.sleep(check_interval)


if __name__ == "__main__":
    # Exemple d'utilisation
    learner = AdaptiveLearner()
    
    # Observer et apprendre des données
    result = learner.observe_and_learn("data/raw/sample_conversations.json")
    print("Résultat de l'apprentissage:", result)
    
    # Tester une réponse
    response = learner.generate_response("Human: Qu'est-ce que l'intelligence artificielle ?\nAssistant:")
    print("Réponse générée:", response)
