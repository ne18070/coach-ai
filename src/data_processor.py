"""
Processeur de données pour le fine-tuning de LLM
"""
import os
import json
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from datasets import Dataset, DatasetDict, load_dataset
from transformers import PreTrainedTokenizer
import logging
from pathlib import Path

from .utils import setup_logging

logger = setup_logging()

class DataProcessor:
    """Classe pour traiter et préparer les données d'entraînement."""
    
    def __init__(self, tokenizer: Optional[PreTrainedTokenizer] = None):
        self.tokenizer = tokenizer
        self.logger = logging.getLogger(__name__)
    
    def load_data(self, file_path: str, data_format: str = "auto") -> Dataset:
        """
        Charge les données depuis un fichier.
        
        Args:
            file_path: Chemin vers le fichier de données
            data_format: Format des données ("json", "csv", "txt", "auto")
        
        Returns:
            Dataset Hugging Face
        """
        if data_format == "auto":
            data_format = self._detect_format(file_path)
        
        self.logger.info(f"Chargement des données depuis {file_path} (format: {data_format})")
        
        if data_format == "json":
            return self._load_json(file_path)
        elif data_format == "csv":
            return self._load_csv(file_path)
        elif data_format == "txt":
            return self._load_txt(file_path)
        else:
            raise ValueError(f"Format non supporté: {data_format}")
    
    def _detect_format(self, file_path: str) -> str:
        """Détecte automatiquement le format du fichier."""
        extension = Path(file_path).suffix.lower()
        if extension == ".json":
            return "json"
        elif extension == ".csv":
            return "csv"
        elif extension in [".txt", ".text"]:
            return "txt"
        else:
            raise ValueError(f"Extension de fichier non reconnue: {extension}")
    
    def _load_json(self, file_path: str) -> Dataset:
        """Charge un fichier JSON."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Si c'est une liste de dictionnaires
        if isinstance(data, list):
            return Dataset.from_list(data)
        # Si c'est un dictionnaire avec des listes
        elif isinstance(data, dict):
            return Dataset.from_dict(data)
        else:
            raise ValueError("Format JSON non supporté")
    
    def _load_csv(self, file_path: str) -> Dataset:
        """Charge un fichier CSV."""
        df = pd.read_csv(file_path)
        return Dataset.from_pandas(df)
    
    def _load_txt(self, file_path: str) -> Dataset:
        """Charge un fichier texte (une ligne = un exemple)."""
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        
        return Dataset.from_dict({"text": lines})
    
    def prepare_conversational_data(self, dataset: Dataset, 
                                  input_column: str = "input",
                                  output_column: str = "output",
                                  system_column: Optional[str] = None) -> Dataset:
        """
        Prépare des données conversationnelles pour le fine-tuning.
        
        Args:
            dataset: Dataset source
            input_column: Nom de la colonne contenant les questions/inputs
            output_column: Nom de la colonne contenant les réponses
            system_column: Nom de la colonne contenant les instructions système (optionnel)
        
        Returns:
            Dataset avec colonne 'text' formatée pour l'entraînement
        """
        def format_conversation(examples):
            texts = []
            for i in range(len(examples[input_column])):
                conversation = ""
                
                # Ajouter l'instruction système si disponible
                if system_column and system_column in examples:
                    system_msg = examples[system_column][i]
                    if system_msg:
                        conversation += f"<|system|>\n{system_msg}\n\n"
                
                # Ajouter la question de l'utilisateur
                user_input = examples[input_column][i]
                conversation += f"<|user|>\n{user_input}\n\n"
                
                # Ajouter la réponse de l'assistant
                assistant_output = examples[output_column][i]
                conversation += f"<|assistant|>\n{assistant_output}<|endoftext|>"
                
                texts.append(conversation)
            
            return {"text": texts}
        
        return dataset.map(format_conversation, batched=True)
    
    def prepare_instruction_data(self, dataset: Dataset,
                               instruction_column: str = "instruction",
                               input_column: str = "input",
                               output_column: str = "output") -> Dataset:
        """
        Prépare des données d'instruction pour le fine-tuning.
        Format Alpaca-like.
        """
        def format_instruction(examples):
            texts = []
            for i in range(len(examples[instruction_column])):
                instruction = examples[instruction_column][i]
                input_text = examples.get(input_column, [""] * len(examples[instruction_column]))[i]
                output_text = examples[output_column][i]
                
                if input_text:
                    text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output_text}"
                else:
                    text = f"### Instruction:\n{instruction}\n\n### Response:\n{output_text}"
                
                texts.append(text)
            
            return {"text": texts}
        
        return dataset.map(format_instruction, batched=True)
    
    def tokenize_dataset(self, dataset: Dataset, max_length: int = 512,
                        padding: str = "max_length", truncation: bool = True) -> Dataset:
        """
        Tokenise le dataset pour l'entraînement.
        
        Args:
            dataset: Dataset à tokeniser
            max_length: Longueur maximale des séquences
            padding: Stratégie de padding
            truncation: Tronquer les séquences trop longues
        
        Returns:
            Dataset tokenisé
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer non initialisé")
        
        def tokenize_function(examples):
            # Tokeniser les textes
            tokenized = self.tokenizer(
                examples["text"],
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                return_tensors=None
            )
            
            # Pour le fine-tuning causal, les labels sont les mêmes que les input_ids
            tokenized["labels"] = tokenized["input_ids"].copy()
            
            return tokenized
        
        return dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenisation"
        )
    
    def split_dataset(self, dataset: Dataset, 
                     train_ratio: float = 0.8,
                     val_ratio: float = 0.1,
                     test_ratio: float = 0.1,
                     seed: int = 42) -> DatasetDict:
        """
        Divise le dataset en train/validation/test.
        
        Args:
            dataset: Dataset à diviser
            train_ratio: Proportion pour l'entraînement
            val_ratio: Proportion pour la validation
            test_ratio: Proportion pour le test
            seed: Graine aléatoire
        
        Returns:
            DatasetDict avec les splits
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Les ratios doivent sommer à 1.0"
        
        # Premier split: train vs (val + test)
        train_val_split = dataset.train_test_split(
            test_size=val_ratio + test_ratio,
            seed=seed
        )
        
        # Deuxième split: val vs test
        val_test_ratio = val_ratio / (val_ratio + test_ratio)
        val_test_split = train_val_split["test"].train_test_split(
            test_size=1 - val_test_ratio,
            seed=seed
        )
        
        return DatasetDict({
            "train": train_val_split["train"],
            "validation": val_test_split["train"],
            "test": val_test_split["test"]
        })
    
    def save_dataset(self, dataset: Union[Dataset, DatasetDict], 
                    output_dir: str, format: str = "json") -> None:
        """
        Sauvegarde le dataset.
        
        Args:
            dataset: Dataset à sauvegarder
            output_dir: Répertoire de sortie
            format: Format de sauvegarde ("json", "csv", "parquet")
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if isinstance(dataset, DatasetDict):
            for split_name, split_dataset in dataset.items():
                output_path = os.path.join(output_dir, f"{split_name}.{format}")
                self._save_single_dataset(split_dataset, output_path, format)
        else:
            output_path = os.path.join(output_dir, f"dataset.{format}")
            self._save_single_dataset(dataset, output_path, format)
        
        self.logger.info(f"Dataset sauvegardé dans {output_dir}")
    
    def _save_single_dataset(self, dataset: Dataset, output_path: str, format: str):
        """Sauvegarde un dataset unique."""
        if format == "json":
            dataset.to_json(output_path)
        elif format == "csv":
            dataset.to_csv(output_path)
        elif format == "parquet":
            dataset.to_parquet(output_path)
        else:
            raise ValueError(f"Format de sauvegarde non supporté: {format}")
    
    def get_dataset_stats(self, dataset: Union[Dataset, DatasetDict]) -> Dict[str, Any]:
        """Retourne des statistiques sur le dataset."""
        stats = {}
        
        if isinstance(dataset, DatasetDict):
            for split_name, split_dataset in dataset.items():
                stats[split_name] = {
                    "num_examples": len(split_dataset),
                    "columns": split_dataset.column_names,
                    "features": split_dataset.features
                }
                
                # Statistiques sur la longueur des textes si disponible
                if "text" in split_dataset.column_names:
                    lengths = [len(text.split()) for text in split_dataset["text"]]
                    stats[split_name]["text_stats"] = {
                        "avg_length": sum(lengths) / len(lengths),
                        "min_length": min(lengths),
                        "max_length": max(lengths)
                    }
        else:
            stats = {
                "num_examples": len(dataset),
                "columns": dataset.column_names,
                "features": dataset.features
            }
            
            if "text" in dataset.column_names:
                lengths = [len(text.split()) for text in dataset["text"]]
                stats["text_stats"] = {
                    "avg_length": sum(lengths) / len(lengths),
                    "min_length": min(lengths),
                    "max_length": max(lengths)
                }
        
        return stats
    
    def prepare_dataset(self, input_file: str, output_dir: str,
                       data_type: str = "conversational",
                       max_length: int = 512) -> DatasetDict:
        """
        Pipeline complet de préparation des données.
        
        Args:
            input_file: Fichier d'entrée
            output_dir: Répertoire de sortie
            data_type: Type de données ("conversational", "instruction", "raw")
            max_length: Longueur maximale pour la tokenisation
        
        Returns:
            DatasetDict préparé
        """
        # 1. Charger les données
        dataset = self.load_data(input_file)
        self.logger.info(f"Dataset chargé: {len(dataset)} exemples")
        
        # 2. Formater selon le type
        if data_type == "conversational":
            dataset = self.prepare_conversational_data(dataset)
        elif data_type == "instruction":
            dataset = self.prepare_instruction_data(dataset)
        # Pour "raw", on garde le dataset tel quel
        
        # 3. Diviser en train/val/test
        dataset_dict = self.split_dataset(dataset)
        
        # 4. Tokeniser si tokenizer disponible
        if self.tokenizer is not None:
            for split_name in dataset_dict:
                dataset_dict[split_name] = self.tokenize_dataset(
                    dataset_dict[split_name], 
                    max_length=max_length
                )
        
        # 5. Sauvegarder
        self.save_dataset(dataset_dict, output_dir)
        
        # 6. Afficher les statistiques
        stats = self.get_dataset_stats(dataset_dict)
        self.logger.info("Statistiques du dataset:")
        for split_name, split_stats in stats.items():
            self.logger.info(f"  {split_name}: {split_stats['num_examples']} exemples")
        
        return dataset_dict
