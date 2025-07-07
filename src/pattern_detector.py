"""
Détecteur de patterns avancé pour l'apprentissage adaptatif.
Capable de détecter automatiquement la structure et le type de données.
"""

import json
import re
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass
import logging


@dataclass
class DataStructure:
    """Structure détectée dans les données"""
    structure_type: str  # 'conversation', 'instruction', 'qa', 'narrative', 'code', 'table'
    confidence: float
    key_fields: List[str]
    sample_data: Any
    preprocessing_steps: List[str]


class AdvancedPatternDetector:
    """Détecteur de patterns avancé pour tout type de données"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Patterns de reconnaissance
        self.conversation_indicators = ['input', 'output', 'user', 'assistant', 'human', 'ai', 'question', 'answer']
        self.instruction_indicators = ['instruction', 'task', 'prompt', 'command', 'request']
        self.response_indicators = ['response', 'reply', 'answer', 'output', 'result']
        self.code_indicators = ['code', 'function', 'class', 'def ', 'import ', 'from ', '```']
        self.narrative_indicators = ['story', 'text', 'content', 'passage', 'paragraph']
    
    def analyze_comprehensive(self, data_source: Any) -> List[DataStructure]:
        """Analyse complète des données pour détecter tous les patterns possibles"""
        structures = []
        
        if isinstance(data_source, str):
            if Path(data_source).exists():
                data = self._load_from_file(data_source)
            else:
                data = data_source
        else:
            data = data_source
        
        # Détecter la structure principale
        if isinstance(data, list):
            structures.extend(self._analyze_list_structure(data))
        elif isinstance(data, dict):
            structures.extend(self._analyze_dict_structure(data))
        elif isinstance(data, str):
            structures.extend(self._analyze_text_structure(data))
        
        # Trier par confiance
        structures.sort(key=lambda x: x.confidence, reverse=True)
        
        return structures
    
    def _load_from_file(self, file_path: str) -> Any:
        """Charge des données depuis différents formats de fichiers"""
        path = Path(file_path)
        
        try:
            if path.suffix.lower() == '.json':
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            elif path.suffix.lower() == '.csv':
                return pd.read_csv(path).to_dict('records')
            elif path.suffix.lower() in ['.txt', '.md']:
                with open(path, 'r', encoding='utf-8') as f:
                    return f.read()
            else:
                # Essayer de lire comme texte
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Essayer de parser comme JSON
                    try:
                        return json.loads(content)
                    except:
                        return content
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement de {file_path}: {e}")
            return None
    
    def _analyze_list_structure(self, data: List) -> List[DataStructure]:
        """Analyse une structure de liste"""
        structures = []
        
        if not data:
            return structures
        
        # Échantillonner les premiers éléments
        sample_size = min(10, len(data))
        sample = data[:sample_size]
        
        # Analyser le premier élément pour déterminer la structure
        first_item = sample[0]
        
        if isinstance(first_item, dict):
            structures.extend(self._analyze_dict_list(sample))
        elif isinstance(first_item, str):
            structures.extend(self._analyze_string_list(sample))
        elif isinstance(first_item, list):
            structures.extend(self._analyze_nested_list(sample))
        
        return structures
    
    def _analyze_dict_list(self, sample: List[Dict]) -> List[DataStructure]:
        """Analyse une liste de dictionnaires"""
        structures = []
        
        # Collecter toutes les clés
        all_keys = set()
        for item in sample:
            if isinstance(item, dict):
                all_keys.update(item.keys())
        
        all_keys = list(all_keys)
        
        # Détecter le pattern de conversation
        conv_score = self._calculate_conversation_score(all_keys)
        if conv_score > 0.7:
            input_key, output_key = self._find_input_output_keys(all_keys)
            structures.append(DataStructure(
                structure_type='conversation',
                confidence=conv_score,
                key_fields=[input_key, output_key],
                sample_data=sample[:3],
                preprocessing_steps=['format_conversation']
            ))
        
        # Détecter le pattern d'instruction
        inst_score = self._calculate_instruction_score(all_keys)
        if inst_score > 0.7:
            inst_key, resp_key = self._find_instruction_response_keys(all_keys)
            structures.append(DataStructure(
                structure_type='instruction',
                confidence=inst_score,
                key_fields=[inst_key, resp_key],
                sample_data=sample[:3],
                preprocessing_steps=['format_instruction']
            ))
        
        # Détecter le pattern de code
        code_score = self._calculate_code_score(all_keys, sample)
        if code_score > 0.6:
            structures.append(DataStructure(
                structure_type='code',
                confidence=code_score,
                key_fields=all_keys,
                sample_data=sample[:3],
                preprocessing_steps=['format_code', 'extract_code_blocks']
            ))
        
        # Détecter le pattern tabulaire
        table_score = self._calculate_table_score(all_keys, sample)
        if table_score > 0.8:
            structures.append(DataStructure(
                structure_type='table',
                confidence=table_score,
                key_fields=all_keys,
                sample_data=sample[:3],
                preprocessing_steps=['format_table_to_text']
            ))
        
        return structures
    
    def _analyze_string_list(self, sample: List[str]) -> List[DataStructure]:
        """Analyse une liste de chaînes"""
        structures = []
        
        # Analyser le contenu des chaînes
        combined_text = ' '.join(sample[:5])
        
        # Détecter du code
        if any(indicator in combined_text for indicator in self.code_indicators):
            structures.append(DataStructure(
                structure_type='code',
                confidence=0.8,
                key_fields=['text'],
                sample_data=sample[:3],
                preprocessing_steps=['clean_code', 'extract_functions']
            ))
        
        # Détecter des conversations formatées en texte
        if re.search(r'(user|human|assistant|ai):', combined_text.lower()):
            structures.append(DataStructure(
                structure_type='conversation',
                confidence=0.7,
                key_fields=['text'],
                sample_data=sample[:3],
                preprocessing_steps=['parse_text_conversation']
            ))
        
        # Pattern narratif par défaut
        structures.append(DataStructure(
            structure_type='narrative',
            confidence=0.6,
            key_fields=['text'],
            sample_data=sample[:3],
            preprocessing_steps=['chunk_text', 'create_completions']
        ))
        
        return structures
    
    def _analyze_text_structure(self, text: str) -> List[DataStructure]:
        """Analyse un texte libre"""
        structures = []
        
        # Détecter du code
        code_patterns = [r'def\s+\w+', r'class\s+\w+', r'import\s+\w+', r'```.*?```']
        code_score = sum(1 for pattern in code_patterns if re.search(pattern, text, re.DOTALL)) / len(code_patterns)
        
        if code_score > 0.3:
            structures.append(DataStructure(
                structure_type='code',
                confidence=min(0.9, code_score * 2),
                key_fields=['content'],
                sample_data=text[:500],
                preprocessing_steps=['extract_code_snippets', 'create_code_examples']
            ))
        
        # Détecter des conversations
        conv_patterns = [r'(user|human):\s*(.+?)\n(assistant|ai):\s*(.+?)(?=\n|$)', 
                        r'Q:\s*(.+?)\nA:\s*(.+?)(?=\n|$)']
        conv_matches = sum(len(re.findall(pattern, text, re.IGNORECASE | re.DOTALL)) for pattern in conv_patterns)
        
        if conv_matches > 0:
            structures.append(DataStructure(
                structure_type='conversation',
                confidence=min(0.9, conv_matches * 0.2),
                key_fields=['content'],
                sample_data=text[:500],
                preprocessing_steps=['extract_conversations', 'format_qa_pairs']
            ))
        
        # Pattern narratif
        structures.append(DataStructure(
            structure_type='narrative',
            confidence=0.5,
            key_fields=['content'],
            sample_data=text[:500],
            preprocessing_steps=['chunk_narrative', 'create_text_completions']
        ))
        
        return structures
    
    def _calculate_conversation_score(self, keys: List[str]) -> float:
        """Calcule le score de probabilité d'être des données de conversation"""
        score = 0.0
        key_str = ' '.join(keys).lower()
        
        # Rechercher des indicateurs de conversation
        for indicator in self.conversation_indicators:
            if indicator in key_str:
                score += 0.2
        
        # Bonus si on a une paire input/output évidente
        if any(inp in key_str for inp in ['input', 'question', 'user']) and \
           any(out in key_str for out in ['output', 'answer', 'assistant']):
            score += 0.4
        
        return min(1.0, score)
    
    def _calculate_instruction_score(self, keys: List[str]) -> float:
        """Calcule le score pour les données d'instruction"""
        score = 0.0
        key_str = ' '.join(keys).lower()
        
        for indicator in self.instruction_indicators:
            if indicator in key_str:
                score += 0.25
        
        for indicator in self.response_indicators:
            if indicator in key_str:
                score += 0.25
        
        return min(1.0, score)
    
    def _calculate_code_score(self, keys: List[str], sample: List[Dict]) -> float:
        """Calcule le score pour les données de code"""
        score = 0.0
        
        # Vérifier les clés
        key_str = ' '.join(keys).lower()
        for indicator in self.code_indicators:
            if indicator in key_str:
                score += 0.2
        
        # Vérifier le contenu
        for item in sample[:3]:
            if isinstance(item, dict):
                content = ' '.join(str(v) for v in item.values()).lower()
                for indicator in self.code_indicators:
                    if indicator in content:
                        score += 0.1
        
        return min(1.0, score)
    
    def _calculate_table_score(self, keys: List[str], sample: List[Dict]) -> float:
        """Calcule le score pour les données tabulaires"""
        # Si tous les éléments ont les mêmes clés et sont simples
        if len(sample) < 2:
            return 0.0
        
        first_keys = set(sample[0].keys()) if isinstance(sample[0], dict) else set()
        
        # Vérifier la consistance des clés
        consistent_keys = all(
            set(item.keys()) == first_keys 
            for item in sample 
            if isinstance(item, dict)
        )
        
        if consistent_keys and len(first_keys) >= 2:
            # Vérifier que les valeurs sont simples (pas d'objets complexes)
            simple_values = all(
                all(isinstance(v, (str, int, float, bool)) for v in item.values())
                for item in sample
                if isinstance(item, dict)
            )
            
            return 0.9 if simple_values else 0.5
        
        return 0.0
    
    def _find_input_output_keys(self, keys: List[str]) -> Tuple[str, str]:
        """Trouve les clés d'entrée et de sortie pour les conversations"""
        input_key = None
        output_key = None
        
        key_str_lower = [k.lower() for k in keys]
        
        # Rechercher les clés d'entrée
        for i, key in enumerate(key_str_lower):
            if any(inp in key for inp in ['input', 'question', 'user', 'human']):
                input_key = keys[i]
                break
        
        # Rechercher les clés de sortie
        for i, key in enumerate(key_str_lower):
            if any(out in key for out in ['output', 'answer', 'assistant', 'ai', 'response']):
                output_key = keys[i]
                break
        
        # Valeurs par défaut si pas trouvées
        if not input_key and len(keys) >= 2:
            input_key = keys[0]
        if not output_key and len(keys) >= 2:
            output_key = keys[1]
        
        return input_key or 'input', output_key or 'output'
    
    def _find_instruction_response_keys(self, keys: List[str]) -> Tuple[str, str]:
        """Trouve les clés d'instruction et de réponse"""
        inst_key = None
        resp_key = None
        
        key_str_lower = [k.lower() for k in keys]
        
        for i, key in enumerate(key_str_lower):
            if any(inst in key for inst in self.instruction_indicators):
                inst_key = keys[i]
                break
        
        for i, key in enumerate(key_str_lower):
            if any(resp in key for resp in self.response_indicators):
                resp_key = keys[i]
                break
        
        return inst_key or keys[0], resp_key or keys[-1]
    
    def _analyze_nested_list(self, sample: List[List]) -> List[DataStructure]:
        """Analyse des listes imbriquées"""
        structures = []
        
        # Pour l'instant, traiter comme des séquences
        structures.append(DataStructure(
            structure_type='sequence',
            confidence=0.6,
            key_fields=['sequences'],
            sample_data=sample[:3],
            preprocessing_steps=['flatten_sequences', 'create_sequence_pairs']
        ))
        
        return structures
    
    def _analyze_dict_structure(self, data: Dict) -> List[DataStructure]:
        """Analyse une structure de dictionnaire"""
        structures = []
        
        # Si le dictionnaire contient des listes, analyser les listes
        for key, value in data.items():
            if isinstance(value, list) and value:
                sub_structures = self._analyze_list_structure(value)
                for struct in sub_structures:
                    struct.key_fields = [key] + struct.key_fields
                    structures.extend([struct])
        
        # Si pas de sous-structures trouvées, traiter comme données simples
        if not structures:
            structures.append(DataStructure(
                structure_type='key_value',
                confidence=0.5,
                key_fields=list(data.keys()),
                sample_data=data,
                preprocessing_steps=['format_key_value_pairs']
            ))
        
        return structures


# Fonctions utilitaires pour le preprocessing
class DataPreprocessor:
    """Préprocesseur de données selon les patterns détectés"""
    
    @staticmethod
    def format_conversation(data: List[Dict], input_key: str, output_key: str) -> List[str]:
        """Formate les données de conversation"""
        formatted = []
        for item in data:
            if input_key in item and output_key in item:
                text = f"Human: {item[input_key]}\nAssistant: {item[output_key]}"
                formatted.append(text)
        return formatted
    
    @staticmethod
    def format_instruction(data: List[Dict], inst_key: str, resp_key: str) -> List[str]:
        """Formate les données d'instruction"""
        formatted = []
        for item in data:
            if inst_key in item and resp_key in item:
                text = f"Instruction: {item[inst_key]}\nRéponse: {item[resp_key]}"
                formatted.append(text)
        return formatted
    
    @staticmethod
    def chunk_narrative(text: str, chunk_size: int = 512) -> List[str]:
        """Découpe un texte narratif en chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        
        return chunks
    
    @staticmethod
    def extract_conversations(text: str) -> List[str]:
        """Extrait les conversations d'un texte"""
        patterns = [
            r'(Human|User):\s*(.+?)\n(Assistant|AI):\s*(.+?)(?=\n(?:Human|User):|$)',
            r'Q:\s*(.+?)\nA:\s*(.+?)(?=\nQ:|$)'
        ]
        
        conversations = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                if len(match) >= 2:
                    conv = f"Human: {match[1].strip()}\nAssistant: {match[3].strip() if len(match) > 3 else match[1].strip()}"
                    conversations.append(conv)
        
        return conversations
