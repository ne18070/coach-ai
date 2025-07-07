"""
🌐 Collecteur de Données Intelligent
Système automatique de collecte de données sur internet
pour nourrir l'apprentissage adaptatif de l'IA
"""

import asyncio
import aiohttp
import requests
import json
import re
import time
import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse, quote
import xml.etree.ElementTree as ET

import feedparser
from bs4 import BeautifulSoup
import wikipedia
from datasets import load_dataset
import arxiv
from googlesearch import search


@dataclass
class DataSource:
    """Configuration d'une source de données"""
    name: str
    url: str
    data_type: str  # 'conversation', 'instruction', 'code', 'text', 'qa'
    extraction_method: str  # 'rss', 'scraping', 'api', 'dataset'
    update_frequency: int  # en heures
    last_update: Optional[datetime] = None
    active: bool = True
    max_items: int = 100


@dataclass
class CollectedData:
    """Structure pour les données collectées"""
    source: str
    data_type: str
    content: Dict[str, Any]
    timestamp: datetime
    quality_score: float
    metadata: Dict[str, Any]


class IntelligentDataCollector:
    """
    Collecteur de données intelligent qui explore internet
    pour trouver des données d'apprentissage de qualité
    """
    
    def __init__(self, storage_dir: str = "data/collected"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = self._setup_logging()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        
        # Sources prédéfinies de qualité
        self.data_sources = self._initialize_sources()
        self.collected_urls = set()  # Éviter les doublons
        
    def _setup_logging(self) -> logging.Logger:
        """Configure le logging pour le collecteur"""
        logger = logging.getLogger('DataCollector')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.FileHandler('logs/data_collector.log')
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def _initialize_sources(self) -> List[DataSource]:
        """Initialise les sources de données par défaut"""
        return [
            # Sources de conversations et Q&A
            DataSource(
                name="StackOverflow", 
                url="https://stackoverflow.com/feeds",
                data_type="qa",
                extraction_method="rss",
                update_frequency=6
            ),
            DataSource(
                name="Reddit_MachineLearning",
                url="https://www.reddit.com/r/MachineLearning/new.json",
                data_type="conversation",
                extraction_method="api",
                update_frequency=4
            ),
            DataSource(
                name="ArXiv_AI",
                url="arxiv.org",
                data_type="instruction",
                extraction_method="api",
                update_frequency=24
            ),
            DataSource(
                name="Wikipedia_AI",
                url="wikipedia.org",
                data_type="text",
                extraction_method="api",
                update_frequency=48
            ),
            DataSource(
                name="HuggingFace_Datasets",
                url="huggingface.co/datasets",
                data_type="mixed",
                extraction_method="dataset",
                update_frequency=24
            ),
            DataSource(
                name="GitHub_AI_Repos",
                url="github.com/search",
                data_type="code",
                extraction_method="scraping",
                update_frequency=12
            )
        ]
    
    async def collect_all_sources(self) -> List[CollectedData]:
        """Collecte des données depuis toutes les sources actives"""
        self.logger.info("🚀 Début de la collecte automatique de données")
        
        all_data = []
        
        for source in self.data_sources:
            if not source.active:
                continue
                
            if self._should_update_source(source):
                try:
                    self.logger.info(f"📡 Collecte depuis {source.name}")
                    source_data = await self._collect_from_source(source)
                    all_data.extend(source_data)
                    source.last_update = datetime.now()
                    
                except Exception as e:
                    self.logger.error(f"❌ Erreur avec {source.name}: {e}")
                    
        self.logger.info(f"✅ Collecte terminée: {len(all_data)} éléments")
        return all_data
    
    def _should_update_source(self, source: DataSource) -> bool:
        """Détermine si une source doit être mise à jour"""
        if source.last_update is None:
            return True
            
        time_since_update = datetime.now() - source.last_update
        return time_since_update.total_seconds() > source.update_frequency * 3600
    
    async def _collect_from_source(self, source: DataSource) -> List[CollectedData]:
        """Collecte des données depuis une source spécifique"""
        method_map = {
            'rss': self._collect_rss,
            'api': self._collect_api,
            'scraping': self._collect_scraping,
            'dataset': self._collect_dataset
        }
        
        collector = method_map.get(source.extraction_method)
        if not collector:
            self.logger.warning(f"Méthode inconnue: {source.extraction_method}")
            return []
            
        return await collector(source)
    
    async def _collect_rss(self, source: DataSource) -> List[CollectedData]:
        """Collecte via flux RSS"""
        data = []
        
        try:
            feed = feedparser.parse(source.url)
            
            for entry in feed.entries[:source.max_items]:
                content = {
                    'title': entry.get('title', ''),
                    'description': entry.get('description', ''),
                    'link': entry.get('link', ''),
                    'published': entry.get('published', '')
                }
                
                # Extraction du contenu détaillé si possible
                if 'link' in entry:
                    detailed_content = await self._extract_article_content(entry.link)
                    if detailed_content:
                        content['full_content'] = detailed_content
                
                quality_score = self._calculate_quality_score(content, source.data_type)
                
                if quality_score > 0.5:  # Seuil de qualité
                    data.append(CollectedData(
                        source=source.name,
                        data_type=source.data_type,
                        content=content,
                        timestamp=datetime.now(),
                        quality_score=quality_score,
                        metadata={'feed_title': feed.feed.get('title', '')}
                    ))
                    
        except Exception as e:
            self.logger.error(f"Erreur RSS {source.name}: {e}")
            
        return data
    
    async def _collect_api(self, source: DataSource) -> List[CollectedData]:
        """Collecte via API spécialisées"""
        data = []
        
        if 'reddit' in source.name.lower():
            data.extend(await self._collect_reddit(source))
        elif 'arxiv' in source.name.lower():
            data.extend(await self._collect_arxiv(source))
        elif 'wikipedia' in source.name.lower():
            data.extend(await self._collect_wikipedia(source))
            
        return data
    
    async def _collect_reddit(self, source: DataSource) -> List[CollectedData]:
        """Collecte depuis Reddit"""
        data = []
        
        try:
            response = self.session.get(source.url)
            reddit_data = response.json()
            
            for post in reddit_data['data']['children'][:source.max_items]:
                post_data = post['data']
                
                # Formatage pour conversation
                content = {
                    'question': post_data.get('title', ''),
                    'context': post_data.get('selftext', ''),
                    'score': post_data.get('score', 0),
                    'comments_count': post_data.get('num_comments', 0)
                }
                
                # Récupération des commentaires de qualité
                if post_data.get('num_comments', 0) > 5:
                    comments = await self._get_reddit_comments(post_data['permalink'])
                    if comments:
                        content['answers'] = comments
                
                quality_score = self._calculate_quality_score(content, source.data_type)
                
                if quality_score > 0.6:
                    data.append(CollectedData(
                        source=source.name,
                        data_type=source.data_type,
                        content=content,
                        timestamp=datetime.now(),
                        quality_score=quality_score,
                        metadata={'subreddit': post_data.get('subreddit', '')}
                    ))
                    
        except Exception as e:
            self.logger.error(f"Erreur Reddit: {e}")
            
        return data
    
    async def _collect_arxiv(self, source: DataSource) -> List[CollectedData]:
        """Collecte depuis ArXiv pour des instructions de qualité"""
        data = []
        
        try:
            # Recherche d'articles récents en IA
            search_queries = [
                'artificial intelligence', 'machine learning', 
                'deep learning', 'natural language processing'
            ]
            
            for query in search_queries:
                search_results = arxiv.Search(
                    query=query,
                    max_results=source.max_items // len(search_queries),
                    sort_by=arxiv.SortCriterion.SubmittedDate
                )
                
                for paper in search_results.results():
                    # Formatage en instruction/réponse
                    content = {
                        'instruction': f"Explique l'article: {paper.title}",
                        'response': f"Résumé: {paper.summary}",
                        'title': paper.title,
                        'authors': [str(author) for author in paper.authors],
                        'url': paper.entry_id
                    }
                    
                    quality_score = self._calculate_quality_score(content, source.data_type)
                    
                    if quality_score > 0.7:
                        data.append(CollectedData(
                            source=source.name,
                            data_type=source.data_type,
                            content=content,
                            timestamp=datetime.now(),
                            quality_score=quality_score,
                            metadata={'categories': paper.categories}
                        ))
                        
        except Exception as e:
            self.logger.error(f"Erreur ArXiv: {e}")
            
        return data
    
    async def _collect_wikipedia(self, source: DataSource) -> List[CollectedData]:
        """Collecte depuis Wikipedia"""
        data = []
        
        try:
            # Sujets IA populaires
            ai_topics = [
                'Artificial intelligence', 'Machine learning', 'Deep learning',
                'Neural network', 'Natural language processing', 'Computer vision',
                'Reinforcement learning', 'Large language model'
            ]
            
            for topic in ai_topics[:source.max_items // len(ai_topics)]:
                try:
                    page = wikipedia.page(topic)
                    
                    # Découpe en sections pour créer des Q&A
                    sections = page.content.split('\n\n')
                    
                    for i, section in enumerate(sections[:5]):  # 5 premières sections
                        if len(section) > 100:  # Sections substantielles
                            content = {
                                'question': f"Qu'est-ce que {topic}?",
                                'answer': section[:1000],  # Limite de taille
                                'title': page.title,
                                'url': page.url
                            }
                            
                            quality_score = self._calculate_quality_score(content, 'qa')
                            
                            if quality_score > 0.6:
                                data.append(CollectedData(
                                    source=source.name,
                                    data_type='qa',
                                    content=content,
                                    timestamp=datetime.now(),
                                    quality_score=quality_score,
                                    metadata={'topic': topic, 'section': i}
                                ))
                                
                except Exception as e:
                    self.logger.debug(f"Page Wikipedia non trouvée: {topic}")
                    
        except Exception as e:
            self.logger.error(f"Erreur Wikipedia: {e}")
            
        return data
    
    async def _collect_dataset(self, source: DataSource) -> List[CollectedData]:
        """Collecte depuis HuggingFace Datasets"""
        data = []
        
        try:
            # Datasets populaires pour l'apprentissage
            quality_datasets = [
                'squad', 'ms_marco', 'natural_questions',
                'quac', 'coqa', 'drop', 'hotpot_qa'
            ]
            
            for dataset_name in quality_datasets:
                try:
                    dataset = load_dataset(dataset_name, split='train[:100]')  # Échantillon
                    
                    for item in dataset:
                        # Conversion en format standardisé
                        if 'question' in item and 'answers' in item:
                            content = {
                                'question': item['question'],
                                'answer': item['answers']['text'][0] if item['answers']['text'] else '',
                                'context': item.get('context', '')
                            }
                        elif 'input' in item and 'output' in item:
                            content = {
                                'input': item['input'],
                                'output': item['output']
                            }
                        else:
                            continue
                            
                        quality_score = self._calculate_quality_score(content, 'qa')
                        
                        if quality_score > 0.7:
                            data.append(CollectedData(
                                source=f"HF_{dataset_name}",
                                data_type='qa',
                                content=content,
                                timestamp=datetime.now(),
                                quality_score=quality_score,
                                metadata={'dataset': dataset_name}
                            ))
                            
                except Exception as e:
                    self.logger.debug(f"Dataset non accessible: {dataset_name}")
                    
        except Exception as e:
            self.logger.error(f"Erreur HuggingFace: {e}")
            
        return data
    
    async def _collect_scraping(self, source: DataSource) -> List[CollectedData]:
        """Collecte par scraping web intelligent"""
        data = []
        
        try:
            if 'github' in source.name.lower():
                data.extend(await self._collect_github_code(source))
                
        except Exception as e:
            self.logger.error(f"Erreur scraping {source.name}: {e}")
            
        return data
    
    async def _collect_github_code(self, source: DataSource) -> List[CollectedData]:
        """Collecte du code depuis GitHub"""
        data = []
        
        try:
            # Recherche de repos Python populaires en IA
            search_terms = [
                'python machine learning', 'pytorch tutorial',
                'tensorflow examples', 'nlp python'
            ]
            
            for term in search_terms:
                # Utilisation de l'API GitHub Search (simulation)
                search_url = f"https://api.github.com/search/repositories"
                params = {
                    'q': f'{term} language:python',
                    'sort': 'stars',
                    'order': 'desc',
                    'per_page': 10
                }
                
                response = self.session.get(search_url, params=params)
                
                if response.status_code == 200:
                    repos = response.json().get('items', [])
                    
                    for repo in repos[:5]:  # Top 5 repos
                        # Récupération d'exemples de code
                        code_content = await self._extract_code_examples(repo)
                        
                        if code_content:
                            content = {
                                'task': f"Écris du code Python pour {repo['description']}",
                                'code': code_content,
                                'repository': repo['full_name'],
                                'stars': repo['stargazers_count']
                            }
                            
                            quality_score = self._calculate_quality_score(content, 'code')
                            
                            if quality_score > 0.6:
                                data.append(CollectedData(
                                    source=source.name,
                                    data_type='code',
                                    content=content,
                                    timestamp=datetime.now(),
                                    quality_score=quality_score,
                                    metadata={'repo': repo['full_name']}
                                ))
                                
        except Exception as e:
            self.logger.error(f"Erreur GitHub: {e}")
            
        return data
    
    def _calculate_quality_score(self, content: Dict, data_type: str) -> float:
        """Calcule un score de qualité pour le contenu"""
        score = 0.5  # Score de base
        
        # Critères généraux
        text_content = ' '.join(str(v) for v in content.values() if isinstance(v, str))
        
        # Longueur appropriée
        if 50 <= len(text_content) <= 2000:
            score += 0.2
        
        # Pas de contenu vide
        if any(v for v in content.values() if v):
            score += 0.1
            
        # Critères spécifiques par type
        if data_type == 'qa':
            if 'question' in content and 'answer' in content:
                if len(content['answer']) > 20:
                    score += 0.2
                    
        elif data_type == 'code':
            if any(keyword in text_content.lower() for keyword in ['def ', 'class ', 'import ']):
                score += 0.3
                
        elif data_type == 'conversation':
            if any(keyword in text_content.lower() for keyword in ['?', 'comment', 'pourquoi']):
                score += 0.2
        
        return min(score, 1.0)
    
    async def _extract_article_content(self, url: str) -> Optional[str]:
        """Extrait le contenu principal d'un article"""
        try:
            response = self.session.get(url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Suppression des éléments indésirables
            for element in soup(['script', 'style', 'nav', 'footer', 'header']):
                element.decompose()
            
            # Extraction du texte principal
            main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
            
            if main_content:
                return main_content.get_text(strip=True)[:2000]  # Limite de taille
                
        except Exception as e:
            self.logger.debug(f"Erreur extraction {url}: {e}")
            
        return None
    
    async def _get_reddit_comments(self, permalink: str) -> List[str]:
        """Récupère les commentaires Reddit de qualité"""
        try:
            comments_url = f"https://www.reddit.com{permalink}.json"
            response = self.session.get(comments_url)
            data = response.json()
            
            comments = []
            if len(data) > 1 and 'children' in data[1]['data']:
                for comment in data[1]['data']['children'][:5]:  # Top 5 commentaires
                    if comment['kind'] == 't1':  # Type commentaire
                        comment_data = comment['data']
                        if comment_data.get('score', 0) > 5:  # Score minimum
                            comments.append(comment_data['body'])
                            
            return comments
            
        except Exception as e:
            self.logger.debug(f"Erreur commentaires Reddit: {e}")
            return []
    
    async def _extract_code_examples(self, repo: Dict) -> Optional[str]:
        """Extrait des exemples de code depuis un repo GitHub"""
        try:
            # Recherche de fichiers exemple
            contents_url = repo['contents_url'].replace('{+path}', '')
            response = self.session.get(contents_url)
            
            if response.status_code == 200:
                files = response.json()
                
                # Recherche de fichiers Python intéressants
                python_files = [f for f in files if f['name'].endswith('.py') and 
                              any(keyword in f['name'].lower() for keyword in 
                                  ['example', 'demo', 'tutorial', 'main'])]
                
                if python_files:
                    # Récupération du premier fichier trouvé
                    file_url = python_files[0]['download_url']
                    file_response = self.session.get(file_url)
                    
                    if file_response.status_code == 200:
                        return file_response.text[:1500]  # Limite de taille
                        
        except Exception as e:
            self.logger.debug(f"Erreur extraction code: {e}")
            
        return None
    
    def save_collected_data(self, data: List[CollectedData]) -> None:
        """Sauvegarde les données collectées"""
        if not data:
            return
            
        # Organisation par type de données
        by_type = {}
        for item in data:
            if item.data_type not in by_type:
                by_type[item.data_type] = []
            by_type[item.data_type].append(item)
        
        # Sauvegarde par type
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for data_type, items in by_type.items():
            filename = f"collected_{data_type}_{timestamp}.json"
            filepath = self.storage_dir / filename
            
            # Conversion en format JSON
            json_data = []
            for item in items:
                json_item = {
                    'source': item.source,
                    'content': item.content,
                    'quality_score': item.quality_score,
                    'timestamp': item.timestamp.isoformat(),
                    'metadata': item.metadata
                }
                json_data.append(json_item)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
                
            self.logger.info(f"💾 Sauvegardé: {len(items)} éléments dans {filename}")
    
    def add_custom_source(self, source: DataSource) -> None:
        """Ajoute une source personnalisée"""
        self.data_sources.append(source)
        self.logger.info(f"➕ Source ajoutée: {source.name}")
    
    def remove_source(self, source_name: str) -> None:
        """Supprime une source"""
        self.data_sources = [s for s in self.data_sources if s.name != source_name]
        self.logger.info(f"➖ Source supprimée: {source_name}")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de collecte"""
        stats = {
            'total_sources': len(self.data_sources),
            'active_sources': len([s for s in self.data_sources if s.active]),
            'sources_by_type': {},
            'last_collection': None
        }
        
        for source in self.data_sources:
            if source.data_type not in stats['sources_by_type']:
                stats['sources_by_type'][source.data_type] = 0
            stats['sources_by_type'][source.data_type] += 1
            
            if source.last_update:
                if not stats['last_collection'] or source.last_update > stats['last_collection']:
                    stats['last_collection'] = source.last_update
        
        return stats


# Interface de haut niveau
class AutoDataCollector:
    """Interface simplifiée pour la collecte automatique"""
    
    def __init__(self, storage_dir: str = "data/collected"):
        self.collector = IntelligentDataCollector(storage_dir)
        
    async def start_continuous_collection(self, interval_hours: int = 6):
        """Lance la collecte continue"""
        print("🤖 Démarrage du collecteur de données intelligent...")
        print("🌐 Exploration d'internet pour nourrir l'IA...")
        
        while True:
            try:
                # Collecte depuis toutes les sources
                data = await self.collector.collect_all_sources()
                
                if data:
                    # Sauvegarde des données
                    self.collector.save_collected_data(data)
                    
                    # Statistiques
                    stats = self.collector.get_collection_stats()
                    print(f"📊 Collecté: {len(data)} éléments")
                    print(f"📈 Sources actives: {stats['active_sources']}")
                    
                    # Signal pour l'apprentissage adaptatif
                    self._signal_new_data()
                
                # Attente avant la prochaine collecte
                await asyncio.sleep(interval_hours * 3600)
                
            except KeyboardInterrupt:
                print("\n🛑 Arrêt de la collecte")
                break
            except Exception as e:
                print(f"❌ Erreur: {e}")
                await asyncio.sleep(300)  # Attente 5 min avant retry
    
    def _signal_new_data(self):
        """Signale à l'apprentissage adaptatif qu'il y a de nouvelles données"""
        signal_file = Path("data/incoming/.new_data_signal")
        signal_file.touch()
    
    def collect_now(self) -> List[CollectedData]:
        """Collecte immédiate"""
        return asyncio.run(self.collector.collect_all_sources())


if __name__ == "__main__":
    # Test du collecteur
    collector = AutoDataCollector()
    
    # Collecte immédiate pour test
    print("🧪 Test de collecte...")
    data = collector.collect_now()
    print(f"✅ Test terminé: {len(data)} éléments collectés")
