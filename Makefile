# 🚀 Fine-tuning LLM Local - Makefile Adaptatif

.PHONY: help venv install setup train inference app clean test data activate adaptive continuous demo gui wolof cleanup

# Configuration
VENV = venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip
CONFIG_FILE = config/training_config.yaml
MODEL_PATH = models/my_finetuned_model

# Couleurs
CYAN = \033[96m
GREEN = \033[92m
YELLOW = \033[93m
RED = \033[91m
NC = \033[0m

# Vérifier si l'environnement virtuel est activé
INVENV = $(shell python -c 'import sys; print("1" if hasattr(sys, "real_prefix") or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix) else "0")')

help: ## Affiche l'aide
	@echo "$(CYAN)🚀 Fine-tuning LLM Local - Système Adaptatif$(NC)"
	@echo ""
	@echo "$(GREEN)📦 Installation:$(NC)"
	@echo "  make venv        - Créer l'environnement virtuel"
	@echo "  make install     - Installer les dépendances"
	@echo "  make setup       - Installation complète"
	@echo ""
	@echo "$(GREEN)🧠 Apprentissage Adaptatif (Nouveau!):$(NC)"
	@echo "  make adaptive    - Démo d'apprentissage adaptatif"
	@echo "  make continuous  - Surveillance continue"
	@echo "  make demo        - Démonstration interactive"
	@echo "  make gui         - Interface graphique"
	@echo ""
	@echo "$(GREEN)🌍 Support Multilingue (Wolof):$(NC)"
	@echo "  make wolof       - Apprentissage avec données wolof"
	@echo "  make collect-wolof - Collecte de données en wolof"
	@echo ""
	@echo "$(GREEN)🧹 Maintenance:$(NC)"
	@echo "  make cleanup     - Nettoyer les fichiers non essentiels"
	@echo ""
	@echo "$(GREEN)🎯 Entraînement Classique:$(NC)"
	@echo "  make train       - Entraînement traditionnel"
	@echo "  make inference   - Test d'inférence"
	@echo "  make app         - Interface web Gradio"
	@echo ""
	@echo "$(GREEN)📊 Utilitaires:$(NC)"
	@echo "  make data        - Préparer les données"
	@echo "  make clean       - Nettoyer"
	@echo "  make activate    - Instructions d'activation"

venv: ## Crée l'environnement virtuel
	@if [ ! -d "$(VENV)" ]; then \
		echo "🌟 Création de l'environnement virtuel..."; \
		python3 -m venv $(VENV); \
		echo "✅ Environnement virtuel créé"; \
	else \
		echo "✅ Environnement virtuel existant"; \
	fi
	@echo "💡 Pour activer: source $(VENV)/bin/activate"

activate: ## Instructions pour activer l'environnement virtuel
	@echo "💡 Pour activer l'environnement virtuel, exécutez:"
	@echo "   source $(VENV)/bin/activate"

install: venv ## Installe les dépendances
	@echo "📦 Installation des dépendances..."
	@if [ "$(INVENV)" = "1" ]; then \
		pip install -r requirements.txt; \
	else \
		$(PIP) install -r requirements.txt; \
	fi
	@echo "✅ Dépendances installées"

install-dev: venv ## Installe les dépendances de développement
	@echo "🔧 Installation des dépendances de développement..."
	@if [ "$(INVENV)" = "1" ]; then \
		pip install -r requirements.txt; \
		pip install -r requirements-distributed.txt; \
	else \
		$(PIP) install -r requirements.txt; \
		$(PIP) install -r requirements-distributed.txt; \
	fi
	@echo "✅ Dépendances de développement installées"

setup: ## Configure l'environnement
	@echo "⚙️ Configuration de l'environnement..."
	mkdir -p data/raw data/processed models/cache models/finetuned logs
	@echo "✅ Répertoires créés"

data: ## Prépare les données d'entraînement
	@echo "📊 Préparation des données..."
	@if [ "$(INVENV)" = "1" ]; then \
		python scripts/prepare_data.py; \
	else \
		$(PYTHON) scripts/prepare_data.py; \
	fi
	@echo "✅ Données préparées"

train: ## Lance l'entraînement
	@echo "🚀 Lancement de l'entraînement..."
	@if [ "$(INVENV)" = "1" ]; then \
		python train.py --config $(CONFIG_FILE); \
	else \
		$(PYTHON) train.py --config $(CONFIG_FILE); \
	fi
	@echo "✅ Entraînement terminé"

train-quick: ## Lance un entraînement rapide (moins d'époques)
	@echo "⚡ Entraînement rapide..."
	@if [ "$(INVENV)" = "1" ]; then \
		python train.py --config $(CONFIG_FILE) --output-dir models/quick_test; \
	else \
		$(PYTHON) train.py --config $(CONFIG_FILE) --output-dir models/quick_test; \
	fi
	@echo "✅ Entraînement rapide terminé"

inference: ## Teste l'inférence avec le modèle
	@echo "🧪 Test d'inférence..."
	@if [ "$(INVENV)" = "1" ]; then \
		python inference.py --model-path $(MODEL_PATH) --interactive; \
	else \
		$(PYTHON) inference.py --model-path $(MODEL_PATH) --interactive; \
	fi
	@echo "✅ Test terminé"

benchmark: ## Lance un benchmark du modèle
	@echo "📈 Benchmark du modèle..."
	@if [ "$(INVENV)" = "1" ]; then \
		python inference.py --model-path $(MODEL_PATH) --benchmark; \
	else \
		$(PYTHON) inference.py --model-path $(MODEL_PATH) --benchmark; \
	fi
	@echo "✅ Benchmark terminé"

app: ## Lance l'interface web
	@echo "🌐 Lancement de l'interface web..."
	@if [ "$(INVENV)" = "1" ]; then \
		python app.py --model-path $(MODEL_PATH); \
	else \
		$(PYTHON) app.py --model-path $(MODEL_PATH); \
	fi

app-share: ## Lance l'interface web avec partage public
	@echo "🌍 Lancement de l'interface web (public)..."
	@if [ "$(INVENV)" = "1" ]; then \
		python app.py --model-path $(MODEL_PATH) --share; \
	else \
		$(PYTHON) app.py --model-path $(MODEL_PATH) --share; \
	fi

notebook: ## Lance Jupyter pour les notebooks
	@echo "📔 Lancement de Jupyter..."
	jupyter lab notebooks/

clean: ## Nettoie les fichiers temporaires
	@echo "🧹 Nettoyage..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/
	@echo "✅ Nettoyage terminé"

clean-models: ## Supprime les modèles entraînés
	@echo "🗑️ Suppression des modèles..."
	rm -rf models/finetuned models/my_finetuned_model models/quick_test
	@echo "✅ Modèles supprimés"

clean-all: clean clean-models ## Nettoyage complet
	@echo "💥 Nettoyage complet terminé"

test: ## Lance les tests (si disponibles)
	@echo "🧪 Lancement des tests..."
	$(PYTHON) -m pytest tests/ -v || echo "Aucun test trouvé"

lint: ## Vérifie le style du code
	@echo "🔍 Vérification du style..."
	flake8 src/ --max-line-length=100 || echo "flake8 non installé"
	black src/ --check || echo "black non installé"

format: ## Formate le code
	@echo "✨ Formatage du code..."
	black src/ || echo "black non installé"
	isort src/ || echo "isort non installé"

docker-build: ## Construit l'image Docker
	@echo "🐳 Construction de l'image Docker..."
	docker build -t llm-finetuning .

docker-run: ## Lance le conteneur Docker
	@echo "🐳 Lancement du conteneur..."
	docker run -p 7860:7860 -v $(PWD)/models:/app/models llm-finetuning

status: ## Affiche le statut du projet
	@echo "📊 Statut du projet:"
	@echo "  📁 Répertoires:"
	@ls -la | grep "^d" | awk '{print "    " $$9}'
	@echo "  📄 Fichiers de configuration:"
	@find config/ -name "*.yaml" 2>/dev/null | awk '{print "    " $$1}' || echo "    Aucun"
	@echo "  🤖 Modèles entraînés:"
	@find models/ -name "adapter_config.json" 2>/dev/null | awk '{print "    " $$1}' | sed 's|/adapter_config.json||' || echo "    Aucun"
	@echo "  📊 Datasets:"
	@find data/ -name "*.json" 2>/dev/null | awk '{print "    " $$1}' || echo "    Aucun"

# Commandes avancées
train-resume: ## Reprend l'entraînement depuis le dernier checkpoint
	@echo "⏯️ Reprise de l'entraînement..."
	$(PYTHON) train.py --config $(CONFIG_FILE) --resume-from-checkpoint models/finetuned

eval: ## Évalue le modèle
	@echo "📊 Évaluation du modèle..."
	$(PYTHON) train.py --config $(CONFIG_FILE) --eval-only

monitor: ## Ouvre TensorBoard pour le monitoring
	@echo "📈 Ouverture de TensorBoard..."
	tensorboard --logdir logs/

## 🧠 Apprentissage Adaptatif - Démonstration
adaptive: venv
	@echo "$(CYAN)🧠 Lancement de la démonstration d'apprentissage adaptatif...$(NC)"
	@if [ ! -d "$(VENV)" ]; then \
		echo "$(RED)❌ Veuillez d'abord exécuter 'make setup'$(NC)"; \
		exit 1; \
	fi
	$(PYTHON) demo_adaptive.py

## 🔄 Apprentissage Continu Automatique
continuous: venv
	@echo "$(CYAN)🔄 Démarrage du système d'apprentissage continu...$(NC)"
	@echo "$(YELLOW)💡 Déposez des fichiers dans data/incoming/ pour voir l'adaptation en temps réel$(NC)"
	@if [ ! -d "$(VENV)" ]; then \
		echo "$(RED)❌ Veuillez d'abord exécuter 'make setup'$(NC)"; \
		exit 1; \
	fi
	$(PYTHON) continuous_adaptive.py

## 🎮 Démonstration Interactive
demo: venv
	@echo "$(CYAN)🎮 Mode démonstration interactive...$(NC)"
	@if [ ! -d "$(VENV)" ]; then \
		echo "$(RED)❌ Veuillez d'abord exécuter 'make setup'$(NC)"; \
		exit 1; \
	fi
	$(PYTHON) demo_adaptive.py --interactive

## 📱 Interface Graphique d'Apprentissage
gui: venv
	@echo "$(CYAN)📱 Lancement de l'interface graphique...$(NC)"
	@if [ ! -d "$(VENV)" ]; then \
		echo "$(RED)❌ Veuillez d'abord exécuter 'make setup'$(NC)"; \
		exit 1; \
	fi
	$(PYTHON) gui_adaptive.py

## 🌍 Apprentissage Wolof
wolof: venv
	@echo "$(CYAN)🌍 Apprentissage adaptatif en Wolof...$(NC)"
	@if [ ! -d "$(VENV)" ]; then \
		echo "$(RED)❌ Veuillez d'abord exécuter 'make setup'$(NC)"; \
		exit 1; \
	fi
	@echo "$(YELLOW)💡 Utilisation des données en wolof d'exemple...$(NC)"
	$(PYTHON) demo_adaptive.py --data data/samples/wolof/wolof_conversations.json

## 📝 Collecte de Données Wolof
collect-wolof: venv
	@echo "$(CYAN)📝 Collecte de données en Wolof...$(NC)"
	@if [ ! -d "$(VENV)" ]; then \
		echo "$(RED)❌ Veuillez d'abord exécuter 'make setup'$(NC)"; \
		exit 1; \
	fi
	@read -p "Type de données (conversation, instruction, text) [conversation]: " type; \
	type="$${type:-conversation}"; \
	$(PYTHON) scripts/collect_wolof_data.py --type $$type --lang wolof

## 🧹 Nettoyage des fichiers non essentiels
cleanup: venv
	@echo "$(CYAN)🧹 Nettoyage du système adaptatif...$(NC)"
	@if [ ! -d "$(VENV)" ]; then \
		echo "$(RED)❌ Veuillez d'abord exécuter 'make setup'$(NC)"; \
		exit 1; \
	fi
	$(PYTHON) scripts/cleanup.py --all

# Installation complète
full-setup: venv install setup data ## Installation et configuration complètes
	@echo "🎉 Configuration complète terminée!"
	@echo "⚠️  N'oubliez pas d'activer l'environnement virtuel: source venv/bin/activate"
	@echo "Puis vous pouvez lancer l'entraînement avec: make train"

# Démo rapide
demo: venv install setup data train-quick app ## Démo complète rapide
	@echo "🎭 Démo terminée! L'interface web est lancée."

clean-venv: ## Supprime l'environnement virtuel
	@echo "🗑️ Suppression de l'environnement virtuel..."
	rm -rf $(VENV)
	@echo "✅ Environnement virtuel supprimé"
