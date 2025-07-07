# Fine-tuning LLM Local

Ce projet permet de faire du fine-tuning de modèles de langage open-source localement en utilisant des techniques modernes et efficaces.

## Fonctionnalités

- 🚀 Fine-tuning avec LoRA (Low-Rank Adaptation) pour l'efficacité
- 📊 Support pour différents formats de données (JSON, CSV, TXT)
- 🔧 Configuration flexible via YAML
- 📈 Monitoring et logging avancés
- 💾 Sauvegarde automatique des checkpoints
- 🧪 Évaluation et métriques de performance
- 🤗 Compatible avec les modèles Hugging Face

## Modèles supportés

- Llama 2/3 (7B, 13B, 70B)
- Mistral 7B
- Code Llama
- Falcon
- GPT-NeoX
- Et tous les modèles compatibles avec Transformers

## Installation

### ✅ Installation rapide et automatique

Le projet utilise un environnement virtuel Python pour éviter les conflits avec votre système.

```bash
# 1. Rendre le script exécutable
chmod +x setup.sh

# 2. Lancer l'installation automatique
./setup.sh

# 3. Activer l'environnement virtuel
source venv/bin/activate
```

**Note**: Sur macOS, l'environnement virtuel est **obligatoire** pour éviter l'erreur "externally-managed-environment".

### Méthode alternative: Makefile
```bash
# Créer l'environnement virtuel et installer
make full-setup

# Activer l'environnement virtuel
source venv/bin/activate
```

### Méthode 3: Manuelle
```bash
# Créer un environnement virtuel
python3 -m venv venv
source venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt

# (Optionnel) Dépendances pour l'entraînement distribué
pip install -r requirements-distributed.txt
```

⚠️ **Important**: Sur macOS, utilisez toujours un environnement virtuel pour éviter les erreurs d'environnement géré externellement.

## Utilisation rapide

⚠️ **Prérequis**: Assurez-vous que l'environnement virtuel est activé:
```bash
source venv/bin/activate
```

### 1. Préparer vos données

```python
from src.data_processor import DataProcessor

processor = DataProcessor()
processor.prepare_dataset("data/raw/mon_dataset.json", "data/processed/")
```

### 2. Configuration

Modifiez `config/training_config.yaml` selon vos besoins.

### 3. Lancer le fine-tuning

```bash
# Avec Python directement
python train.py --config config/training_config.yaml

# Ou avec Makefile
make train
```

### 4. Tester le modèle

```python
from src.inference import ModelInference

model = ModelInference("models/mon_modele_finetuned")
response = model.generate("Votre question ici")
print(response)
```

### 5. Interface web

```bash
# Lancer l'interface Gradio
python app.py --model-path models/finetuned

# Ou avec Makefile
make app
```

## Structure du projet

```
coach/
├── 📘 README.md                    # Documentation complète
├── ⚙️ config/
│   └── training_config.yaml        # Configuration d'entraînement
├── 📊 data/
│   └── raw/sample_conversations.json # Données d'exemple
├── 🧠 src/                         # Code source modulaire
│   ├── utils.py                    # Utilitaires
│   ├── data_processor.py           # Traitement des données
│   ├── model_manager.py            # Gestion des modèles
│   ├── trainer.py                  # Logique d'entraînement
│   └── inference.py                # Inférence
├── 📔 notebooks/
│   └── fine_tuning_llm_guide.ipynb # Tutoriel interactif
├── 🚀 Scripts principaux
│   ├── train.py                    # Entraînement
│   ├── inference.py                # Test d'inférence
│   └── app.py                      # Interface web Gradio
└── 🔧 Utilitaires
    ├── Makefile                    # Commandes simplifiées
    ├── setup.sh                    # Configuration initiale
    ├── activate.sh                 # Activation environnement virtuel
    └── requirements.txt            # Dépendances
```

## 🔧 Commandes Makefile

Le projet inclut un Makefile avec des commandes pratiques :

```bash
# Afficher l'aide
make help

# Configuration initiale
make venv                 # Créer l'environnement virtuel
make install             # Installer les dépendances
make setup               # Créer les répertoires
make full-setup          # Configuration complète

# Données et entraînement
make data                # Préparer les données
make train               # Lancer l'entraînement complet
make train-quick         # Entraînement rapide pour test

# Test et déploiement
make inference           # Test d'inférence interactif
make benchmark           # Benchmark de performance
make app                 # Interface web locale
make app-share           # Interface web publique

# Utilitaires
make clean               # Nettoyage fichiers temporaires
make clean-models        # Supprimer modèles entraînés
make clean-venv          # Supprimer environnement virtuel
make status              # Statut du projet
```

## Exemples

Consultez le dossier `notebooks/` pour des exemples détaillés et des tutoriels.

## Dépannage

### Erreur "externally-managed-environment"
Cette erreur est courante sur macOS. Utilisez un environnement virtuel :
```bash
./setup.sh
source venv/bin/activate
```

### Problèmes de mémoire GPU
- Réduisez `per_device_train_batch_size` dans la configuration
- Activez la quantification 4-bit dans la configuration
- Utilisez `gradient_accumulation_steps` plus élevé

### Modèle non trouvé
Assurez-vous d'avoir terminé l'entraînement avant de tester l'inférence.

## Licence

MIT License
