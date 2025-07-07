#!/bin/bash
# Script de configuration initiale

echo "🚀 Configuration initiale du projet de fine-tuning LLM"
echo "=================================================="

# Créer les répertoires nécessaires
echo "📁 Création des répertoires..."
mkdir -p data/raw data/processed models/cache models/finetuned logs

# Vérifier Python
echo "🐍 Vérification de Python..."
if command -v python3 &> /dev/null; then
    echo "✅ Python3 trouvé: $(python3 --version)"
else
    echo "❌ Python3 non trouvé. Veuillez l'installer."
    exit 1
fi

# Créer et activer un environnement virtuel
echo "🌟 Création de l'environnement virtuel..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Environnement virtuel créé"
else
    echo "✅ Environnement virtuel existant trouvé"
fi

# Activer l'environnement virtuel
echo "🔌 Activation de l'environnement virtuel..."
source venv/bin/activate

# Vérifier pip dans l'environnement virtuel
echo "📦 Mise à jour de pip..."
pip install --upgrade pip

# Installer les dépendances
echo "📦 Installation des dépendances..."
pip install -r requirements.txt

echo ""
echo "🎉 Configuration terminée!"
echo ""
echo "⚠️  IMPORTANT: Pour utiliser le projet, activez d'abord l'environnement virtuel:"
echo "   source venv/bin/activate"
echo ""
echo "📚 Prochaines étapes:"
echo "  1. Activez l'environnement: source venv/bin/activate"
echo "  2. Modifiez les données dans data/raw/sample_conversations.json"
echo "  3. Ajustez la configuration dans config/training_config.yaml"
echo "  4. Lancez l'entraînement: python train.py --config config/training_config.yaml"
echo "  5. Testez le modèle: python inference.py --model-path models/finetuned --interactive"
echo "  6. Interface web: python app.py --model-path models/finetuned"
echo ""
echo "📖 Ou utilisez le notebook: jupyter lab notebooks/fine_tuning_llm_guide.ipynb"
echo ""
echo "🔧 Commandes Makefile disponibles:"
echo "  make help - Afficher l'aide"
echo "  make venv - Créer/activer l'environnement virtuel"
echo "  make install - Installer les dépendances"
echo "  make train - Lancer l'entraînement"
echo "  make app - Interface web"
