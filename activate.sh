#!/bin/bash
# Script d'activation de l'environnement virtuel

if [ -d "venv" ]; then
    echo "🔌 Activation de l'environnement virtuel..."
    source venv/bin/activate
    echo "✅ Environnement virtuel activé"
    echo "💡 Vous pouvez maintenant utiliser les commandes Python normalement"
    echo ""
    echo "🚀 Commandes utiles:"
    echo "  python train.py --config config/training_config.yaml"
    echo "  python inference.py --model-path models/finetuned --interactive"
    echo "  python app.py --model-path models/finetuned"
    echo "  make help"
    echo ""
    echo "🔚 Pour désactiver: deactivate"
else
    echo "❌ Environnement virtuel non trouvé"
    echo "💡 Créez-le d'abord avec: make venv"
fi
