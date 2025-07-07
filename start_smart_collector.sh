#!/bin/bash
# Script de démarrage du collecteur intelligent

echo "🤖 === COLLECTEUR INTELLIGENT AUTONOME ==="
echo "🌐 Démarrage de l'apprentissage automatique..."

# Activation de l'environnement virtuel
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "✅ Environnement virtuel activé"
fi

# Démarrage du collecteur avec gestion d'erreurs
while true; do
    echo "🚀 Lancement du collecteur..."
    python autonomous_collector.py --collection-interval 4 --learning-interval 8
    
    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "👋 Arrêt normal du collecteur"
        break
    else
        echo "⚠️ Redémarrage automatique dans 30 secondes..."
        sleep 30
    fi
done
