#!/bin/bash
# Script de dashboard web

echo "🖥️ === DASHBOARD WEB ==="

# Activation de l'environnement virtuel
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# Démarrage du dashboard
echo "🌐 Dashboard disponible sur: http://localhost:8501"
streamlit run dashboard_collector.py --server.port 8501
