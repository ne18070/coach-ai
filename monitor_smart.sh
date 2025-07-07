#!/bin/bash
# Script de monitoring intelligent

echo "📊 === MONITORING INTELLIGENT ==="

# Activation de l'environnement virtuel
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# Monitoring continu
watch -n 30 python monitor_collector.py
