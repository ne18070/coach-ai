# Guide d'Apprentissage Adaptatif en Wolof

Ce guide explique comment collecter des données en wolof (ou d'autres langues) et entraîner votre système d'apprentissage adaptatif sur ces données.

## 1. Collecte de données

### 1.1 Utiliser le script de collecte interactive

Le script `scripts/collect_wolof_data.py` vous permet de collecter des données de manière interactive:

```bash
# Pour collecter des conversations (question-réponse)
python scripts/collect_wolof_data.py --type conversation --lang wolof

# Pour collecter des instructions
python scripts/collect_wolof_data.py --type instruction --lang wolof

# Pour collecter du texte brut
python scripts/collect_wolof_data.py --type text --lang wolof
```

### 1.2 Convertir des données existantes

Si vous disposez déjà de textes en wolof, vous pouvez les convertir au format approprié:

```bash
# Convertir un texte en format de complétion
python scripts/convert_wolof_data.py --action text_to_completion --input mon_texte.txt --output data/processed/wolof_completions.json

# Convertir un CSV en conversations
python scripts/convert_wolof_data.py --action csv_to_conversation --input mes_qa.csv --output data/processed/wolof_conversations.json

# Convertir des QA en instructions
python scripts/convert_wolof_data.py --action qa_to_instruction --input wolof_qa.json --output data/processed/wolof_instructions.json
```

### 1.3 Générer des exemples de données wolof

Pour générer des exemples de données en wolof:

```bash
python scripts/convert_wolof_data.py --action create_samples --output data/samples/wolof
```

## 2. Préparation pour l'entraînement

### 2.1 Structure recommandée des données

Pour des conversations:
```json
[
  {
    "input": "Naka nga def?",
    "output": "Mangi fi rekk, yow?"
  },
  {
    "input": "Fan la université bi nekk?",
    "output": "Université bi dafa nekk ci Dakar, ci route de l'aéroport bi."
  }
]
```

Pour des instructions:
```json
[
  {
    "instruction": "Wadial ci mbir mi aju ci agriculture bi ci Sénégal",
    "response": "Agriculture bi ci Sénégal dafa am importance bu rey. Ñi ngi dimbali population bi ci lekk ak liggéey..."
  }
]
```

### 2.2 Organisation des fichiers

Placez vos fichiers dans ces dossiers:

- `data/incoming/` - Pour l'apprentissage adaptatif continu
- `data/raw/` - Pour les données brutes
- `data/processed/` - Pour les données prétraitées

## 3. Entraînement et adaptation

### 3.1 Entraînement avec l'apprentissage adaptatif

```bash
# Entraînement adaptatif sur un fichier spécifique
python demo_adaptive.py --data data/incoming/wolof_conversations.json

# OU avec la commande make
make adaptive DATA=data/incoming/wolof_conversations.json
```

### 3.2 Mode d'apprentissage continu

Pour surveiller continuellement les nouveaux fichiers:

```bash
python continuous_adaptive.py --watch-dir data/incoming --processed-dir data/processed
```

### 3.3 Interface graphique

```bash
python gui_adaptive.py
```

## 4. Évaluation et test

Après l'entraînement, testez votre modèle:

```bash
# Mode interactif
python demo_adaptive.py --interactive

# OU
make demo
```

## 5. Astuces pour l'apprentissage optimal en wolof

1. **Cohérence orthographique** - Essayez de maintenir une cohérence dans l'écriture du wolof (même si plusieurs variantes orthographiques existent).

2. **Données variées** - Incluez divers sujets et registres de langue pour une meilleure généralisation.

3. **Apprentissage progressif** - Commencez par des conversations simples, puis progressez vers des textes plus complexes.

4. **Adaptation itérative** - Après un premier cycle d'apprentissage, évaluez les résultats et ajustez vos données si nécessaire.

5. **Mélange linguistique** - Si pertinent, incluez des exemples de code-switching wolof-français comme souvent pratiqué dans les conversations réelles.

## 6. Ressources pour le wolof

- [Wolofal](https://wolofal.org/) - Ressources linguistiques en wolof
- [Jangal](https://jangal.digital/) - Plateforme éducative incluant du contenu en wolof
- [Common Voice](https://commonvoice.mozilla.org/) - Peut contenir des transcriptions en wolof
- [Corpus parallèles OPUS](https://opus.nlpl.eu/) - Pour des traductions

---

N'hésitez pas à adapter ce guide à vos besoins spécifiques. Le système d'apprentissage adaptatif est conçu pour s'adapter automatiquement à vos données, quelle que soit la langue!
