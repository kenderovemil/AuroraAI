# AuroraAI
# AuroraAI 🌌

AuroraAI is an intelligent assistant inspired by the northern lights and the legends of Sleeping Beauty.  
The project aims to create a modular, expandable, and aesthetically elegant AI system with a focus on:

- 💬 Natural language and dialogue
- 🧠 Memory and context
- 📊 Data processing
- 🖼️ Visual capabilities (optional)



Project scaffold for AuroraAI.

Tree:

AuroraAI/
├── README.md
├── requirements.txt
├── .gitignore
├── aurora/                  # Основна логика на AuroraAI
│   ├── __init__.py
│   ├── core.py              # Ядро на логиката
│   ├── memory.py            # Дългосрочна памет / контекст
│   ├── planner.py           # Модул за планиране и задачи
│   ├── nlp_tools.py         # Езикови инструменти
│   └── vision.py            # Ако ще има визуални възможности
├── models/                  # Модели и конфигурации
│   ├── config/
│   └── checkpoints/
├── data/                    # Данни, които Aurora обработва
│   ├── raw/
│   └── processed/
├── scripts/                 # Помощни скриптове
│   └── setup_env.sh
├── tests/                   # Тестове
│   └── test_core.py
└── notebooks/               # Jupyter експерименти
    └── exploration.ipynb

Description
-----------
This repository is a minimal scaffold to start building AuroraAI: core logic, memory, planning, simple NLP/vision placeholders, and a basic test harness.

How to use
----------
1. Create a virtual environment and install dependencies:

```bash
bash scripts/setup_env.sh
```

2. Run the unit tests:

```bash
python -m unittest discover -v
```

3. Start implementing features under `aurora/`.

