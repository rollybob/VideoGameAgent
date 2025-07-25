# VideoGameAgent

An offline, modular AI agent that can autonomously play video games through an emulator. The system is split into clear layers so components can be swapped or extended easily.

## Architecture

1. **Perception** – YOLOv8 for object detection plus EasyOCR/Pytesseract for text.
2. **Text understanding** – spaCy and Sentence‑Transformers extract game instructions.
3. **Reasoning** – rule based logic or a local LLM (e.g. llama‑cpp) chooses actions.
4. **Controllers** – abstract input modules for different platforms.

## Repository layout

```
VideoGameAgent/
├── agent/              # core packages
│   ├── perception/
│   ├── nlp/
│   ├── reasoning/
│   ├── controllers/
│   └── scripts/        # demos and utilities
├── Emulator/           # place ROMs here (empty by default)
├── maps/               # persistent game memory
├── training_data/      # labelled screenshots for ML
├── requirements.txt
└── README.md
```

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Install Tesseract OCR and ensure it is in your PATH.
3. Put your ROM files inside the `Emulator` folder (they are ignored by git).

## Running

Start your emulator, then run the main agent from the repository root:

```bash
python -m agent.VGA
```

## License

This project is provided for research or personal experimentation. Only use ROMs that you legally own.
