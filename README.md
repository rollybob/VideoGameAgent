# Game Agent Emulator Bot

This project is an AI-driven game agent designed to interact with emulators (like mGBA) using OCR (Optical Character Recognition) and simulated key inputs. The goal is to automate gameplay tasks like movement, battle actions, and dialogue progression using screen analysis and simple decision logic.

---

## Features

- Simulates directional and button input using `pyautogui`
- Reads screen text with `pytesseract` (OCR)
- Detects game states: Overworld, Dialogue/Menu, and Battle
- Automatically explores or battles depending on context
- Works with mGBA emulator running Game Boy Advance games

---

## Project Structure

```
GameAgentUSB/
├── agent/
│   ├── main.py
│   ├── screen_reader.py
│   ├── input_controller.py
│   ├── ocr_test.py
│   ├── test_input.py
│   ├── tools/
│   │   └── tesseract/
├── Emulator/
│   └── (Place your ROMs here)
├── Results/
├── ocr_training/
└── requirements.txt
```

---

## Requirements

Install the following Python packages:

```bash
pip install -r requirements.txt
```

Make sure you also:

- Download and install Tesseract OCR: https://github.com/tesseract-ocr/tesseract
- Add Tesseract to your system PATH or manually configure the path in your scripts.

---

## Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/GameAgentUSB.git
   cd GameAgentUSB/agent
   ```

2. Install Python dependencies:
   ```bash
   pip install -r ../requirements.txt
   ```

3. Install Tesseract OCR:
   - Download and install from the link above.
   - Ensure it's added to your system PATH or update the script path manually.

4. Start your emulator (e.g., mGBA) with a valid `.gba` ROM in the `/Emulator/` directory.

5. Run the agent:
   ```bash
   python main.py
   ```

---

## ROM Disclaimer

This project does not include any ROM files, nor does it encourage piracy. If you're using this agent:

- Ensure that you own physical copies of any ROMs you use.
- Do not push ROM files to GitHub (they're listed in `.gitignore` for this reason).
- Respect all copyright and licensing laws.

---

## Notes

- Currently optimized for the mGBA emulator. You can change the emulator title in `find_emulator_window()` if needed.
- Works best with pixel-perfect rendering settings for OCR accuracy.

---

## License

This repository is open for academic, research, or personal development. Do not use it for unauthorized automation in commercial software or services.