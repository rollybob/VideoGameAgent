import time
import VGA_wrapped

def startup_sequence():
    steps = [
        "Initializing VGA agent...",
        "Loading OCR and detection modules...",
        "Warming up NLP engine...",
        "Verifying controller state...",
        "Finalizing launch..."
    ]

    for step in steps:
        print(f"[VGA] {step}")
        time.sleep(1.2)

    print("[VGA] Launching agent. Good luck, Commander.\n")

if __name__ == "__main__":
    startup_sequence()
    VGA_wrapped.run()
