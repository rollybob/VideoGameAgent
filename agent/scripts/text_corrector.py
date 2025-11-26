
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class TextCorrector:
    def __init__(self, model_name="vennify/t5-base-grammar-correction"):
        print("[TextCorrector] Loading model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        print("[TextCorrector] Model loaded.")

    def correct(self, text: str) -> str:
        inputs = self.tokenizer.encode("grammar: " + text, return_tensors="pt", max_length=512, truncation=True)
        outputs = self.model.generate(inputs, max_length=512, num_beams=4, early_stopping=True)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
if __name__ == "__main__":
    corrector = TextCorrector()
    raw = "In the world which you are abowt to enter, you will embark on a grand adventure with you as the hera."
    print("Original:", raw)
    print("Corrected:", corrector.correct(raw))
