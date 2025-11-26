import torch, cv2
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

MODEL_ID = "IDEA-Research/grounding-dino-tiny"  # lighter; try "grounding-dino-base" for stronger
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(device).eval()

def run_grounding_dino(img_bgr, prompts, box_thresh=0.35, text_thresh=0.25):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    # HF likes a list-of-list of phrases; use short noun phrases
    text_labels = [[p.replace("_"," ") for p in prompts]]
    inputs = processor(images=pil_img, text=text_labels, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    res = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=box_thresh,
        text_threshold=text_thresh,
        target_sizes=[pil_img.size[::-1]]  # (H, W)
    )[0]
    dets = [(lbl, float(score), [float(x) for x in box.tolist()])
            for box, score, lbl in zip(res["boxes"], res["scores"], res["labels"])]
    return dets