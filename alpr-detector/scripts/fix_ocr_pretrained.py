from transformers import TrOCRProcessor, ViTImageProcessor, XLMRobertaTokenizer

MODEL_NAME = "microsoft/trocr-small-printed"
SAVE_DIR = "runs/ocr_trocr/best"   # samakan dengan path model kamu

# Buat processor persis seperti saat training
image_processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_NAME)
processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)

# Simpan ke folder yang sama dengan model
processor.save_pretrained(SAVE_DIR)
print("Processor disimpan ke:", SAVE_DIR)
