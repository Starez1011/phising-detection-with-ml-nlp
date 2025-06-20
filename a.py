from transformers import AutoModelForSequenceClassification, AutoTokenizer
import shutil
import os

model_name = "facebook/bart-large-mnli"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Find the cache directory
model_dir = model_name.replace("/", "--")
cache_dir = os.path.expanduser("~/.cache/huggingface/hub/models--" + model_dir)
# Find the actual model directory in the cache
import glob
model_cache_path = glob.glob(os.path.join(cache_dir, "*"))[0]

# Copy the model to your project directory
target_dir = os.path.join(os.path.dirname(__file__), "bart-large-mnli-local")
if not os.path.exists(target_dir):
    shutil.copytree(model_cache_path, target_dir)
    print(f"Model copied to {target_dir}")
else:
    print(f"Model already exists at {target_dir}")