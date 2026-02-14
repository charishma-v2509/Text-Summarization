# from transformers import T5Tokenizer, T5ForConditionalGeneration
# import torch
# import sys
# import json

# # Load tokenizer and model
# tokenizer = T5Tokenizer.from_pretrained("t5-small")
# model = T5ForConditionalGeneration.from_pretrained("t5-small")

# def summarize(text, mode="medium", bullet_points=False):
#     # Base prompt
#     if bullet_points:
#         prompt = "summarize in bullet points: "
#         max_len = 100
#     else:
#         if mode == "short":
#             prompt = "summarize briefly: "
#             max_len = 50
#         elif mode == "long":
#             prompt = "summarize in detail: "
#             max_len = 150
#         else:  # medium or default
#             prompt = "summarize: "
#             max_len = 80

#     input_text = prompt + text.strip().replace("\n", " ")

#     inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

#     # Generate summary
#     summary_ids = model.generate(
#         inputs,
#         max_length=max_len,
#         min_length=20,
#         length_penalty=2.0,
#         num_beams=4,
#         early_stopping=True,
#     )

#     output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
#     return output

# if __name__ == "__main__":
#     try:
#         # Read JSON input from stdin
#         data = json.load(sys.stdin)
#         text = data.get("text", "")
#         mode = data.get("mode", "medium")
#         bullet_points = data.get("bullet_points", False)

#         result = summarize(text, mode, bullet_points)
#         print(json.dumps({"summary": result}))
#     except Exception as e:
#         print(json.dumps({"error": str(e)}))
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import sys
import json

# Load model once at startup
tokenizer = None
model = None

def load_model():
    global tokenizer, model
    if tokenizer is None or model is None:
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        model = T5ForConditionalGeneration.from_pretrained("t5-small")

def summarize(text, mode="medium", bullet_points=False):
    load_model()  # Ensure model is loaded
    
    # Validate inputs
    if not text or not isinstance(text, str):
        raise ValueError("Text must be a non-empty string")
    
    if mode not in ["short", "medium", "long"]:
        mode = "medium"
    
    # Configure parameters based on mode
    config = {
        "short": {
            "prompt": "summarize briefly: ",
            "max_len": 50,
            "min_len": 20,
            "length_penalty": 0.6
        },
        "medium": {
            "prompt": "summarize: ",
            "max_len": 100,
            "min_len": 40,
            "length_penalty": 1.0
        },
        "long": {
            "prompt": "summarize in detail: ",
            "max_len": 150,
            "min_len": 60,
            "length_penalty": 2.0
        }
    }
    
    params = config[mode]
    if bullet_points:
        params["prompt"] = "summarize in bullet points: " + params["prompt"]
    
    input_text = params["prompt"] + text.strip().replace("\n", " ")
    
    try:
        inputs = tokenizer.encode(
            input_text, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True
        )
        
        summary_ids = model.generate(
            inputs,
            max_length=params["max_len"],
            min_length=params["min_len"],
            length_penalty=params["length_penalty"],
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=2,
            temperature=0.7
        )
        
        return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    except Exception as e:
        raise RuntimeError(f"Failed to generate summary: {str(e)}")

if __name__ == "__main__":
    try:
        data = json.load(sys.stdin)
        text = data.get("text", "")
        mode = data.get("mode", "medium")
        bullet_points = data.get("bullet_points", False)
        
        result = summarize(text, mode, bullet_points)
        print(json.dumps({"summary": result}))
    except Exception as e:
        print(json.dumps({"error": str(e)}))