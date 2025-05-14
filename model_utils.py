import torch
from transformers import RobertaModel, RobertaTokenizer
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from model_architecture import PatternAttentionRobertaModel  
import json


tokenizer = RobertaTokenizer.from_pretrained('roberta-base', truncation=True, do_lower_case=True)

backbone = RobertaModel.from_pretrained('roberta-base')

num_patterns = 9  

max_len = 128

model = PatternAttentionRobertaModel(
    backbone=backbone,
    num_patterns=num_patterns
)

model.to(device)



state_dict = torch.load("roberta.pt", map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

model.eval()

all_patterns = [
    'black-and-white or polarized thinking / all or nothing thinking',
    'catastrophizing',
    'discounting the positive',
    'jumping to conclusions: fortune-telling',
    'jumping to conclusions: mind reading',
    'labeling and mislabeling',
    'mental filtering',
    'overgeneralization',
    'personalization'
]

# Step 1: Classify Patterns


def classify_patterns(user_input):
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        logits, _ = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    threshold = 0.5 

    detected_patterns = []
    for idx, prob in enumerate(probs):
        if prob >= threshold:
            detected_patterns.append(all_patterns[idx])

    return detected_patterns
def classify_patterns_with_probs(user_input):
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        logits, _ = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    detected = {}
    for idx, prob in enumerate(probs):
        if prob >= 0.8:  
            detected[all_patterns[idx].strip().lower()] = float(prob)

    return detected

# Step2: Choosing best technique 
def load_counts():
    with open("combo_to_technique_counts.json") as f:
        combo_to_technique_counts = json.load(f)
    with open("technique_to_pattern_counts.json") as f:
        technique_to_pattern_counts = json.load(f)
    return combo_to_technique_counts, technique_to_pattern_counts

def choose_best_technique(detected_patterns_probs):
    combo_counts, technique_counts = load_counts()

    detected_patterns = [p.strip().lower() for p in detected_patterns_probs.keys()]
    pattern_probs = {p.strip().lower(): prob for p, prob in detected_patterns_probs.items()}

    pattern_combo_key = str(tuple(sorted(detected_patterns)))

    candidate_techniques = combo_counts.get(pattern_combo_key)
    if not candidate_techniques:
        return "No matching pattern combination found in dataset.", {}

    technique_scores = {}

    for technique, _ in candidate_techniques.items():
        score = 0.0
        for pattern in detected_patterns:
            model_prob = pattern_probs.get(pattern, 0) # probability that the model assigned to a specific pattern
            pattern_counts = technique_counts.get(technique, {})
            total = sum(pattern_counts.values())
            dataset_prob = (pattern_counts.get(pattern, 0) / total) if total > 0 else 0
            score += model_prob * dataset_prob
        technique_scores[technique] = score

    best_technique = max(technique_scores, key=technique_scores.get)
    return best_technique, technique_scores



# Step 3: Generate response 


from peft import PeftModel, PeftConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load PEFT config to get base model
peft_model_path = "llama_model"
peft_config = PeftConfig.from_pretrained(peft_model_path)
base_model_id = peft_config.base_model_name_or_path

# Load tokenizer and base model
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

# Load paths
peft_model_path = "llama_model"
peft_config = PeftConfig.from_pretrained(peft_model_path)
base_model_id = peft_config.base_model_name_or_path

# Load tokenizer
llama_tokenizer = AutoTokenizer.from_pretrained(base_model_id, local_files_only=True)
if "<END>" not in llama_tokenizer.get_vocab():
    llama_tokenizer.add_special_tokens({"additional_special_tokens": ["<END>"]})

# Load base model (no quantization)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    device_map=None,  # ✅ Don't offload to disk
    local_files_only=True
)
base_model.resize_token_embeddings(len(llama_tokenizer), mean_resizing=False)

# Load adapter
llama_model = PeftModel.from_pretrained(base_model, peft_model_path, local_files_only=True)
llama_model.eval()

# Set special tokens
llama_tokenizer.eos_token = "<END>"
llama_tokenizer.pad_token = "<END>"
llama_model.config.eos_token_id = llama_tokenizer.convert_tokens_to_ids("<END>")

def generate_llama_response(user_input, detected_patterns_probs):
    best_technique, _ = choose_best_technique(detected_patterns_probs)
    prompt = f"""You are a compassionate CBT therapist. A user has shared the following concern:

"{user_input}"

You have identified that they may benefit from the technique: {best_technique}.

Write a helpful, empathetic response explaining why this technique is helpful and how they can apply it."""
    
    inputs = llama_tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = llama_model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            eos_token_id=llama_tokenizer.eos_token_id,
            pad_token_id=llama_tokenizer.pad_token_id
        )
    decoded = llama_tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded[len(prompt):].strip()