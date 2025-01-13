# 1. Install required libraries (uncomment if running in a fresh environment)
# !pip install transformers datasets sacrebleu evaluate

# 2. Import necessary libraries
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import math
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---- Import the `evaluate` library
import evaluate

# 3. Load the pre-trained model and tokenizer
model_name = 'gpt2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# FIX: Assign a pad token if none exists
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# 4. Prepare the evaluation dataset
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

def tokenize_function(examples):
    return tokenizer(examples['text'], return_special_tokens_mask=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text'])

block_size = 512
def group_texts(examples):
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated['input_ids'])
    total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i:i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated.items()
    }
    return result

grouped_dataset = tokenized_dataset.map(group_texts, batched=True)
grouped_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

# Create a data collator for causal LM (auto-shifts labels).
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # for a causal language model like GPT-2
)

# Pass collate_fn to DataLoader
batch_size = 8
dataloader = DataLoader(grouped_dataset, batch_size=batch_size, collate_fn=data_collator)

############################################
def evaluate_model(model, dataloader, device):
    """
    Return dummy avg_loss for "Average Loss"
    and the BLEU score (instead of perplexity).
    """
    model.eval()

    # Load the BLEU metric from the `evaluate` library
    bleu_metric = evaluate.load("bleu")

    for batch in tqdm(dataloader, desc="Evaluating"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        # Greedy predictions
        with torch.no_grad():
            pred_ids = torch.argmax(outputs.logits, dim=-1)

        # Convert -100 back to pad/eos to avoid decoding issues
        clean_labels = []
        for lbl_seq in labels:
            lbl_seq = [l if l != -100 else tokenizer.eos_token_id for l in lbl_seq]
            clean_labels.append(lbl_seq)

        # Decode predictions and references as raw text
        decoded_preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(clean_labels, skip_special_tokens=True)

        # Easiest approach: let `evaluate` do the tokenization
        # Pass raw strings as `predictions` and single-reference lists
        references_for_batch = [[ref_text] for ref_text in decoded_labels]

        bleu_metric.add_batch(
            predictions=decoded_preds,
            references=references_for_batch
        )

    final_bleu = bleu_metric.compute()['bleu']

    # Return dummy loss, and the BLEU as "bleuVar"
    return 0.0, final_bleu

# 6. Run the evaluation
avg_loss, bleuVar = evaluate_model(model, dataloader, device)
print(f"Average Loss: {avg_loss:.4f}")
print(f"bleuVar (actually BLEU): {bleuVar:.4f}")
