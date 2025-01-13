# 1. Install required libraries (uncomment if running in a fresh environment)
# !pip install transformers datasets

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

# 5. Define the evaluation loop
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # batch already contains "input_ids", "labels", etc. from the collator
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)  # DataCollator handles shifting

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss

            # Multiply by number of valid tokens to get total loss
            valid_tokens = (labels != -100).sum().item()
            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity

# 6. Run the evaluation and compute perplexity
avg_loss, perplexity = evaluate(model, dataloader, device)
print(f"Average Loss: {avg_loss:.4f}")
print(f"Perplexity: {perplexity:.4f}")
