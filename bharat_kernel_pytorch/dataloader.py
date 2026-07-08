import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorForLanguageModeling

# ---------------------------------------------------------
# Place this OUTSIDE your main() function (at the top)
# ---------------------------------------------------------

# 1. The Dummy Input Text Data
dummy_data = [
    {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Write a python function to add two numbers."},
            {"role": "assistant", "content": "def add(a, b):\n    return a + b"}
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Print Hello World in Bash."},
            {"role": "assistant", "content": "echo 'Hello World'"}
        ]
    }
]

# 2. The Custom Dataset Class
class QwenDummyDataset(Dataset):
    def __init__(self, data, tokenizer, max_seq_length=1024):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        messages = self.data[idx]["messages"]
        
        # 1. Full text including assistant response
        full_text = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False
        )
        
        # 2. Prompt text only (system + user + assistant generation header)
        prompt_messages = [m for m in messages if m["role"] != "assistant"]
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        # 3. Tokenize the full conversation
        encodings = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_seq_length,
            padding="max_length",
            return_tensors="pt"
        )
        input_ids = encodings["input_ids"].squeeze(0)
        attention_mask = encodings["attention_mask"].squeeze(0)
        
        # 4. Tokenize prompt to determine how many tokens to mask out
        prompt_ids = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=self.max_seq_length
        )["input_ids"]
        prompt_len = len(prompt_ids)

        # 5. TARGET MASKING for SFT:
        # Mask out system/user prompt tokens AND padding tokens with -100
        labels = input_ids.clone()
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


# ---------------------------------------------------------
# Place this INSIDE your main() function (Replaces Step 5)
# ---------------------------------------------------------
if __name__ == "__main__":
    # 3. The DataLoader Call
    accelerator.print("Loading custom dummy dataset...")
    
    dummy_dataset = QwenDummyDataset(
        data=dummy_data, 
        tokenizer=tokenizer, 
        max_seq_length=max_seq_length
    )

    # DataCollator automatically shifts inputs to create labels for causal LM
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # The actual DataLoader
    dataloader = DataLoader(
        dummy_dataset, 
        shuffle=True, 
        batch_size=batch_size, 
        collate_fn=data_collator
    )