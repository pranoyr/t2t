import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, default_data_collator

class KernelBookDataset(Dataset):
    def __init__(self, data, tokenizer, max_seq_length=1024):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        messages = self.data[idx]["messages"]
 
        full_text = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False
        )
        
        # Prompt text only (system + user + assistant generation header)
        prompt_messages = [m for m in messages if m["role"] != "assistant"]
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        encodings = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_seq_length,
            padding="max_length",
            return_tensors="pt"
        )
        input_ids = encodings["input_ids"].squeeze(0)
        attention_mask = encodings["attention_mask"].squeeze(0)
        
    
        prompt_ids = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=self.max_seq_length
        )["input_ids"]
        prompt_len = len(prompt_ids)

    
        labels = input_ids.clone()
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


def get_kernelbook_dataloader(tokenizer, batch_size=2, max_seq_length=1024, split="train"):
    """
    Downloads GPUMODE/KernelBook, formats it for ChatML, applies target masking, 
    and returns a ready-to-use PyTorch DataLoader.
    """
    print("Downloading GPUMODE/KernelBook dataset...")
    raw_dataset = load_dataset("GPUMODE/KernelBook", split=split)
    
    def format_to_messages(example):
        # Gracefully handle varying column names in case of dataset updates
        user_prompt = example.get("prompt", example.get("instruction", example.get("question", example.get("python_code", ""))))
        target_kernel = example.get("completion", example.get("output", example.get("answer", example.get("triton_code", ""))))
        
        return {
            "messages": [
                {"role": "system", "content": "You are an expert GPU kernel developer. Write optimized CUDA/Triton kernels."},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": target_kernel}
            ]
        }
        
    print("Formatting dataset into ChatML structure...")
    formatted_dataset = raw_dataset.map(format_to_messages, remove_columns=raw_dataset.column_names)
    
    print("Initializing PyTorch Dataset with target masking...")
    train_dataset = KernelBookDataset(
        data=formatted_dataset, 
        tokenizer=tokenizer, 
        max_seq_length=max_seq_length
    )

    data_collator = default_data_collator

    dataloader = DataLoader(
        train_dataset, 
        shuffle=True, 
        batch_size=batch_size, 
        collate_fn=data_collator
    )
    
    return dataloader



if __name__ == "__main__":
    # Initialize the tokenizer
    model_id = "Qwen/Qwen2.5-Coder-7B"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Get the dataloader
    train_dataloader = get_kernelbook_dataloader(
        tokenizer=tokenizer, 
        batch_size=2, 
        max_seq_length=1024
    )
    
    # Test it by pulling one batch
    for batch in train_dataloader:
        # print("\n--- BATCH SHAPES ---")
        # print("Input IDs shape:", batch["input_ids"].shape)
        # print("Attention Mask shape:", batch["attention_mask"].shape)
        # print("Labels shape:", batch["labels"].shape)
        
        # # Verify masking worked (should see -100 at the start of the labels)
        # print("\nFirst 50 label tokens (Notice the -100s for the prompt):")
        # print(batch["labels"][0].tolist())
        
        # # Verify actual text content from KernelBook is present
        # sample_text = tokenizer.decode(batch["input_ids"][0], skip_special_tokens=False)
        # print("\n--- SAMPLE DECODED TEXT (First 300 chars) ---")
        # print(sample_text + "...")
        break