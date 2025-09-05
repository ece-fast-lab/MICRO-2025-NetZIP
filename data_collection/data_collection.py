import os
import sys
import pickle
import torch
import argparse
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

class DataCollector:
    def __init__(self, model_name):
        self.model_name = model_name
        self.module_name_map = {}
        self.current_step = 0
        self.forward_counter = 0
        self.backward_counter = 0
        self.total_modules = 0
        self.forward_module_count = 0
        self.backward_module_count = 0

    def clean_filename(self, s):
        return s.replace("/", "-").replace("\\", "-")

    def save_activation_to_disk(self, step, module_name, call_idx, tensor_or_structure, subdir, suffix):
        save_dir = f"{self.model_name.split('/')[-1]}_step_{step}/{subdir}"
        os.makedirs(save_dir, exist_ok=True)

        safe_name = self.clean_filename(module_name)
        filename = f"{call_idx:04d}_{safe_name}{suffix}"
        filepath = os.path.join(save_dir, filename)

        with open(filepath, "wb") as f:
            pickle.dump(tensor_or_structure, f)

    def forward_hook(self, module, inp, out):
        mod_name = self.module_name_map[id(module)]

        self.forward_module_count += 1
        pct = (self.forward_module_count / self.total_modules) * 100
        print(f"Forward pass: {self.forward_module_count}/{self.total_modules} ({pct:5.2f}%) - {mod_name}")

        suffix = ".pkl"

        if isinstance(out, torch.Tensor):
            data_to_save = out.detach().cpu()
        elif isinstance(out, (tuple, list)):
            data_to_save = [x.detach().cpu() if isinstance(x, torch.Tensor) else x for x in out]
        else:
            data_to_save = out

        self.save_activation_to_disk(
            step=self.current_step,
            module_name=mod_name,
            call_idx=self.forward_counter,
            tensor_or_structure=data_to_save,
            subdir="forward_activations",
            suffix=suffix
        )
        self.forward_counter += 1

    def backward_hook(self, module, grad_in, grad_out):
        mod_name = self.module_name_map[id(module)]

        self.backward_module_count += 1
        pct = (self.backward_module_count / self.total_modules) * 100
        print(f"Backward pass: {self.backward_module_count}/{self.total_modules} ({pct:5.2f}%) - {mod_name}")

        suffix = ".pkl"

        if len(grad_out) == 1 and isinstance(grad_out[0], torch.Tensor):
            data_to_save = grad_out[0].detach().cpu()
        else:
            data_to_save = [g.detach().cpu() if isinstance(g, torch.Tensor) else g for g in grad_out]

        self.save_activation_to_disk(
            step=self.current_step,
            module_name=mod_name,
            call_idx=self.backward_counter,
            tensor_or_structure=data_to_save,
            subdir="activation_gradients",
            suffix=suffix
        )
        self.backward_counter += 1

    def make_param_grad_hook(self, param_name):
        def hook_fn(grad):
            save_dir = f"{self.model_name.split('/')[-1]}_step_{self.current_step}/param_grads"
            os.makedirs(save_dir, exist_ok=True)

            safe_name = self.clean_filename(param_name)
            file_path = os.path.join(save_dir, f"{safe_name}.pkl")
            with open(file_path, "wb") as f:
                pickle.dump(grad.detach().cpu(), f)
        return hook_fn

    def setup_hooks(self, model):
        for name, module in model.named_modules():
            self.module_name_map[id(module)] = name

        self.total_modules = sum(1 for _ in model.modules())

        for module in model.modules():
            module.register_forward_hook(self.forward_hook)
            module.register_full_backward_hook(self.backward_hook)

        for param_name, param in model.named_parameters():
            if param.requires_grad:
                param.register_hook(self.make_param_grad_hook(param_name))

    def reset_counters(self):
        self.forward_counter = 0
        self.backward_counter = 0
        self.forward_module_count = 0
        self.backward_module_count = 0

class AlpacaDataset(Dataset):
    def __init__(self, tokenizer, alpaca_data, seq_len=4096):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.samples = []

        for row in alpaca_data:
            instruction = row["instruction"] or ""
            inp = row["input"] or ""
            output = row["output"] or ""

            if inp.strip():
                text = f"Instruction: {instruction}\nInput: {inp}\nResponse: {output}"
            else:
                text = f"Instruction: {instruction}\nResponse: {output}"

            token_ids = tokenizer.encode(text, add_special_tokens=False)

            for i in range(0, len(token_ids), seq_len):
                chunk = token_ids[i:i + seq_len]
                if len(chunk) < seq_len:
                    chunk += [tokenizer.pad_token_id] * (self.seq_len - len(chunk))
                self.samples.append(chunk)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx], dtype=torch.long)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Data collection for model training analysis')
    parser.add_argument('--model_name', type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument('--num_steps', type=int, default=50)

    args = parser.parse_args()
    
    alpaca_dataset = load_dataset("tatsu-lab/alpaca", split="train")

    model_name = args.model_name

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if not tokenizer.pad_token_id:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )

    model.gradient_checkpointing_enable()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    collector = DataCollector(model_name)
    collector.setup_hooks(model)

    seq_length = 4096
    batch_size = 1

    alpaca_dataset_processed = AlpacaDataset(tokenizer, alpaca_dataset, seq_len=seq_length)
    dataloader = DataLoader(alpaca_dataset_processed, batch_size=batch_size, shuffle=True)

    device = torch.device("cpu")
    model.to(device)

    num_steps = args.num_steps

    for step, batch in enumerate(tqdm(dataloader, total=num_steps, desc="Training Steps")):
        if step >= num_steps:
            break

        collector.current_step = step
        collector.reset_counters()

        batch = batch.to(device)
        inputs = {
            "input_ids": batch,
            "labels": batch
        }

        optimizer.zero_grad()

        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            outputs = model(**inputs)
            loss = outputs.loss

        print("Backward pass starting...")
        loss.backward()
        print("Backward pass complete.")

        print("Optimizer step starting...")
        optimizer.step()
        print("Optimizer step complete.")

        print(f"Step {step}, Loss = {loss.item():.4f}")

if __name__ == "__main__":
    main()