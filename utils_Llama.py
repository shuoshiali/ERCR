import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import json
from typing import Optional, List, Dict, Mapping, Any


device = "cuda" if torch.cuda.is_available() else "cpu"


class APILLM:
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained('/T20200133/Models/meta-llama/Meta-Llama-3.1-8B-Instruct', device_map="auto", trust_remote_code=True).eval()
        self.tokenizer = AutoTokenizer.from_pretrained('/T20200133/Models/meta-llama/Meta-Llama-3.1-8B-Instruct', trust_remote_code=True)
        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

    def generate_prompt(self, prompt, do_sample, temperature, max_new_tokens=1024):
        inputs = self.tokenizer.apply_chat_template(
            conversation = [{"role": "user", "content": prompt}],
            tokenize = True,
            return_tensors = "pt",
            return_dict = True
            ).to(device)
        response = self.model.generate(
            **inputs,
            do_sample = do_sample,
            temperature = temperature,
            max_new_tokens = max_new_tokens,
            top_p = 1.0,
            pad_token_id = self.tokenizer.eos_token_id,
            eos_token_id = self.terminators
        )[:, inputs['input_ids'].shape[1]:]
        result = self.tokenizer.decode(response[0], skip_special_tokens=True).strip()
        return result
    
    def generate(self, messages, do_sample, temperature, max_new_tokens=1024):
        inputs = self.tokenizer.apply_chat_template(
            conversation = messages,
            tokenize = True,
            return_tensors = "pt",
            return_dict = True
            ).to(device)
        response = self.model.generate(
            **inputs,
            do_sample = do_sample,
            temperature = temperature,
            max_new_tokens = max_new_tokens,
            top_p = 1.0,
            pad_token_id = self.tokenizer.eos_token_id,
            eos_token_id = self.terminators
        )[:, inputs['input_ids'].shape[1]:]
        result = self.tokenizer.decode(response[0], skip_special_tokens=True).strip()
        return result
    
    def query(self, msg, temperature=0.01, max_new_tokens=1024):
        try:
            inputs = self.tokenizer.apply_chat_template(
                conversation = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": msg}
                ],
                tokenize = True,
                return_tensors = "pt",
                return_dict = True
                ).to(device)
            completion = self.model.generate(
                **inputs,
                temperature = temperature,
                max_new_tokens = max_new_tokens,
                top_p = 1.0,
                pad_token_id = self.tokenizer.eos_token_id,
                eos_token_id = self.terminators
            )[:, inputs['input_ids'].shape[1]:]
            response = self.tokenizer.decode(completion[0], skip_special_tokens=True).strip()

        except Exception as e:
            print(e)
            response = ""

        return response


if __name__ == "__main__":
    
    api_llm = APILLM()
    response = api_llm.generate(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "Please briefly present the development history of AI."
            }
        ],
        do_sample=True,
        temperature=0.01
    )
    print(response)