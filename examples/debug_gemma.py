#!/usr/bin/env python3
"""Quick debug script to test Gemma model directly."""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Model path')
    parser.add_argument('--device', default='mps')
    args = parser.parse_args()

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=args.device
    )
    model.eval()

    # Test with chat template
    messages = [{"role": "user", "content": "Explain quantum computing in one sentence."}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print(f"\nFormatted prompt:\n{repr(prompt)}\n")

    inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
    print(f"Input shape: {inputs.input_ids.shape}")

    # Generate with HF generate()
    print("Generating with model.generate()...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nGenerated:\n{response}")

    # Test single forward pass
    print("\n--- Testing single forward pass ---")
    with torch.no_grad():
        out = model(input_ids=inputs.input_ids)
        logits = out.logits[:, -1, :]
        print(f"Logits shape: {logits.shape}")
        print(f"Logits range: [{logits.min():.2f}, {logits.max():.2f}]")

        # Top 5 predicted tokens
        top5 = torch.topk(logits[0], 5)
        print("Top 5 next tokens:")
        for i, (idx, score) in enumerate(zip(top5.indices, top5.values)):
            token = tokenizer.decode([idx])
            print(f"  {i+1}. {repr(token)} (score: {score:.2f})")


if __name__ == '__main__':
    main()
