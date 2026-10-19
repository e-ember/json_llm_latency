"""Feed a structured driving scenario (JSON) to a Hugging Face LLM and measure
latency to first token (TTFT) and to final token.

pip install torch transformers accelerate
python av_latency.py --model Qwen/Qwen2.5-0.5B-Instruct
"""
import argparse
import json
import time
from threading import Thread

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

SCENARIO = {
    "ego_vehicle": {"speed_kph": 40, "lane": "right", "braking_distance_m": 14.0},
    "environment": {"road": "urban", "surface": "dry", "visibility": "good"},
    "pedestrians": [
        {"id": "A", "age_group": "elderly", "age": 78, "distance_m": 12.0,
         "speed_mps": 0.8, "location": "crosswalk", "collision_probability": 0.6},
        {"id": "B", "age_group": "young", "age": 8, "distance_m": 12.5,
         "speed_mps": 1.4, "location": "crosswalk", "collision_probability": 0.6},
    ],
    "options": [
        {"action": "brake_hard", "risk_to_A": 0.3, "risk_to_B": 0.3, "risk_to_occupant": 0.05},
        {"action": "swerve_left", "risk_to_A": 0.7, "risk_to_B": 0.0, "risk_to_occupant": 0.1},
        {"action": "swerve_right", "risk_to_A": 0.0, "risk_to_B": 0.7, "risk_to_occupant": 0.1},
    ],
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--scenario", help="optional path to a scenario JSON file")
    args = p.parse_args()

    scenario = json.load(open(args.scenario)) if args.scenario else SCENARIO
    messages = [
        {"role": "system", "content": "You are an autonomous vehicle decision module. Decide the best course of action: "
         'Reply ONLY with JSON: {"action": str, "reason": str}.'},
        {"role": "user", "content": json.dumps(scenario)},
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto").to(device)

    inputs = tok.apply_chat_template(messages, add_generation_prompt=True,
                                     return_tensors="pt", return_dict=True).to(device)
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    gen = Thread(target=model.generate, kwargs=dict(
        **inputs, max_new_tokens=args.max_new_tokens, do_sample=False, streamer=streamer))

    t0 = time.perf_counter()  # input fed in
    gen.start()
    first, out = None, ""
    for chunk in streamer:
        if first is None:
            first = time.perf_counter() - t0
        out += chunk
    total = time.perf_counter() - t0
    gen.join()

    print(out)
    print(f"\nTime to first token: {first * 1000:.1f} ms")
    print(f"Time to final token: {total * 1000:.1f} ms")


if __name__ == "__main__":
    main()
