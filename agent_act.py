"""agent.act(state) -> (action, meta) latency benchmark.

Model specs (--models, space separated):
  hf:<hf_repo_id>     local Hugging Face model (GPU if available)
  api:<provider>:<model>   OpenAI-compatible API; provider is groq, gemini or openai.
                      Keys are read from the environment or a .env file (see .env.example).
  mock:<delay_ms>     fake agent that sleeps; use it to test the pipeline

Examples:
  python agent_act.py --models mock:200 --runs 10
  python agent_act.py --models hf:Qwen/Qwen2.5-0.5B-Instruct --runs 10
  python agent_act.py --models api:groq:llama-3.1-8b-instant api:gemini:gemini-2.0-flash
"""
import argparse
import json
import os
import platform
import re
import statistics as st
import time
from threading import Thread


def load_dotenv(path=None):
    """Minimal .env reader (KEY=VALUE lines); real environment variables take precedence."""
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            if v.strip():
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_dotenv()

ACTIONS = ["maintain", "brake_soft", "brake_hard", "steer_left", "steer_right",
           "brake_steer_left", "brake_steer_right"]

DEFAULT_MODELS = [
    "hf:Qwen/Qwen2.5-0.5B-Instruct",
    "hf:HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "hf:microsoft/Phi-3.5-mini-instruct",
    "hf:mistralai/Mistral-7B-Instruct-v0.3",
    "hf:Qwen/Qwen2.5-14B-Instruct",
]

# Oct 19 scenario in the locked interface format (ego at origin, +x ahead, +y left).
# Pedestrian speeds are not in the shared interface for this scenario; vx=vy=0 matches
# the hand-check's "pedestrians held still" simplification.
SCENARIO_STATE = {
    "t": 0.0,
    "ego": {"x": 0.0, "y": 0.0, "v": 40 / 3.6, "heading": 0.0},
    "agents": [
        {"id": "A", "type": "pedestrian", "x": 12.0, "y": 0.6, "vx": 0.0, "vy": 0.0},
        {"id": "B", "type": "pedestrian", "x": 12.5, "y": -0.6, "vx": 0.0, "vy": 0.0},
    ],
    "road": {"lane_width": 3.5, "num_lanes": 2, "speed_limit": 40 / 3.6},
}

SYSTEM_PROMPT = (
    "You are the decision module of an autonomous vehicle. Every call you receive the "
    "current state and must choose exactly one action.\n"
    "Actions:\n"
    "- maintain: no brake, no steer\n"
    "- brake_soft: brake at 0.3 g\n"
    "- brake_hard: maximum braking\n"
    "- steer_left: maximum lateral acceleration left, no brake\n"
    "- steer_right: maximum lateral acceleration right, no brake\n"
    "- brake_steer_left: braking and steering left\n"
    "- brake_steer_right: braking and steering right\n"
    "Coordinates: x is ahead of the ego vehicle, y is positive to the left, units are "
    "meters and seconds.\n"
    'Reply ONLY with JSON: {"action": "<one of the actions above>"}'
)


def format_prompt(state):
    """state dict -> chat messages."""
    ego = state["ego"]
    lines = [f"t = {state['t']:.2f} s",
             f"ego: x={ego['x']:.2f} y={ego['y']:.2f} v={ego['v']:.2f} m/s heading={ego['heading']:.2f} rad"]
    for a in state["agents"]:
        lines.append(f"agent {a['id']} ({a['type']}): x={a['x']:.2f} y={a['y']:.2f} "
                     f"vx={a['vx']:.2f} vy={a['vy']:.2f}")
    r = state["road"]
    lines.append(f"road: lane_width={r['lane_width']} m, num_lanes={r['num_lanes']}, "
                 f"speed_limit={r['speed_limit']:.2f} m/s")
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(lines)}]


_ACTION_RE = re.compile(r"\b(" + "|".join(sorted(ACTIONS, key=len, reverse=True)) + r")\b")


def parse_action(text):
    """text -> (action or None, how). how in {'json', 'text', 'invalid'}.

    json: parsed JSON object whose "action" is one of the 7.
    text: no usable JSON, but exactly one distinct action name appears in the text.
    Anything else (no action, several different actions, unknown action) is invalid.
    """
    m = re.search(r"\{.*?\}", text, re.S)
    if m:
        try:
            a = str(json.loads(m.group(0)).get("action", "")).strip().lower()
            if a in ACTIONS:
                return a, "json"
        except (json.JSONDecodeError, AttributeError):
            pass
    found = set(_ACTION_RE.findall(text.lower()))
    if len(found) == 1:
        return found.pop(), "text"
    return None, "invalid"


class Agent:
    """Subclasses implement generate(messages) -> (text, ttft_s, n_prompt_tokens, n_new_tokens)."""
    name = "agent"

    def __init__(self, max_new_tokens=32, temperature=0.0):
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def act(self, state):
        messages = format_prompt(state)
        t0 = time.perf_counter()
        text, ttft, n_in, n_out = self.generate(messages, t0)
        total = time.perf_counter() - t0
        action, how = parse_action(text)
        meta = {
            "latency_ms": total * 1000,
            "ttft_ms": None if ttft is None else ttft * 1000,
            "raw_text": text,
            "tokens": {"prompt": n_in, "completion": n_out},
            "parse": how,
            "valid": action is not None,
        }
        return action, meta


class MockAgent(Agent):
    def __init__(self, delay_ms, **kw):
        super().__init__(**kw)
        self.name = f"mock:{delay_ms}ms"
        self.delay = delay_ms / 1000

    def generate(self, messages, t0):
        time.sleep(self.delay / 2)
        ttft = time.perf_counter() - t0
        time.sleep(self.delay / 2)
        return '{"action": "brake_hard"}', ttft, 0, 7


class HFAgent(Agent):
    def __init__(self, repo, dtype="auto", **kw):
        super().__init__(**kw)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.name = f"hf:{repo}"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(repo)
        # Load weights straight onto the GPU so large models don't have to fit in CPU RAM first.
        if self.device == "cuda":
            self.model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=dtype, device_map="cuda")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=dtype)
        self.model.eval()

    def generate(self, messages, t0):
        from transformers import TextIteratorStreamer

        class CountingStreamer(TextIteratorStreamer):
            n_tokens = 0

            def put(self, value):
                if not self.next_tokens_are_prompt:
                    self.n_tokens += value.numel()
                super().put(value)

        # Tokenization is inside the timed region on purpose: act() latency includes it.
        try:
            templated = self.tok.apply_chat_template(messages, add_generation_prompt=True,
                                                     return_tensors="pt", return_dict=True)
        except Exception:  # template rejects a system role: fold it into the user turn
            merged = [{"role": "user",
                       "content": messages[0]["content"] + "\n\n" + messages[1]["content"]}]
            templated = self.tok.apply_chat_template(merged, add_generation_prompt=True,
                                                     return_tensors="pt", return_dict=True)
        inputs = templated.to(self.device)
        n_in = inputs["input_ids"].shape[1]
        streamer = CountingStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)
        kwargs = dict(**inputs, max_new_tokens=self.max_new_tokens, streamer=streamer,
                      pad_token_id=self.tok.eos_token_id)
        if self.temperature > 0:
            kwargs.update(do_sample=True, temperature=self.temperature)
        else:
            kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        th = Thread(target=self.model.generate, kwargs=kwargs)
        th.start()
        ttft, text = None, ""
        for chunk in streamer:
            if ttft is None:
                ttft = time.perf_counter() - t0
            text += chunk
        th.join()
        if self.device == "cuda":
            self.torch.cuda.synchronize()
        return text, ttft, n_in, streamer.n_tokens


PROVIDERS = {
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY"),
    "openai": (None, "OPENAI_API_KEY"),
}


class APIAgent(Agent):
    def __init__(self, provider, model, **kw):
        super().__init__(**kw)
        from openai import OpenAI
        if provider not in PROVIDERS:
            raise ValueError(f"unknown provider '{provider}', choose from {list(PROVIDERS)}")
        base_url, key_env = PROVIDERS[provider]
        key = os.environ.get(key_env)
        if not key:
            raise ValueError(f"{key_env} is not set (add it to .env)")
        self.name = f"api:{provider}:{model}"
        self.model = model
        self.client = OpenAI(api_key=key, base_url=base_url)

    # Reasoning models spend hidden tokens before the answer, so API calls get a larger budget
    # and the lowest reasoning effort the model supports.
    def generate(self, messages, t0):
        extra = {}
        if self.model.startswith("gemini-2.5-flash"):
            extra["reasoning_effort"] = "none"
        elif self.model.startswith("gemini-3") or "gpt-oss" in self.model:
            extra["reasoning_effort"] = "low"
        stream = self.client.chat.completions.create(
            model=self.model, messages=messages, max_tokens=max(self.max_new_tokens, 512),
            temperature=self.temperature, stream=True, stream_options={"include_usage": True},
            **extra)
        ttft, text, n_in, n_out = None, "", None, None
        for ev in stream:
            if ev.usage:
                n_in, n_out = ev.usage.prompt_tokens, ev.usage.completion_tokens
            if ev.choices and ev.choices[0].delta.content:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                text += ev.choices[0].delta.content
        return text, ttft, n_in, n_out


def build_agent(spec, args):
    kind, _, rest = spec.partition(":")
    kw = dict(max_new_tokens=args.max_new_tokens, temperature=args.temperature)
    if kind == "hf":
        return HFAgent(rest, dtype=args.dtype, **kw)
    if kind == "api":
        provider, _, model = rest.partition(":")
        return APIAgent(provider, model, **kw)
    if kind == "mock":
        return MockAgent(float(rest), **kw)
    raise SystemExit(f"bad model spec: {spec}")


def describe_hardware():
    info = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return info


def med(xs):
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--runs", type=int, default=10)
    p.add_argument("--warmup", type=int, default=1, help="untimed runs per model (CUDA init, caches)")
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--dtype", default="auto", help="auto, float16, bfloat16, float32")
    p.add_argument("--state", help="optional path to a state JSON file")
    p.add_argument("--out", default="runs.jsonl", help="raw per-run log")
    p.add_argument("--table", default="latency_table.md", help="markdown latency table")
    p.add_argument("--require_cuda", action="store_true", help="abort unless a GPU is visible")
    args = p.parse_args()

    state = json.load(open(args.state)) if args.state else SCENARIO_STATE
    hw = describe_hardware()
    if args.require_cuda and not hw.get("cuda"):
        raise SystemExit("--require_cuda set but no GPU is visible")
    print("hardware:", hw)
    print(f"settings: dtype={args.dtype} max_new_tokens={args.max_new_tokens} "
          f"temperature={args.temperature} runs={args.runs} warmup={args.warmup}\n")

    rows = []
    with open(args.out, "a", encoding="utf-8") as log:
        for spec in args.models:
            print(f"loading {spec} ...", flush=True)
            try:
                agent = build_agent(spec, args)
            except Exception as e:  # keep going so one gated/huge model doesn't kill the sweep
                print(f"  SKIPPED {spec}: {e}")
                continue
            metas = []
            try:
                for _ in range(args.warmup):
                    agent.act(state)
                for i in range(args.runs):
                    action, meta = agent.act(state)
                    metas.append(meta)
                    log.write(json.dumps({"model": agent.name, "run": i, "action": action,
                                          "settings": vars(args) | {"hardware": hw},
                                          "state": state, **meta}) + "\n")
                    log.flush()
                    print(f"  run {i}: ttft={meta['ttft_ms'] or float('nan'):.0f} ms "
                          f"total={meta['latency_ms']:.0f} ms action={action}")
            except Exception as e:  # a failing model (bad name, rate limit) must not kill the sweep
                print(f"  FAILED {spec} after {len(metas)} timed runs: {type(e).__name__}: {e}")
            if metas:
                rows.append((agent.name, metas))
            del agent
            try:
                import torch
                torch.cuda.empty_cache()
            except ImportError:
                pass

    header = ("| model | n | TTFT med | TTFT mean | total med | total mean | total sd | total min | "
              "total max | prompt tok | out tok | valid % | json % |")
    lines = [header, "|" + "---|" * 13]
    for name, ms in rows:
        n = len(ms)
        tot = [m["latency_ms"] for m in ms]
        ttfts = [m["ttft_ms"] for m in ms if m["ttft_ms"] is not None]
        lines.append(
            f"| {name} | {n} | {med(ttfts):.0f} | {st.mean(ttfts) if ttfts else float('nan'):.0f} "
            f"| {med(tot):.0f} | {st.mean(tot):.0f} | {st.stdev(tot) if n > 1 else 0:.0f} "
            f"| {min(tot):.0f} | {max(tot):.0f} "
            f"| {med(m['tokens']['prompt'] for m in ms):.0f} "
            f"| {med(m['tokens']['completion'] for m in ms):.0f} "
            f"| {100 * sum(m['valid'] for m in ms) / n:.0f} "
            f"| {100 * sum(m['parse'] == 'json' for m in ms) / n:.0f} |")
    footer = (f"\nTimes in ms. valid % = parses to one of the 7 actions; json % = valid via strict JSON.\n"
              f"Settings: dtype={args.dtype}, max_new_tokens={args.max_new_tokens}, "
              f"temperature={args.temperature}, warmup={args.warmup}. Hardware: {hw}")
    table = "\n".join(lines) + "\n" + footer
    print("\n" + table)
    open(args.table, "w", encoding="utf-8").write(table + "\n")


if __name__ == "__main__":
    main()
