# json_llm_latency

Latency benchmark for `agent.act(state) -> action, meta`: how fast can a language model pick one of 7 driving actions?

For each model it records time to first token (TTFT), total time, token counts and the raw text, and reports the % of outputs that parse to a valid action.

## Files
- `agent_act.py` - prompt formatter, action parser, `act()` and the benchmark loop
- `k8s/` - Nautilus PVC and Job YAML for the timed GPU run
- `av_latency.py` - old single-shot script (kept for reference)
- `.env.example` - template for API keys

## Setup
```
pip install torch transformers accelerate openai
copy .env.example .env      # then fill in GROQ_API_KEY and GEMINI_API_KEY
```
`.env` is git-ignored. Never commit keys. For a GPU locally, install a CUDA build of torch (an RTX 50-series card needs CUDA 12.8+: `pip install torch --index-url https://download.pytorch.org/whl/cu128`).

## Run
```
python agent_act.py --models mock:200 --runs 3                       # pipeline test, no model needed
python agent_act.py --models hf:Qwen/Qwen2.5-0.5B-Instruct --runs 10
python agent_act.py --require_cuda --runs 10 --models hf:... api:groq:openai/gpt-oss-20b api:gemini:gemini-3.8-flash
```
Model specs:
- `hf:<repo>` local Hugging Face model (GPU if available). With no `--models`, the 5 defaults in `DEFAULT_MODELS` run.
- `api:<provider>:<model>` provider is `groq`, `gemini` or `openai`.
- `mock:<ms>` fake agent that sleeps, for testing.

Options: `--runs` (timed runs per model, default 10), `--warmup` (untimed runs, default 1), `--max_new_tokens` (default 32), `--temperature` (default 0, greedy), `--dtype`, `--state` (custom state JSON), `--require_cuda` (abort if no GPU), `--out`, `--table`.

## Output
- `runs.jsonl` - one row per timed run: raw text, TTFT, total time, tokens, parse result, settings, hardware.
- `latency_table.md` - per-model median/mean/sd/min/max of TTFT and total time, tokens, valid % and strict-JSON %.

A model that fails to load or call (bad name, rate limit, gated repo) prints `SKIPPED` or `FAILED` and the sweep continues.

## Notes for reporting
- Fix and report one GPU, dtype, max tokens and temperature. Do not mix hardware in one table.
- Greedy decoding gives identical outputs across runs, so valid % will be 0 or 100. Use `--temperature` above 0 or more scenarios for a real rate.
- API calls use a 512-token budget and low reasoning effort, because those models spend hidden tokens before answering. API latency includes network time.
- The state's lane width, lane count and speed limit are placeholders until sourced values arrive.

## Run on Nautilus
Run these from WSL, in this folder, with your kubeconfig set to the `aiea-auditors` namespace.
```
kubectl get nodes -L nvidia.com/gpu.product                 # pick one GPU type; set it in k8s/av-latency-job.yaml
kubectl apply -f k8s/av-latency-pvc.yaml
tr -d '\r' < .env > /tmp/av.env
kubectl create secret generic av-latency-keys --from-env-file=/tmp/av.env ; rm /tmp/av.env
kubectl create configmap av-latency-code --from-file=agent_act.py
kubectl apply -f k8s/av-latency-job.yaml
kubectl logs -f job/emrlu-av-latency
```
Results are written to the PVC (`/pvcvolume/runs.jsonl`, `/pvcvolume/latency_table.md`) and the table prints at the end of the logs. After editing `agent_act.py`, delete and re-create the ConfigMap. When done:
```
kubectl delete job emrlu-av-latency
kubectl delete configmap av-latency-code
kubectl delete secret av-latency-keys
```

## Old script
```
python av_latency.py --model Qwen/Qwen2.5-0.5B-Instruct
```