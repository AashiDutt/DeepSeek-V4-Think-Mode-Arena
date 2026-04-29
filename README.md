# ⚔️ DeepSeek V4 Think Mode Arena

A Streamlit app that runs the same prompt across **Non-think**, **Think High**, and **Think Max** in parallel and compares results side-by-side.

## What it compares

| Metric | Details |
|---|---|
| Answer | Full response, with collapsible thinking trace |
| Latency | Wall-clock seconds |
| Output tokens | Total completion tokens |
| Tok/s | Throughput |
| Cost | Estimated USD (approximate) |
| Thinking words | Size of the internal reasoning trace |
| User rating | 1–5 stars via slider |

## Task templates

| Task | Expected winner | Why |
|---|---|---|
| 📄 Trivial / Lookup (JSON→YAML, one-sentence summary) | Non-think | No reasoning needed |
| 🐛 Coding / Debugging | Think High | Best quality-per-dollar for bugs |
| 🏗️ System Design | Think High | Depth without overkill |
| 📅 Planning (6-month roadmap) | Think High | Structure > exhaustion |
| 🧮 Math (IMO-style proof) | Think Max | Verification is critical |

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

Enter your DeepSeek API key in the sidebar (get it at platform.deepseek.com).

## Model names

The app defaults to `deepseek-v4-flash`. If you hit a model-not-found error (V4 API is freshly released), try editing the `MODELS` dict in `app.py`:

```python
MODELS = {
    "DeepSeek-V4-Flash": "deepseek-chat",        # fallback
    "DeepSeek-V4-Pro":   "deepseek-reasoner",    # fallback
}
```

## Architecture notes

- All 3 modes fire **concurrently** via `ThreadPoolExecutor(max_workers=3)` — total latency ≈ slowest single mode
- Think modes are controlled via system prompt until V4 API docs publish official mode parameters
- `<think>...</think>` blocks in responses are parsed and shown separately in a collapsible trace
- Pricing is approximate — verify at platform.deepseek.com before publishing cost comparisons
