"""
DeepSeek V4 Think Mode Arena
Run the same prompt across Non-think, Think High, and Think Max in parallel.
Compare quality, latency, tokens, cost, and user ratings side-by-side.
"""

import os
import time
import concurrent.futures
from dataclasses import dataclass
from typing import Optional, Dict

import streamlit as st
from openai import OpenAI


# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_API_KEY_ENV = "DEEPSEEK_API_KEY"

# ── Model names 
MODELS = {
    "DeepSeek-V4-Flash (default)": "deepseek-v4-flash",
    "DeepSeek-V4-Pro": "deepseek-v4-pro",
}

# ── Three reasoning modes — controlled via official V4 thinking parameters
MODES: Dict[str, dict] = {
    "Non-think": {
        "icon": "⚡",
        "color": "#10b981",
        "badge": "green",
        "desc": "Fast, direct answers — no internal reasoning",
        "thinking_type": "disabled",
        "reasoning_effort": None,
    },
    "Think High": {
        "icon": "🧠",
        "color": "#3b82f6",
        "badge": "blue",
        "desc": "Careful step-by-step reasoning before responding",
        "thinking_type": "enabled",
        "reasoning_effort": "high",
    },
    "Think Max": {
        "icon": "🔥",
        "color": "#ef4444",
        "badge": "red",
        "desc": "Exhaustive reasoning — push analysis to the limit",
        "thinking_type": "enabled",
        "reasoning_effort": "max",
    },
}

# ── Pricing per 1M tokens
# These values reflect the official DeepSeek pricing page as of 2026-04-29.
# Cache-hit pricing changed on 2026-04-26, and V4 Pro is currently discounted.
PRICING = {
    "deepseek-v4-flash": {
        "input_cache_hit": 0.0028,
        "input_cache_miss": 0.14,
        "output": 0.28,
    },
    "deepseek-v4-pro": {
        "input_cache_hit": 0.003625,
        "input_cache_miss": 0.435,
        "output": 0.87,
    },
}

# ── Task templates 
TASKS = {
    "Trivial / Lookup": {
        "prompt": (
            "Complete both tasks below:\n\n"
            "Task A — Convert this JSON to YAML:\n"
            '{"name": "Alice", "age": 30, "skills": ["Python", "ML", "LLMs"],\n'
            ' "address": {"city": "San Francisco", "zip": "94105"}}\n\n'
            "Task B — Summarize this paragraph in exactly one sentence:\n"
            '"Large language models have rapidly transformed natural language processing '
            "by demonstrating unprecedented capabilities across translation, summarization, "
            "reasoning, and code generation, driven by scale and alignment techniques like RLHF "
            'that bring model outputs closer to human intent."'
        ),
        "expected_winner": "Non-think",
        "tip": "Non-think should dominate here. No reasoning is required.",
    },
    "Coding / Debugging": {
        "prompt": (
            "Find every bug in the Python code below, explain each bug clearly, "
            "and provide a fully corrected version:\n\n"
            "```python\n"
            "def binary_search(arr, target):\n"
            "    left, right = 0, len(arr)\n"
            "    while left < right:\n"
            "        mid = (left + right) // 2\n"
            "        if arr[mid] == target:\n"
            "            return mid\n"
            "        elif arr[mid] < target:\n"
            "            left = mid\n"
            "        else:\n"
            "            right = mid - 1\n"
            "    return -1\n\n"
            "print(binary_search([1, 3, 5, 7, 9], 7))\n"
            "```"
        ),
        "expected_winner": "Think High",
        "tip": "Think High usually finds the bugs and explains them well.",
    },
    "System Design": {
        "prompt": (
            "Design a scalable vector search system for 100 million documents.\n\n"
            "Address each of the following:\n"
            "1. Indexing strategy and pipeline\n"
            "2. ANN algorithm selection (HNSW vs IVF-PQ vs ScaNN — justify your choice)\n"
            "3. Sharding and replication strategy\n"
            "4. p99 query latency target (< 50 ms) — how do you hit it?\n"
            "5. Real-time document update handling\n"
            "6. Top 3 failure modes and their mitigations"
        ),
        "expected_winner": "Think High",
        "tip": "Think High often gives the best quality-per-dollar design answer.",
    },
    "Planning": {
        "prompt": (
            "Create a detailed 6-month roadmap for deploying an enterprise RAG system.\n\n"
            "Include:\n"
            "- Month-by-month phases with concrete, measurable milestones\n"
            "- Top 5 risks and mitigation strategies\n"
            "- Team roles and headcount required per phase\n"
            "- Evaluation metrics for each phase (how do you know it's working?)\n"
            "- Go / no-go production checklist"
        ),
        "expected_winner": "Think High",
        "tip": "Think High should produce a more structured, complete roadmap.",
    },
    "Math (IMO-style)": {
        "prompt": (
            "Solve this problem completely and verify your answer:\n\n"
            "Find all positive integers n such that n² + 1 is divisible by n + 1.\n\n"
            "Your answer must include:\n"
            "1. A complete proof with clear logical steps\n"
            "2. Verification with at least 3 concrete numerical examples\n"
            "3. A rigorous argument for why your solution set is complete "
            "(i.e., there are no other solutions)"
        ),
        "expected_winner": "Think Max",
        "tip": "Think Max earns its cost when thorough verification matters.",
    },
}


# ═══════════════════════════════════════════════════════════════
# DATA CLASS
# ═══════════════════════════════════════════════════════════════

@dataclass
class RunResult:
    mode: str
    answer: str = ""
    thinking: str = ""
    latency: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    error: Optional[str] = None

    @property
    def tokens_per_second(self) -> Optional[float]:
        if self.latency > 0 and self.output_tokens > 0:
            return self.output_tokens / self.latency
        return None

    @property
    def thinking_word_count(self) -> int:
        return len(self.thinking.split()) if self.thinking else 0


# ═══════════════════════════════════════════════════════════════
# API LOGIC
# ═══════════════════════════════════════════════════════════════

def _get_cached_prompt_tokens(usage) -> int:
    """Best-effort extraction of cached prompt tokens from the SDK response."""
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    if prompt_details is None:
        return 0

    cached_tokens = getattr(prompt_details, "cached_tokens", None)
    if cached_tokens is not None:
        return cached_tokens or 0

    if isinstance(prompt_details, dict):
        return prompt_details.get("cached_tokens", 0) or 0

    return 0


def _estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int, cached_prompt_tokens: int) -> float:
    pricing = PRICING.get(model, PRICING["deepseek-v4-flash"])
    cached_tokens = min(cached_prompt_tokens, prompt_tokens)
    uncached_tokens = max(prompt_tokens - cached_tokens, 0)
    return (
        cached_tokens / 1_000_000 * pricing["input_cache_hit"]
        + uncached_tokens / 1_000_000 * pricing["input_cache_miss"]
        + completion_tokens / 1_000_000 * pricing["output"]
    )

def call_mode(client: OpenAI, model: str, mode_name: str, user_prompt: str) -> RunResult:
    """Call the DeepSeek API for one mode and return a RunResult."""
    result = RunResult(mode=mode_name)
    mode_cfg = MODES[mode_name]
    start = time.perf_counter()

    try:
        request_kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": user_prompt}],
            "max_tokens": 4096,
            "extra_body": {"thinking": {"type": mode_cfg["thinking_type"]}},
        }
        if mode_cfg["reasoning_effort"]:
            request_kwargs["reasoning_effort"] = mode_cfg["reasoning_effort"]

        response = client.chat.completions.create(
            **request_kwargs,
        )
        result.latency = time.perf_counter() - start

        message = response.choices[0].message
        result.thinking = (getattr(message, "reasoning_content", None) or "").strip()
        result.answer = (message.content or "").strip()

        usage = response.usage
        result.input_tokens = getattr(usage, "prompt_tokens", 0) or 0
        result.output_tokens = getattr(usage, "completion_tokens", 0) or 0
        result.cost_usd = _estimate_cost_usd(
            model=model,
            prompt_tokens=result.input_tokens,
            completion_tokens=result.output_tokens,
            cached_prompt_tokens=_get_cached_prompt_tokens(usage),
        )

    except Exception as exc:
        result.latency = time.perf_counter() - start
        result.error = str(exc)

    return result


def run_parallel(client: OpenAI, model: str, prompt: str) -> Dict[str, RunResult]:
    """Fire all 3 modes concurrently and return results keyed by mode name."""
    results: Dict[str, RunResult] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            pool.submit(call_mode, client, model, mode_name, prompt): mode_name
            for mode_name in MODES
        }
        for fut in concurrent.futures.as_completed(futures):
            results[futures[fut]] = fut.result()
    return results


# ═══════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════

def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

        :root {
            --bg-top: #fcf6ee;
            --bg-bottom: #f3efe8;
            --panel: rgba(255, 252, 247, 0.88);
            --border: rgba(108, 87, 52, 0.14);
            --text: #1f2937;
            --muted: #6b7280;
            --heading: #111827;
            --green-soft: #e7f7ef;
            --blue-soft: #e8f1ff;
            --red-soft: #fdecec;
            --shadow: 0 24px 60px rgba(73, 49, 16, 0.08);
        }

        html, body, [class*="css"] {
            font-family: 'DM Sans', sans-serif;
            color: var(--text);
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(250, 204, 21, 0.10), transparent 30%),
                radial-gradient(circle at top right, rgba(59, 130, 246, 0.08), transparent 28%),
                linear-gradient(180deg, var(--bg-top) 0%, var(--bg-bottom) 100%);
        }

        [data-testid="stAppViewContainer"] {
            background: transparent;
        }

        [data-testid="stHeader"] {
            background: rgba(255, 255, 255, 0.6);
            backdrop-filter: blur(12px);
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1380px;
        }

        .section-eyebrow {
            color: #8b5e34;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }

        .section-title {
            color: var(--heading);
            font-size: 1.45rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }

        .section-copy {
            color: var(--muted);
            font-size: 0.98rem;
            margin-bottom: 1rem;
        }

        .arena-header {
            background:
                linear-gradient(145deg, rgba(18, 29, 58, 0.96) 0%, rgba(31, 48, 91, 0.93) 58%, rgba(70, 49, 124, 0.90) 100%);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 24px;
            padding: 2.4rem 2.5rem;
            margin-bottom: 1.75rem;
            box-shadow: 0 30px 80px rgba(28, 33, 65, 0.26);
            overflow: hidden;
            position: relative;
        }
        .arena-header::after {
            content: "";
            position: absolute;
            inset: auto -8% -35% auto;
            width: 260px;
            height: 260px;
            border-radius: 999px;
            background: radial-gradient(circle, rgba(255, 255, 255, 0.16), transparent 65%);
        }
        .arena-header h1 {
            font-family: 'Space Mono', monospace;
            font-size: 2.15rem;
            color: #f9fafb;
            margin: 0 0 0.6rem 0;
            letter-spacing: -0.5px;
        }
        .arena-header p {
            color: rgba(245, 247, 250, 0.82);
            font-size: 1rem;
            margin: 0 0 1.2rem 0;
            max-width: 780px;
            line-height: 1.7;
        }
        .hero-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.7rem;
        }
        .hero-chip {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.62rem 0.95rem;
            border-radius: 999px;
            font-size: 0.86rem;
            color: #f8fafc;
            background: rgba(255, 255, 255, 0.10);
            border: 1px solid rgba(255, 255, 255, 0.18);
        }

        .mode-header {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.9rem 1rem;
            border-radius: 16px;
            margin-bottom: 0.55rem;
            font-family: 'Space Mono', monospace;
            font-size: 0.92rem;
            font-weight: 700;
        }
        .mode-green  { background: var(--green-soft); color: #047857; border: 1px solid rgba(4, 120, 87, 0.16); }
        .mode-blue   { background: var(--blue-soft); color: #1d4ed8; border: 1px solid rgba(29, 78, 216, 0.16); }
        .mode-red    { background: var(--red-soft); color: #dc2626; border: 1px solid rgba(220, 38, 38, 0.16); }

        .subtle-copy {
            color: var(--muted);
            font-size: 0.92rem;
            margin-bottom: 0.9rem;
        }

        .metric-chip {
            display: inline-flex;
            align-items: center;
            gap: 0.3rem;
            background: rgba(255, 255, 255, 0.85);
            border: 1px solid rgba(84, 59, 30, 0.12);
            border-radius: 999px;
            padding: 0.36rem 0.72rem;
            font-family: 'Space Mono', monospace;
            font-size: 0.75rem;
            color: #6b7280;
            margin: 0.2rem;
        }
        .metric-chip span { color: #111827; font-weight: 700; }

        .info-card,
        .winner-badge,
        .empty-state {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 22px;
            box-shadow: var(--shadow);
        }

        .info-card {
            padding: 1.15rem 1.2rem;
            margin-bottom: 0.95rem;
        }
        .info-card h3 {
            margin: 0;
            color: var(--heading);
            font-size: 1.12rem;
        }
        .info-card p {
            color: var(--muted);
            margin: 0.55rem 0 0 0;
            line-height: 1.65;
        }
        .info-card .row {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            margin-top: 0.9rem;
            padding-top: 0.9rem;
            border-top: 1px solid rgba(84, 59, 30, 0.10);
            color: var(--muted);
            font-size: 0.92rem;
        }
        .info-card .row strong {
            color: var(--heading);
            font-size: 0.96rem;
        }
        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            margin-top: 0.95rem;
            padding: 0.55rem 0.8rem;
            border-radius: 999px;
            font-size: 0.84rem;
            font-weight: 600;
        }
        .status-ready {
            background: rgba(16, 185, 129, 0.10);
            color: #047857;
        }
        .status-waiting {
            background: rgba(245, 158, 11, 0.12);
            color: #b45309;
        }

        .winner-badge {
            display: inline-block;
            padding: 0.9rem 1rem;
            text-align: center;
            width: 100%;
        }
        .winner-badge .label {
            font-size: 0.72rem;
            color: #6b7280;
            font-family: 'Space Mono', monospace;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .winner-badge .name  {
            font-size: 1rem;
            font-weight: 700;
            margin-top: 0.3rem;
            color: var(--heading);
        }
        .winner-badge .value {
            font-size: 0.8rem;
            color: var(--muted);
            font-family: 'Space Mono', monospace;
            margin-top: 0.25rem;
        }

        .stat-card {
            background: rgba(255, 255, 255, 0.78);
            border: 1px solid rgba(84, 59, 30, 0.12);
            border-radius: 18px;
            padding: 0.9rem 1rem;
            min-height: 112px;
            box-shadow: 0 18px 40px rgba(73, 49, 16, 0.06);
        }
        .stat-label {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #8b5e34;
            margin-bottom: 0.45rem;
            font-weight: 700;
        }
        .stat-value {
            color: var(--heading);
            font-size: 1.25rem;
            font-weight: 700;
            line-height: 1.2;
        }
        .stat-note {
            color: var(--muted);
            font-size: 0.86rem;
            margin-top: 0.45rem;
            line-height: 1.5;
        }

        .tip-banner {
            background: rgba(29, 78, 216, 0.08);
            border: 1px solid rgba(29, 78, 216, 0.12);
            color: #1d4ed8;
            padding: 0.95rem 1rem;
            border-radius: 16px;
            margin-bottom: 0.95rem;
            font-size: 0.95rem;
        }

        .progress-note {
            color: var(--heading);
            font-size: 0.95rem;
            font-weight: 600;
            margin: 0.2rem 0 0.8rem 0;
        }

        .answer-label {
            color: #8b5e34;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin: 1rem 0 0.5rem 0;
        }

        .empty-state {
            padding: 1.4rem 1.5rem;
            margin-top: 1.15rem;
        }
        .empty-state h3 {
            color: var(--heading);
            margin: 0 0 0.35rem 0;
            font-size: 1.08rem;
        }
        .empty-state p {
            color: var(--muted);
            margin: 0;
            line-height: 1.7;
        }

        .stProgress > div > div { background: #2563eb !important; }

        hr {
            border-color: rgba(84, 59, 30, 0.10) !important;
            margin: 1.4rem 0 !important;
        }

        section[data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, rgba(255, 250, 243, 0.98) 0%, rgba(246, 239, 227, 0.96) 100%);
            border-right: 1px solid rgba(84, 59, 30, 0.08);
        }
        section[data-testid="stSidebar"] .block-container {
            padding-top: 1.25rem;
        }

        .sidebar-title {
            color: var(--heading);
            font-size: 1.35rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }
        .sidebar-copy {
            color: var(--muted);
            font-size: 0.92rem;
            line-height: 1.6;
            margin-bottom: 1rem;
        }

        div[data-baseweb="select"] > div,
        .stTextInput > div > div > input,
        .stTextArea textarea {
            border-radius: 16px !important;
            border: 1px solid rgba(84, 59, 30, 0.14) !important;
            background: rgba(255, 255, 255, 0.92) !important;
        }

        .stTextArea textarea {
            line-height: 1.65 !important;
        }

        .stButton > button {
            border-radius: 14px;
            min-height: 2.8rem;
            font-weight: 600;
            border: 1px solid rgba(84, 59, 30, 0.12);
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.4rem;
        }
        .stTabs [data-baseweb="tab"] {
            background: rgba(255, 255, 255, 0.7);
            border: 1px solid rgba(84, 59, 30, 0.10);
            border-radius: 999px;
            padding: 0.45rem 0.9rem;
        }
        .stTabs [aria-selected="true"] {
            background: #111827 !important;
            color: #f9fafb !important;
        }

        .stAlert {
            border-radius: 16px;
            border: 1px solid rgba(84, 59, 30, 0.08);
        }

        [data-testid="stTable"] {
            background: rgba(255, 255, 255, 0.78);
            border-radius: 18px;
            padding: 0.2rem 0.45rem;
            border: 1px solid rgba(84, 59, 30, 0.10);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def stat_card(label: str, value: str, note: str) -> str:
    return (
        '<div class="stat-card">'
        f'<div class="stat-label">{label}</div>'
        f'<div class="stat-value">{value}</div>'
        f'<div class="stat-note">{note}</div>'
        "</div>"
    )


def render_intro():
    st.markdown(
        """
        <div class="arena-header">
            <h1>DeepSeek V4 Think Mode Arena</h1>
            <p>Compare the same prompt across three reasoning styles in one run. This layout is built to help you spot the fastest answer, the best answer, and the answer that was actually worth the cost.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state():
    st.markdown(
        """
        <div class="empty-state">
            <h3>Results will appear here</h3>
            <p>Once you run the arena, you’ll get a friendlier side-by-side comparison with winners, costs, response speed, and the full answers in separate tabs.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_run_snapshot(results: Dict[str, RunResult], model_label: str, task_label: str):
    valid = [result for result in results.values() if not result.error]
    total_cost = sum(result.cost_usd for result in valid)
    fastest = min(valid, key=lambda item: item.latency) if valid else None
    total_output = sum(result.output_tokens for result in valid)

    cols = st.columns(4)
    snapshot_cards = [
        ("Run type", task_label, "The current prompt template"),
        ("Model", model_label, "Same model used for all three modes"),
        ("Total cost", f"${total_cost:.5f}" if valid else "—", "Combined estimated spend for this run"),
        ("Fastest answer", f"{fastest.mode} · {fastest.latency:.1f}s" if fastest else "—", f"{total_output:,} output tokens across successful runs" if valid else "No successful runs yet"),
    ]

    for col, (label, value, note) in zip(cols, snapshot_cards):
        with col:
            st.markdown(stat_card(label, value, note), unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# UI RENDERERS
# ═══════════════════════════════════════════════════════════════

def render_mode_column(result: RunResult, mode_name: str):
    cfg = MODES[mode_name]
    badge_cls = f"mode-{cfg['badge']}"

    st.markdown(
        f'<div class="mode-header {badge_cls}">{mode_name}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(f'<div class="subtle-copy">{cfg["desc"]}</div>', unsafe_allow_html=True)

    if result.error:
        st.error(f"**API Error:** {result.error}")
        return

    chips = [
        ("Latency", f"{result.latency:.1f}s"),
        ("Output tokens", f"{result.output_tokens:,}"),
        ("Cost", f"${result.cost_usd:.5f}"),
    ]
    if result.tokens_per_second:
        chips.append(("Tok/s", f"{result.tokens_per_second:.0f}"))

    chip_html = "".join(
        f'<span class="metric-chip">{label} <span>{val}</span></span>'
        for label, val in chips
    )
    st.markdown(chip_html, unsafe_allow_html=True)

    if result.thinking:
        with st.expander(f"🔍 Thinking trace — {result.thinking_word_count:,} words"):
            preview = result.thinking[:5000]
            if len(result.thinking) > 5000:
                preview += "\n\n[… truncated for display …]"
            st.text(preview)
    elif mode_name != "Non-think":
        st.caption("_No thinking trace emitted_")

    st.markdown('<div class="answer-label">Final answer</div>', unsafe_allow_html=True)
    st.markdown(result.answer if result.answer else "_No answer returned._")


def render_metrics_table(results: Dict[str, RunResult], ratings: Dict[str, int]):
    rows = []
    for mode_name, res in results.items():
        ok = not res.error
        rows.append({
            "Mode": mode_name,
            "Latency (s)":     f"{res.latency:.2f}" if ok else "—",
            "Input Tokens":    f"{res.input_tokens:,}" if ok else "—",
            "Output Tokens":   f"{res.output_tokens:,}" if ok else "—",
            "Tok/s":           f"{res.tokens_per_second:.0f}" if (ok and res.tokens_per_second) else "—",
            "Est. Cost (USD)": f"${res.cost_usd:.5f}" if ok else "—",
            "Thinking Words":  f"{res.thinking_word_count:,}" if ok else "—",
            "User Rating":  f"{ratings.get(mode_name)}/5" if ratings.get(mode_name) else "—",
        })
    st.table(rows)


def render_winner_summary(results: Dict[str, RunResult], ratings: Dict[str, int], expected: str):
    valid = {k: v for k, v in results.items() if not v.error}
    if not valid:
        st.warning("No valid results to summarise.")
        return

    fastest       = min(valid, key=lambda k: valid[k].latency)
    cheapest      = min(valid, key=lambda k: valid[k].cost_usd)
    most_efficient = max(
        valid,
        key=lambda k: valid[k].output_tokens / max(valid[k].cost_usd, 1e-9),
    )
    top_rated = max(ratings, key=ratings.get) if ratings else None

    def badge(label: str, color: str, mode: str, value: str) -> str:
        return (
            f'<div class="winner-badge" style="border-color:{color}33;">'
            f'<div class="label">{label}</div>'
            f'<div class="name" style="color:{color};">{mode}</div>'
            f'<div class="value">{value}</div>'
            f'</div>'
        )

    cols = st.columns(4)
    with cols[0]:
        st.markdown(
            badge("Fastest", "#10b981", fastest, f"{valid[fastest].latency:.1f}s"),
            unsafe_allow_html=True,
        )
    with cols[1]:
        st.markdown(
            badge("Cheapest", "#f59e0b", cheapest, f"${valid[cheapest].cost_usd:.5f}"),
            unsafe_allow_html=True,
        )
    with cols[2]:
        st.markdown(
            badge("Most Efficient", "#8b5cf6", most_efficient, "best tok/$"),
            unsafe_allow_html=True,
        )
    with cols[3]:
        if top_rated:
            st.markdown(
                badge("Top Rated", "#ef4444", top_rated, f"{ratings[top_rated]}/5"),
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="winner-badge"><div class="label">Top Rated</div>'
                '<div class="name" style="color:#555;">Rate answers above</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown(
        f"\n> **Expected winner for this task type:** `{expected}` — "
        "does your result match? Rate answers to confirm.",
        unsafe_allow_html=False,
    )


# ═══════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="DeepSeek V4 Think Mode Arena",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()

    for key, default in [
        ("results", None),
        ("ratings", {}),
        ("run_task", None),
        ("run_model_label", None),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    with st.sidebar:
        st.markdown('<div class="sidebar-title">Setup</div>', unsafe_allow_html=True)
        api_key = os.getenv(DEEPSEEK_API_KEY_ENV, "")

        model_label = st.selectbox("Model", list(MODELS.keys()), index=0)
        selected_model = MODELS[model_label]

    render_intro()

    # st.markdown('<div class="section-eyebrow">Step 2</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Build the prompt you want to test</div>', unsafe_allow_html=True)

    task_label = st.selectbox(
        "Choose a starting template",
        list(TASKS.keys()),
        help="Load a sample prompt and edit it before running.",
    )
    task = TASKS[task_label]

    st.markdown(f'<div class="tip-banner">{task["tip"]}</div>', unsafe_allow_html=True)
    prompt = st.text_area(
        "Prompt",
        value=task["prompt"],
        height=300,
        help="You can keep the template, tweak it, or replace it completely.",
    )

    prompt_words = len(prompt.split())
    prompt_chars = len(prompt)
    prompt_lines = len(prompt.splitlines())

    stat_cols = st.columns(3)
    with stat_cols[0]:
        st.markdown(
            stat_card("Prompt size", f"{prompt_words:,} words", "A quick sense of prompt complexity"),
            unsafe_allow_html=True,
        )
    with stat_cols[1]:
        st.markdown(
            stat_card("Characters", f"{prompt_chars:,}", "Useful when you are trimming prompt length"),
            unsafe_allow_html=True,
        )
    with stat_cols[2]:
        st.markdown(
            stat_card("Lines", f"{prompt_lines:,}", "Helpful for longer structured or coding prompts"),
            unsafe_allow_html=True,
        )

    c_run, c_clear, _ = st.columns([1.2, 1, 4.8])
    run_btn = c_run.button("Run Arena", type="primary", disabled=not api_key)
    clear_btn = c_clear.button("Clear")

    if not api_key:
        st.caption(f"Set `{DEEPSEEK_API_KEY_ENV}` in your environment to enable runs.")
    else:
        st.caption("Everything is set. When you run, all three modes will start in parallel.")

    if clear_btn:
        st.session_state.results = None
        st.session_state.ratings = {}
        st.session_state.run_task = None
        st.session_state.run_model_label = None
        st.rerun()

    if run_btn:
        st.session_state.ratings = {}
        client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)

        progress_note = st.empty()
        progress_note.markdown(
            '<div class="progress-note">Running all three modes in parallel...</div>',
            unsafe_allow_html=True,
        )
        progress = st.progress(0)
        completed_results: Dict[str, RunResult] = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures = {
                pool.submit(call_mode, client, selected_model, mode_name, prompt): mode_name
                for mode_name in MODES
            }
            done = 0
            for fut in concurrent.futures.as_completed(futures):
                mode_name = futures[fut]
                completed_results[mode_name] = fut.result()
                done += 1
                progress.progress(done / 3)
                progress_note.markdown(
                    f'<div class="progress-note">{mode_name} complete ({done}/3)</div>',
                    unsafe_allow_html=True,
                )

        progress.empty()
        progress_note.empty()
        st.session_state.results = completed_results
        st.session_state.run_task = task_label
        st.session_state.run_model_label = model_label
        st.markdown(
            '<div class="progress-note">Run complete. Compare the results below.</div>',
            unsafe_allow_html=True,
        )

    if st.session_state.results:
        results: Dict[str, RunResult] = st.session_state.results
        run_task_label = st.session_state.run_task or task_label
        run_task = TASKS.get(run_task_label, task)
        run_model_label = st.session_state.run_model_label or model_label

        st.divider()
        st.markdown('<div class="section-title">Review the comparison</div>', unsafe_allow_html=True)
        render_run_snapshot(results, run_model_label, run_task_label)

        overview_tab, answers_tab, ratings_tab = st.tabs(
            ["Overview", "Full Responses", "Ratings"]
        )

        with overview_tab:
            st.markdown("### Winner Summary")
            render_winner_summary(
                results,
                st.session_state.ratings,
                expected=run_task.get("expected_winner", "—"),
            )
            st.markdown("### Metrics Comparison")
            st.caption(
                "Cost values are estimates based on current published V4 pricing. "
                "Cached prompt discounts and promotional rates may change."
            )
            render_metrics_table(results, st.session_state.ratings)

        with answers_tab:
            st.markdown("### Side-by-side Answers")
            cols = st.columns(3)
            for i, mode_name in enumerate(MODES):
                with cols[i]:
                    render_mode_column(results[mode_name], mode_name)

        with ratings_tab:
            st.markdown("### Rate the Answers")
            st.caption("Score each mode from 1 (poor) to 5 (excellent). The overview updates after each change.")

            r_cols = st.columns(3)
            for i, mode_name in enumerate(MODES):
                with r_cols[i]:
                    rating = st.slider(
                        mode_name,
                        min_value=1,
                        max_value=5,
                        value=3,
                        key=f"rating_{mode_name}_{st.session_state.run_task}",
                    )
                    st.session_state.ratings[mode_name] = rating

            if st.session_state.ratings:
                leader = max(st.session_state.ratings, key=st.session_state.ratings.get)
                st.markdown(
                    f'<div class="progress-note">Current top-rated answer: {leader} with {st.session_state.ratings[leader]} / 5.</div>',
                    unsafe_allow_html=True,
                )
    else:
        render_empty_state()


if __name__ == "__main__":
    main()
