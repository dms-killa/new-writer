#!/usr/bin/env python3
"""DraftFlow — Local long-form content generation pipeline using Ollama."""

# This pipeline uses Ollama's local constrained decoding (BNF grammar).
# It NEVER calls external APIs for structured output. If validation fails,
# fix the prompt or schema, not the inference backend.

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from pydantic import BaseModel, ValidationError

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

# ---------------------------------------------------------------------------
# Default configuration (used when config.toml is absent)
# ---------------------------------------------------------------------------
DEFAULTS: dict[str, Any] = {
    "ollama": {
        "host": "http://localhost:11434",
        "primary_model": "llama3.1:70b-instruct-q4_K_M",
        "secondary_model": "phi3.5:3.8b-mini-instruct-q4",
        "primary_context": 32768,
        "secondary_context": 8192,
    },
    "draft_temperature": {"start": 0.9, "middle": 0.7, "end": 0.3},
    "pipeline": {"max_refactor_cut_percent": 15},
}

DEFAULT_CONFIG_TOML = """\
[ollama]
host = "http://localhost:11434"
primary_model = "llama3.1:70b-instruct-q4_K_M"
secondary_model = "phi3.5:3.8b-mini-instruct-q4"
primary_context = 32768
secondary_context = 8192

[draft_temperature]
start = 0.9
middle = 0.7
end = 0.3

[pipeline]
max_refactor_cut_percent = 15
"""

# ---------------------------------------------------------------------------
# Pydantic models for structured‑output stages
# ---------------------------------------------------------------------------

class GapReport(BaseModel):
    gaps: list[str]
    suggested_subpoints: list[str]


class DriftReport(BaseModel):
    severity: float
    deviated_nodes: list[str]


# ---------------------------------------------------------------------------
# Configuration loader
# ---------------------------------------------------------------------------

def load_config(path: str = "config.toml") -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        p.write_text(DEFAULT_CONFIG_TOML)
        logging.info("Created default config.toml")
    with open(p, "rb") as f:
        return tomllib.load(f)


# ---------------------------------------------------------------------------
# Ollama HTTP client wrapper
# ---------------------------------------------------------------------------

class OllamaClient:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.host = cfg["ollama"]["host"].rstrip("/")
        self.primary = cfg["ollama"]["primary_model"]
        self.secondary = cfg["ollama"]["secondary_model"]
        self.primary_ctx = int(cfg["ollama"]["primary_context"])
        self.secondary_ctx = int(cfg["ollama"]["secondary_context"])
        self.logger = logging.getLogger("ollama_client")

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return len(text) // 4

    def _post(self, payload: dict, retries: int = 3) -> dict:
        url = f"{self.host}/api/chat"
        for attempt in range(1, retries + 1):
            try:
                resp = requests.post(url, json=payload, timeout=600)
                resp.raise_for_status()
                return resp.json()
            except (requests.RequestException, requests.ConnectionError) as exc:
                self.logger.warning("Attempt %d/%d failed: %s", attempt, retries, exc)
                if attempt == retries:
                    raise
                time.sleep(2 ** attempt)
        raise RuntimeError("unreachable")

    def _context_limit(self, model: str) -> int:
        if model == self.primary:
            return self.primary_ctx
        return self.secondary_ctx

    def _exceeds_context(self, text: str, model: str, threshold: float = 0.85) -> bool:
        return self._estimate_tokens(text) > threshold * self._context_limit(model)

    def _split_by_headings(self, text: str) -> list[str]:
        sections: list[str] = []
        current: list[str] = []
        for line in text.splitlines(keepends=True):
            if line.startswith("## ") and current:
                sections.append("".join(current))
                current = [line]
            else:
                current.append(line)
        if current:
            sections.append("".join(current))
        return sections

    def chat(
        self,
        prompt: str,
        model: str | None = None,
        temperature: float = 0.7,
        schema: dict | None = None,
        context_text: str | None = None,
    ) -> str:
        model = model or self.primary
        effective_text = context_text or prompt

        if self._exceeds_context(effective_text, model):
            if model == self.primary:
                self.logger.warning(
                    "Primary context overflow — splitting by ## headings"
                )
                sections = self._split_by_headings(effective_text)
                results = []

                base_instruction = ""
                if context_text and prompt.endswith(context_text):
                    base_instruction = prompt[:-len(context_text)].strip()

                for sec in sections:
                    if base_instruction:
                        chunk_prompt = f"{base_instruction}\n\n{sec}"
                        chunk_ctx = sec
                    else:
                        chunk_prompt = sec
                        chunk_ctx = None

                    results.append(
                        self.chat(
                            prompt=chunk_prompt,
                            model=model,
                            temperature=temperature,
                            schema=schema,
                            context_text=chunk_ctx,
                        )
                    )
                return "\n\n".join(results)
            else:
                self.logger.warning(
                    "Secondary model context overflow — escalating to primary"
                )
                model = self.primary

        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": temperature},
        }
        if schema is not None:
            payload["format"] = schema
            payload["options"]["temperature"] = 0

        data = self._post(payload)
        return data["message"]["content"]

    def structured_chat(
        self,
        prompt: str,
        pydantic_model: type[BaseModel],
        model: str | None = None,
        max_retries: int = 2,
    ) -> BaseModel:
        schema = pydantic_model.model_json_schema()
        full_prompt = f"{prompt}\n\nRespond using this JSON schema:\n{json.dumps(schema, indent=2)}"
        for attempt in range(1, max_retries + 1):
            raw = self.chat(prompt=full_prompt, model=model, schema=schema)
            try:
                return pydantic_model.model_validate_json(raw)
            except ValidationError as exc:
                self.logger.error(
                    "Validation failed (attempt %d/%d): %s", attempt, max_retries, exc
                )
                if attempt == max_retries:
                    raise RuntimeError(
                        f"Structured output validation failed after {max_retries} "
                        f"retries: {exc}"
                    ) from exc
                full_prompt = (
                    f"{prompt}\n\nYour previous response did not match the schema. "
                    f"You MUST respond with valid JSON matching this schema exactly:\n"
                    f"{json.dumps(schema, indent=2)}"
                )
        raise RuntimeError("unreachable")


# ---------------------------------------------------------------------------
# Phase 1 — Scaffolding
# ---------------------------------------------------------------------------

def parse_outline(text: str) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    current_section: dict[str, Any] | None = None
    current_sub: dict[str, Any] | None = None

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            current_section = {
                "title": stripped[3:].strip(),
                "subpoints": [],
                "intent": "",
                "counterpoint": "",
            }
            sections.append(current_section)
            current_sub = None
        elif stripped.startswith("### ") and current_section is not None:
            label = stripped[4:].strip()
            if label.lower().startswith("gap:"):
                current_sub = {
                    "title": label,
                    "intent": "",
                    "counterpoint": "",
                    "is_gap": True,
                }
            else:
                current_sub = {
                    "title": label,
                    "intent": "",
                    "counterpoint": "",
                    "is_gap": False,
                }
            current_section["subpoints"].append(current_sub)
        elif stripped.startswith("#### Intent:"):
            intent_text = stripped[len("#### Intent:"):].strip()
            if current_sub is not None:
                current_sub["intent"] = intent_text
            elif current_section is not None:
                current_section["intent"] = intent_text
        elif stripped.startswith("#### Counterpoint:"):
            cp_text = stripped[len("#### Counterpoint:"):].strip()
            if current_sub is not None:
                current_sub["counterpoint"] = cp_text
            elif current_section is not None:
                current_section["counterpoint"] = cp_text

    return sections


def scaffold(client: OllamaClient, topic: str) -> tuple[list[dict], str, str]:
    prompt = (
        f'Generate a hierarchical outline for "{topic}" using EXACTLY this format:\n'
        "## Section Title\n"
        "### Subpoint\n"
        "#### Intent: What this subpoint must achieve.\n"
        "#### Counterpoint: (for each main argument, add one counterargument to address)\n\n"
        "Also identify gaps: add a ## Gaps section with ### Gap: items.\n"
    )
    raw = client.chat(prompt, temperature=0.7)
    sections = parse_outline(raw)
    thesis = sections[0]["title"] if sections else topic
    return sections, thesis, raw


# ---------------------------------------------------------------------------
# Phase 2 — Chunked Drafting
# ---------------------------------------------------------------------------

def interpolate_temp(progress: float, cfg: dict[str, Any]) -> float:
    start = cfg["draft_temperature"]["start"]
    middle = cfg["draft_temperature"]["middle"]
    end = cfg["draft_temperature"]["end"]
    if progress <= 0.5:
        return start - (progress * 2) * (start - middle)
    return middle - ((progress - 0.5) * 2) * (middle - end)


def flatten_nodes(sections: list[dict]) -> list[dict]:
    nodes: list[dict] = []
    for sec in sections:
        nodes.append(sec)
        for sub in sec.get("subpoints", []):
            nodes.append(sub)
    return nodes


def draft(
    client: OllamaClient,
    sections: list[dict],
    global_anchor: str,
    cfg: dict[str, Any],
    run_dir: Path,
) -> Path:
    chunks_dir = run_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    intro_prompt = (
        f"You are writing a long-form article with this thesis: \"{global_anchor}\".\n"
        f"Write a compelling 2-3 paragraph introduction that sets up the topic and "
        f"states the thesis clearly. Output Markdown only."
    )
    intro = client.chat(intro_prompt, temperature=cfg["draft_temperature"]["start"])
    (chunks_dir / "intro.md").write_text(intro)

    conclusion_prompt = (
        f"You are writing a long-form article with this thesis: \"{global_anchor}\".\n"
        f"Write a strong 2-3 paragraph conclusion that ties everything together and "
        f"reinforces the thesis. Output Markdown only."
    )
    conclusion = client.chat(conclusion_prompt, temperature=cfg["draft_temperature"]["end"])
    (chunks_dir / "conclusion.md").write_text(conclusion)

    nodes = flatten_nodes(sections)
    total = len(nodes)
    all_chunks: list[str] = [intro]
    prev_last_sentence = ""

    for idx, node in enumerate(nodes):
        progress = idx / max(total - 1, 1)
        temp = interpolate_temp(progress, cfg)
        intent = node.get("intent", "") or f"Discuss {node['title']}"
        next_intent = nodes[idx + 1].get("intent", "") if idx + 1 < total else ""
        counterpoint = node.get("counterpoint", "")

        prompt_parts = [
            f"Global thesis (reference at least once): \"{global_anchor}\"",
            f"Current section: {node['title']}",
            f"Intent: {intent}",
        ]
        if counterpoint:
            prompt_parts.append(f"Address this counterargument: {counterpoint}")
        if prev_last_sentence:
            prompt_parts.append(
                f"Previous chunk ended with: \"{prev_last_sentence}\""
            )
        if next_intent:
            prompt_parts.append(f"Next section's intent: {next_intent}")
        prompt_parts.append(
            "Write 1-3 focused paragraphs for this section. Reference the thesis "
            "at least once. Output Markdown only."
        )
        chunk_prompt = "\n".join(prompt_parts)

        chunk = client.chat(chunk_prompt, temperature=temp)
        node_id = f"node_{idx + 1:03d}"
        (chunks_dir / f"{node_id}.md").write_text(chunk)
        all_chunks.append(chunk)

        sentences = [s.strip() for s in chunk.replace("\n", " ").split(".") if s.strip()]
        prev_last_sentence = (sentences[-1] + ".") if sentences else ""

        logging.info("Drafted %s (%.0f%% progress, temp=%.2f)", node_id, progress * 100, temp)

    all_chunks.append(conclusion)
    draft_path = run_dir / "draft_full.md"
    draft_path.write_text("\n\n---\n\n".join(all_chunks))
    return draft_path


# ---------------------------------------------------------------------------
# Phase 3 — Bi‑Directional Polish
# ---------------------------------------------------------------------------

def _smooth(client: OllamaClient, text: str, temp: float) -> str:
    prompt = (
        "You are an expert editor. Smooth the following draft for readability: "
        "fix awkward transitions, improve sentence flow, and ensure consistent "
        "voice. Preserve all content and meaning. Output the revised draft only.\n\n"
        + text
    )
    return client.chat(prompt, temperature=temp, context_text=text)


def _refactor(client: OllamaClient, text: str, cfg: dict[str, Any]) -> str:
    cut = cfg["pipeline"]["max_refactor_cut_percent"]
    prompt = (
        f"You are a concise editor. Tighten the following draft: remove filler, "
        f"merge redundant sentences, and cut up to {cut}% of the word count while "
        f"preserving all key arguments. Output the revised draft only.\n\n"
        + text
    )
    model = client.secondary
    if client._exceeds_context(text, model):
        logging.warning("Secondary model overflow for refactor — using primary")
        model = client.primary
    return client.chat(prompt, model=model, temperature=0.4, context_text=text)


def polish(client: OllamaClient, draft_path: Path, cfg: dict[str, Any]) -> None:
    text = draft_path.read_text()

    logging.info("Polish pass 1: Smooth → Refactor")
    text = _smooth(client, text, 0.8)
    text = _refactor(client, text, cfg)

    logging.info("Polish pass 2: Refactor → Smooth")
    text = _refactor(client, text, cfg)
    text = _smooth(client, text, 0.6)

    logging.info("Reverse outline check")
    ro_prompt = (
        "Read the entire draft. Identify its 5 central claims. For each claim "
        "that lacks explicit supporting evidence, insert a new sentence immediately "
        "after the claim that provides plausible reasoning. Do not output a list of "
        "claims. Output ONLY the fully revised draft with these insertions.\n\n"
        + text
    )
    text = client.chat(ro_prompt, temperature=0.5, context_text=text)
    draft_path.write_text(text)


# ---------------------------------------------------------------------------
# Phase 4 — Final Lenses
# ---------------------------------------------------------------------------

def lenses(client: OllamaClient, draft_path: Path, run_dir: Path) -> Path:
    text = draft_path.read_text()
    prompt = (
        "Apply final edits to the following draft in a single pass: ensure the "
        "hook and closer form a loop, tone is consistent throughout, remove "
        "redundant phrases, and verify every causal claim has evidence. Output "
        "the revised draft only.\n\n"
        + text
    )
    final = client.chat(prompt, temperature=0.1, context_text=text)
    out = run_dir / "final.md"
    out.write_text(final)
    return out


# ---------------------------------------------------------------------------
# Optional evaluation stages (--evaluate)
# ---------------------------------------------------------------------------

def evaluate_gaps(client: OllamaClient, outline_raw: str) -> GapReport:
    prompt = (
        "Analyze the following outline and identify content gaps — topics that "
        "are missing or underrepresented. For each gap, suggest a subpoint that "
        "would fill it.\n\n" + outline_raw
    )
    return client.structured_chat(prompt, GapReport)


def evaluate_drift(client: OllamaClient, thesis: str, draft_text: str) -> DriftReport:
    prompt = (
        f"The thesis of this article is: \"{thesis}\".\n\n"
        f"Analyze the following draft and determine how much it has drifted from "
        f"the thesis. Rate severity from 0.0 (no drift) to 1.0 (completely off-topic). "
        f"List any nodes/sections that deviate.\n\n{draft_text}"
    )
    return client.structured_chat(prompt, DriftReport)


# ---------------------------------------------------------------------------
# CLI & main
# ---------------------------------------------------------------------------

def setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "draftflow.log"),
            logging.StreamHandler(sys.stderr),
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="draftflow",
        description="Generate polished long-form articles using a local Ollama pipeline.",
    )
    parser.add_argument("topic", help="The topic for the article")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: workspace/run_YYYYMMDD_HHMMSS/)",
    )
    parser.add_argument(
        "--config",
        default="config.toml",
        help="Path to config.toml (default: ./config.toml)",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run structured evaluation stages (gap analysis, drift report)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    for section, defaults in DEFAULTS.items():
        if section not in cfg:
            cfg[section] = defaults
        elif isinstance(defaults, dict):
            for k, v in defaults.items():
                cfg[section].setdefault(k, v)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) if args.output_dir else Path(f"workspace/run_{stamp}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "chunks").mkdir(exist_ok=True)

    setup_logging(run_dir / "logs")
    logger = logging.getLogger("draftflow")
    logger.info("Starting DraftFlow for topic: %s", args.topic)
    logger.info("Run directory: %s", run_dir)

    client = OllamaClient(cfg)

    # Phase 1 — Scaffolding
    logger.info("Phase 1: Scaffolding")
    sections, thesis, outline_raw = scaffold(client, args.topic)
    (run_dir / "outline.md").write_text(outline_raw)
    logger.info("Outline generated with %d sections, thesis: %s", len(sections), thesis)

    # Phase 2 — Chunked Drafting
    logger.info("Phase 2: Chunked Drafting")
    draft_path = draft(client, sections, thesis, cfg, run_dir)
    logger.info("Draft complete: %s", draft_path)

    # Phase 3 — Bi‑Directional Polish
    logger.info("Phase 3: Bi-Directional Polish")
    polish(client, draft_path, cfg)
    logger.info("Polish complete")

    # Phase 4 — Final Lenses
    logger.info("Phase 4: Final Lenses")
    final_path = lenses(client, draft_path, run_dir)
    logger.info("Final article: %s", final_path)

    # Optional evaluation
    if args.evaluate:
        logger.info("Running evaluation stages")
        gap_report = evaluate_gaps(client, outline_raw)
        logger.info("Gap report: %s", gap_report.model_dump_json(indent=2))
        (run_dir / "gap_report.json").write_text(gap_report.model_dump_json(indent=2))

        draft_text = draft_path.read_text()
        drift_report = evaluate_drift(client, thesis, draft_text)
        logger.info("Drift report: %s", drift_report.model_dump_json(indent=2))
        (run_dir / "drift_report.json").write_text(drift_report.model_dump_json(indent=2))

    logger.info("DraftFlow complete. Output: %s", final_path)
    print(f"\nDone! Final article: {final_path}")


if __name__ == "__main__":
    main()
