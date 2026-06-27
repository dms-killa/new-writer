# DraftFlow

A local, long-form content generation pipeline powered by Ollama. Takes a topic and produces a polished Markdown article through a 4-phase pipeline — no cloud APIs required.

## Prerequisites

- **Python 3.11+**
- **Ollama** running locally with GPU acceleration
- Models pulled and ready:
  ```bash
  ollama pull llama3.1:70b-instruct-q4_K_M
  ollama pull phi3.5:3.8b-mini-instruct-q4
  ```

## Installation

```bash
pip install requests pydantic
```

No other dependencies are needed. The pipeline uses only `requests`, `pydantic`, `tomllib` (built-in), and the Python standard library.

## Usage

```bash
# Basic usage
python draftflow.py "The Future of Renewable Energy"

# Custom output directory
python draftflow.py "AI Ethics in Healthcare" --output-dir ./my-article

# With structured evaluation (gap analysis + drift report)
python draftflow.py "Quantum Computing Applications" --evaluate

# Custom config path
python draftflow.py "Urban Planning" --config /path/to/config.toml
```

## The 4-Phase Pipeline

### Phase 1: Scaffolding
Generates a hierarchical outline with intent sentences and counterpoints using the primary model. Outlines use Markdown headings (`##`, `###`, `#### Intent:`) for deterministic parsing — no JSON needed at this stage. Gaps are identified and folded into the outline.

### Phase 2: Chunked Drafting
Each outline node is drafted as 1-3 paragraphs with context injection:
- **Global anchor** (thesis statement) referenced in every chunk
- **Previous chunk's** last sentence for continuity
- **Next node's** intent for forward coherence
- **Temperature interpolation** from creative (0.9) to precise (0.3) across the article

Introduction and conclusion are generated first to bookend the draft.

### Phase 3: Bi-Directional Polish
Two fixed editing passes in alternating order:
1. Smooth (readability) → Refactor (tighten by up to 15%)
2. Refactor → Smooth

Followed by a reverse-outline check: the model identifies the 5 central claims and inserts supporting evidence where missing, outputting the complete revised draft.

### Phase 4: Final Lenses
A single low-temperature pass applies four micro-edits:
- **Hook/Closer**: Ensures the opening and closing form a thematic loop
- **Tone**: Normalizes voice throughout
- **Redundancy**: Removes repeated phrases
- **Logic**: Verifies causal claims have evidence

## Output Structure

```
workspace/run_YYYYMMDD_HHMMSS/
├── logs/
│   └── draftflow.log
├── chunks/
│   ├── intro.md
│   ├── node_001.md
│   ├── node_002.md
│   └── conclusion.md
├── outline.md
├── draft_full.md
├── final.md
├── gap_report.json      (if --evaluate)
└── drift_report.json    (if --evaluate)
```

Each run gets a timestamped directory — previous runs are never overwritten.

## Configuration

Edit `config.toml` (created automatically on first run):

```toml
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
```

## Troubleshooting

### VRAM Requirements
- **Primary model (70B Q4)**: ~40 GB VRAM. Use a smaller quant or model if limited (e.g., `llama3.1:8b-instruct-q4_K_M`).
- **Secondary model (3.8B)**: ~3 GB VRAM. Runs comfortably on most GPUs.

### Context Overflow
The pipeline estimates tokens as `len(text) // 4` and warns when nearing context limits:
- **Primary overflow**: The draft is split by `##` headings and processed section-by-section.
- **Secondary overflow**: The task is escalated to the primary model (logged as a warning).

Reduce `primary_context` / `secondary_context` in `config.toml` if you see truncation.

### Connection Errors
The client retries up to 3 times with exponential backoff. Ensure Ollama is running:
```bash
ollama serve
```

### Structured Output Validation
Structured outputs (gap analysis, drift report) use Ollama's constrained decoding — the model cannot emit malformed JSON when a schema is provided. If validation errors occur, the prompt is automatically refined and retried. After 2 failures, a `RuntimeError` is raised with details to help you adjust the prompt or schema.
