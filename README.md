<div align="center">

# 🔍 DocLens-RAG

### Visual Retrieval-Augmented Generation for Complex PDFs

**Read the page, don't parse the text.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-4.57%2B-yellow)](https://github.com/huggingface/transformers)
[![Sentence--Transformers](https://img.shields.io/badge/Sentence--Transformers-5.4%2B%20%5Bimage%5D-orange)](https://github.com/UKPLab/sentence-transformers)
[![Models](https://img.shields.io/badge/Models-Qwen%20VL-green)](https://huggingface.co/Qwen)

</div>

---

## Overview

**DocLens-RAG** is a vision-first RAG pipeline for reports and layout-heavy PDFs. Instead of extracting text and chopping it into chunks, it **renders each PDF page as an image**, indexes those images with a **vision-language embedding model**, retrieves the most relevant pages, and hands them to a **vision-language model (VLM)** to produce a cited answer.

```
PDFs ─▶ page images ─▶ VL embeddings ─▶ ranking ─▶ VLM ─▶ cited answer
```

The core question it answers is **not** *"is visual retrieval always better?"* — it's *"does reading the page as an image give a more dependable retrieval path for the parts of a document that break text extraction?"* For dense financial tables, charts, and mixed-content pages, the answer is often yes.

---

## Why Visual RAG?

The traditional pipeline we started from:

```
PDFs ─▶ structure extraction (Docling) ─▶ HierarchicalChunker ─▶ noisy chunks
     ─▶ text embeddings ─▶ ranking ─▶ LLM
```

works when the PDF's structure and text layer are clean. But some tables in real annual reports are effectively impossible to extract correctly — they come out **distorted, noisy, or polluted**. Because both the embeddings and the final answer are built on those chunks, a single extraction error can silently poison retrieval *and* generation.

DocLens-RAG keeps the source appearance intact:

| | Text-based RAG | **DocLens-RAG (Visual)** |
|---|---|---|
| **Parsing** | Structure extraction + chunking | Render pages to images |
| **Tables / charts** | Often mangled by the extractor | Preserved exactly as printed |
| **Retrieval unit** | Text chunk | Page image (or tile/strip) |
| **Embedding** | Text embedding | Vision-language embedding |
| **Answer model** | Text LLM | Vision-language model |
| **Failure mode** | Extraction errors propagate | Limited by render resolution |
| **Provenance** | Chunk → page | Page image → `[file, p. N]` |

> **Note:** Visual retrieval is a *complement*, not a replacement. DocLens-RAG optionally fuses a lightweight **BM25 lexical path** over the PDF's native text layer to strengthen exact-value lookups.

---

## Architecture

```
┌────────────────────────── OFFLINE · run once ──────────────────────────┐
│                                                                        │
│  PDFs ──▶ pypdfium2 ──▶ page images ──▶ Qwen-VL-Embedding ──▶ index    │
│           render @DPI    (+ optional     (2B / 8B)           (.npy)    │
│                            tiles/strips)                               │
└────────────────────────────────────────────────────────────────────────┘

┌────────────────────────── ONLINE · per query ──────────────────────────┐
│                                                                        │
│  query ─▶ text embed ─▶ cosine ranking ─▶ [+ BM25 RRF] ─▶ [+ rerank]   │
│                                    │                                   │
│                                    ▼                                   │
│                          top-K page images                             │
│                                    │                                   │
│                                    ▼   (token-budget packing)          │
│                        VLM  (Qwen3.5 / Qwen3.8)                        │
│                                    │                                   │
│                                    ▼                                   │
│                    answer + [file, p. N] citations                     │
└────────────────────────────────────────────────────────────────────────┘
```

**Pipeline stages:**

1. **Render** — `pypdfium2` rasterizes every page at a configurable DPI (lossless PNG by default). Each document gets a fingerprint-keyed cache, so unchanged files are never re-rendered.
2. **Index units** — By default one vector per **whole page**. Optionally split pages into overlapping horizontal **tiles** (`--index_mode both`) for finer-grained retrieval on dense pages.
3. **Embed** — A multimodal `SentenceTransformer` (`Qwen3-VL-Embedding-2B/8B`) encodes each page image. Indexing is **resumable** via a partial `.npy` file.
4. **Retrieve** — The text query is embedded in the same space and matched by cosine similarity. Tile scores are **max-pooled** up to the page level.
5. **Fuse / rerank (optional)** — `--hybrid` adds BM25 over the native text layer and merges with **Reciprocal Rank Fusion**; `--reranker_model` adds a cross-encoder VL reranker.
6. **Answer** — Retrieved page images (each prefixed with a provenance header) are packed into the VLM prompt **within a strict token budget**, and the model answers with `[file, p. N]` citations.

---

## Features

- **Visual page-level indexing** — no text chunking, no extraction noise
- **Optional tile/strip indexing** for fine-grained retrieval on dense pages
- **Hybrid retrieval** — BM25 (native text layer) + RRF fusion for exact-value queries
- **Optional VL cross-encoder reranker** (`Qwen3-VL-Reranker-2B`)
- **Token-budget-aware prompting** — drops lowest-ranked pages until the prompt fits
- **Provenance headers** — every page carries source, company, entity, year, and page number for citation
- **Built-in evaluation harness** — hit@k, MRR, and answer-exactness, broken down by category
- **Resumable + cached indexing** — per-document render cache, fingerprint-keyed embedding cache, partial-file resume
- **4-bit quantization** (`--load_in_4bit`) and **Flash Attention → SDPA** auto-fallback
- **Multi-turn chat** with rolling history (images are not re-sent)

---

## Installation

Tested API levels — verify against your environment:

```bash
pip install -U pypdfium2 pillow \
    "sentence-transformers[image]>=5.4" \
    "transformers>=4.57" \
    accelerate rich numpy
```

Optional (recommended for large models):

```bash
pip install bitsandbytes flash-attn
```

> `sentence-transformers[image]>=5.4` is required for multimodal `encode_query` / `encode_document` and image-capable cross-encoders. `transformers>=4.57` provides `AutoModelForImageTextToText`.

Set your Hugging Face token if you use gated models:

```bash
export HUGGINGFACE_TOKEN=hf_xxx
```

## Dataset

DocLens-RAG includes a helper script to download a starter corpus of public automotive financial reports (BMW, Tesla, and Ford 10-Ks/Annual Reports). 

```bash
# 1. Make the script executable
chmod +x download_auto_financials.sh

# 2. Download the dataset (defaults to ./automotive_finance)
./download_auto_financials.sh

# Or specify a custom directory
./download_auto_financials.sh /path/to/my/pdfs
```

This will create a structured directory containing ~9 large, layout-heavy PDFs:
```text
automotive_finance/
├── BMW/
│   ├── BMW_Group_Report_2023.pdf
│   ├── BMW_Group_Report_2024.pdf
│   └── BMW_Group_Report_2025.pdf
├── Ford/
│   ├── Ford_Annual_Report_31_12_2023.pdf
│   ├── Ford_Annual_Report_31_12_2024.pdf
│   └── Ford_Annual_Report_31_12_2025.pdf
└── Tesla/
    ├── Tesla_10-K_31_12_2023.pdf
    ├── Tesla_10-K_31_12_2024.pdf
    └── Tesla_10-K_31_12_2025.pdf
```

Once downloaded, you can immediately test the RAG pipeline against it:
```bash
python run.py -d ./automotive_finance -vlm "Qwen/Qwen3.5-0.8B" -emb "Qwen/Qwen3-VL-Embedding-2B" -v
```

---

## Quick Start

```bash
# Minimal: index + chat with defaults (Qwen3.5-0.8B VLM, Qwen3-VL-Embedding-2B)
python run.py -d /path/to/pdfs
```

**Production example** (as run on the Roihu cluster):

```bash
python run.py \
  -d /scratch/project_2004072/datasets/automotive_finance \
  -vlm "Qwen/Qwen3.8-27B" \
  -emb "Qwen/Qwen3-VL-Embedding-8B" \
  --dpi 200 -bs 16 -v
```

---

## Usage

```bash
# Build the index only (e.g. as a batch job), then chat later
python run.py -d /path/to/pdfs --index_only -v

# Chat with hybrid text fusion + VL reranker
python run.py -d /path/to/pdfs \
  --hybrid \
  --reranker_model Qwen/Qwen3-VL-Reranker-2B

# Index page strips in addition to whole pages (better for dense tables)
python run.py -d /path/to/pdfs --index_mode both --tile_rows 3

# Run the evaluation harness (retrieval + optional generation)
python run.py -d /path/to/pdfs --eval_file questions.jsonl --eval_generate
```

---

## CLI Reference

### Required

| Flag | Description |
|---|---|
| `-d, --dataset_directory` | Directory containing the PDF corpus (searched recursively) |

### Models

| Flag | Default | Description |
|---|---|---|
| `-vlm, --vlm_model` | `Qwen/Qwen3.5-0.8B` | Vision-language model for answer generation |
| `-emb, --embedding_model` | `Qwen/Qwen3-VL-Embedding-2B` | Multimodal embedding model |
| `--reranker_model` | `None` | Optional VL cross-encoder reranker (e.g. `Qwen/Qwen3-VL-Reranker-2B`) |
| `--truncate_dim` | `None` | Matryoshka (MRL) dimension truncation for the embedder |
| `--query_instruction` | *(see below)* | Instruction prepended to queries; pass `none` to use the model default |

Default query instruction:
> *"Given a question about a company's annual report, retrieve the report pages (text, tables, or charts) that contain the answer."*

### Rendering & Indexing

| Flag | Default | Description |
|---|---|---|
| `--dpi` | `50` | Render resolution (higher = sharper small text, more VRAM/tokens) |
| `--image_format` | `png` | `png` (lossless) or `jpg` |
| `--index_mode` | `page` | `page`, `tiles`, or `both` |
| `--tile_rows` | `3` | Number of horizontal strips per page (tiling) |
| `--tile_overlap` | `0.15` | Overlap between strips |
| `--emb_max_pixels` | `1_000_000` | Pixel budget per unit when embedding |
| `-bs, --batch_size` | `4` | Embedding batch size |
| `--rebuild_index` | off | Force re-embedding even if a cache exists |
| `--index_only` | off | Build the index and exit (no chat) |

### Retrieval

| Flag | Default | Description |
|---|---|---|
| `-k, --top_k` | `5` | Pages retrieved / shown |
| `--candidate_k` | `10` | Candidates kept before rerank / fusion cut |
| `--hybrid` | off | Fuse BM25 (native PDF text layer) via RRF |
| `--rrf_k` | `60` | RRF smoothing constant |
| `--rerank_max_pixels` | `1_000_000` | Pixel budget per page during reranking |

### Generation

| Flag | Default | Description |
|---|---|---|
| `--max_pages_to_vlm` | `3` | Max pages forwarded to the VLM |
| `--vlm_max_pixels` | `1_600_000` | Pixel budget per page sent to the VLM |
| `--max_seq_len` | `16384` | VLM context window (tokens) |
| `--max_new_tokens` | `1024` | Max tokens to generate |
| `--load_in_4bit` | off | Load the VLM in 4-bit (bitsandbytes NF4) |

### Evaluation & Misc

| Flag | Default | Description |
|---|---|---|
| `--eval_file` | `None` | JSONL eval set (triggers evaluation mode) |
| `--eval_generate` | off | Also run the VLM during evaluation |
| `--eval_output` | `reportvision_eval.jsonl` | Output prefix for eval results |
| `-v, --verbose` | off | Verbose logging + retrieval panels |

---

## Evaluation

Provide a JSONL file where each line is a question with gold pages and (optionally) an expected answer:

```json
{"question": "What was Ford's revenue in 2023?", "relevant": [{"source": "Ford_Annual_Report_2023.pdf", "page": 54}], "answer": "176.2 billion", "category": "table"}
```

```bash
python run.py -d /path/to/pdfs --eval_file questions.jsonl --eval_generate
```

**Reported metrics** (aggregated overall and per `category`):

| Metric | Meaning |
|---|---|
| `hit@1 / @3 / @5 / @k` | A gold page appears within the top-K |
| `mrr` | Mean reciprocal rank of the first gold hit |
| `answer_contains_expected` | Normalized expected string appears in the VLM answer |

Outputs: a per-question `*.jsonl` and a `*_summary.json` (config + aggregated metrics).

> **Tip:** `category` (`table`, `chart`, `text`, `mixed`) lets you see whether visual retrieval actually helps on the layout types that motivated the project. Skew the eval set toward table-heavy questions if that's your target.

---

## Recommended Models & Hardware

| Component | Small GPU (~8–16 GB) | Large GPU (24 GB+) |
|---|---|---|
| **VLM** | `Qwen/Qwen3.5-0.8B` | `Qwen/Qwen3.8-27B` |
| **Embedder** | `Qwen/Qwen3-VL-Embedding-2B` | `Qwen/Qwen3-VL-Embedding-8B` |
| **Reranker** *(optional)* | `Qwen/Qwen3-VL-Reranker-2B` | `Qwen/Qwen3-VL-Reranker-2B` |

Practical guidance:

- **4-bit**: add `--load_in_4bit` to fit a larger VLM on a smaller card.
- **DPI vs. budget**: `--dpi 200` gives crisp tables but raises token counts. If you hit the context cap, lower `--vlm_max_pixels` or raise `--max_seq_len` before lowering DPI.
- **Embedder offload**: the embedder only encodes one short string per query; if VRAM is tight, build the index on the large GPU (`--index_only`) and serve chat with a lighter setup.
- **Batch size**: `-bs 16` works well on large GPUs; drop to `2–4` on smaller ones.

---

## Caching & Reproducibility

DocLens-RAG is built for iterative research, so it caches aggressively:

- **Rendered pages** — per-document cache keyed by `path | size | mtime` + DPI. Unchanged PDFs are never re-rendered.
- **Embeddings** — keyed by a hash of *all* indexing hyperparameters (`embedding_model`, `truncate_dim`, `emb_max_pixels`, `index_mode`, `tile_rows`, `tile_overlap`, `dpi`, `image_format`) **plus every PDF fingerprint**. Change any knob → automatic rebuild.
- **Resumable embedding** — a `.partial.npy` is written as blocks complete, so an interrupted indexing job picks up where it stopped.
- **Rebuild manually** with `--rebuild_index`.

Cache roots are resolved per-user via the `cache_directory` map at the top of `run.py` (falling back to the current working directory).

---

## Limitations

- A **whole-page embedding is a coarse representation**; a page with several unrelated tables/charts may retrieve for the wrong reason. Use `--index_mode both` (tiles) or hybrid fusion to mitigate.
- **Small print and footnotes** depend on render resolution and the VLM's pixel budget — relevant details can be missed if under-rendered.
- Visual retrieval locates material but does **not guarantee exact-value accuracy**; for precise figures, combine with `--hybrid` and verify against the source page.
- Rendering, embedding, and multi-page VLM inference are **more compute- and storage-intensive** than text RAG. Caching and pixel budgets matter as the corpus grows.

---

## Acknowledgments

Built on [pypdfium2](https://github.com/pypdfium2-team/pypdfium2), [sentence-transformers](https://github.com/UKPLab/sentence-transformers), [transformers](https://github.com/huggingface/transformers), and the [Qwen](https://huggingface.co/Qwen) vision-language model family.

---

<div align="center">

*If DocLens-RAG helps your research or product, consider starring the repo.*

</div>