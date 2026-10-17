# DocLens-RAG
retrieving whole pages (or tiles) via a "lens" (VL embeddings) rather than chunking text

VL embeddings are trained on a large corpus of documents, so they can be used to retrieve pages from a smaller corpus.

## Usage

```bash
python run.py -d /path/to/pdfs --index_mode both --tile_rows 3
```

This will index all PDFs in `/path/to/pdfs` and create a JSONL file `vindex_page_tiles_3x3.jsonl` with the following format:

```json
{
  "index_key": "page|tiles|both:300|80|1.0",
  "embedding_model": "Qwen/Qwen3-VL-Embedding-2B",
  "n_units": 100,
  "n_pages": 100,
  "dim": 300,
  "index_mode": "both",
  "dpi": 80,
  "emb_max_pixels": 1000000,
  "tile_rows": 3,
  "tile_overlap": 0.15,
  "n_pages_without_text_layer": 0,
  "mean_size_px": [
    0.0,
    0.0,
    ...
  ],
  "std_size_px": [
    0.0,
    0.0,
    ...
  ]
}
```

The `index_key` is a string that uniquely identifies the index. It includes the embedding model, the index mode, and the tile size. The index is stored in `vindex_page_tiles_3x3.npy`.

To retrieve pages, run:

```bash
python run.py -d /path/to/pdfs --eval_file questions.jsonl --eval_generate
```

This will generate answers for each question in `questions.jsonl` and save them to `reportvision_eval.jsonl`.

## Models

The default models are fine-tuned on the [DocLens-RAG dataset](https://github.com/doclens/DocLens-RAG-Dataset).

| Model | Description | Size |
| --- | --- | --- |
| Qwen/Qwen3-VL-Embedding-2B | VL embeddings trained on DocLens-RAG | 300-dim |
| Qwen/Qwen3-VL-Reranker-2B | VL reranker trained on DocLens-RAG | 300-dim |
