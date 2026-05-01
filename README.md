# The Enrichment Crossover Effect

Code for "The Enrichment Crossover Effect: Context Granularity and Knowledge Enrichment in Geospatial Prediction".

External knowledge enrichment (e.g., Overture Maps place categories) improves geospatial prediction at low behavioral context granularity but **hurts** at high granularity. This crossover is consistent across LightGBM, MLPs, zero-shot LLMs, and fine-tuned Llama 3.1 8B, on friendship prediction, next-location ranking, taxi trip tasks, and Yelp review prediction.

## 1. Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # add your OpenAI API key
```

Python 3.10+ required. GPU scripts (Llama fine-tuning) additionally need `torch`, `unsloth`, and a CUDA GPU.

## 2. Data Download

All raw data must be downloaded by the user. Create the directory structure below and place files as indicated.

### Gowalla

Download from [SNAP](https://snap.stanford.edu/data/loc-gowalla.html):

```
data/raw/gowalla/
├── loc-gowalla_edges.txt
└── loc-gowalla_totalCheckins.txt
```

### Overture Maps (for enrichment)

The enrichment script downloads Overture places automatically via DuckDB's S3 integration. Alternatively, pre-download a US places parquet:

```
data/raw/gowalla/overture_us_places.parquet
```

### NYC Taxi

```bash
bash scripts/download_nyc_taxi_data.sh
```

This downloads 2024 Yellow Taxi trip parquets and taxi zone shapefiles into `data/raw/nyc_taxi/`.

### Yelp

Download the [Yelp Academic Dataset](https://www.yelp.com/dataset) and place:

```
data/raw/yelp/
├── yelp_academic_dataset_business.json
└── yelp_academic_dataset_review.json
```

## 3. Reproduce Results

All scripts are run from the repository root. Each dataset follows a **preprocess → enrich → train/evaluate** pipeline.

### Gowalla Friendship Prediction

```bash
# Preprocess raw check-ins and edges
python scripts/gowalla_preprocess.py

# Enrich locations with Overture Maps categories
python scripts/gowalla_enrich_overture.py
python scripts/gowalla_enrich_preprocess.py

# ML training (LightGBM + MLP) across all tiers and granularity levels
python scripts/gowalla_handcrafted_train.py           # venue_id tier
python scripts/gowalla_latlng_handcrafted_train.py     # latlng tier
python scripts/gowalla_enrich_handcrafted_train.py     # enriched tier

# Zero-shot LLM experiments (requires OpenAI API key)
python scripts/gowalla_enrich_experiment.py

# Prepare Llama fine-tuning JSONL (all tiers)
python scripts/gowalla_llama_prepare.py                # enriched
python scripts/gowalla_llama_prepare_nonenriched.py     # venue_id
python scripts/gowalla_llama_prepare_latlng.py          # latlng
```

### Gowalla Next-Location Ranking

```bash
python scripts/gowalla_nextloc_preprocess.py
python scripts/gowalla_nextloc_ml_train.py
python scripts/gowalla_nextloc_llama_prepare.py
```

### NYC Taxi (Ranking + Duration)

```bash
python scripts/nyc_taxi_preprocess.py
python scripts/nyc_taxi_zone_enrichment.py
python scripts/nyc_taxi_ml_train.py
python scripts/nyc_taxi_llama_prepare.py
```

### Yelp (Star Rating + Next-Business Ranking)

```bash
python scripts/yelp_preprocess.py
python scripts/yelp_handcrafted_train.py
python scripts/yelp_experiment.py

python scripts/yelp_ranking_preprocess.py
python scripts/yelp_ranking_handcrafted_train.py
python scripts/yelp_ranking_experiment.py
```

### Analysis (Crossover, SHAP, UMAP, Ablation)

```bash
python scripts/crossover_analysis.py
python scripts/feature_importance_analysis.py
python scripts/generate_umap_progression.py
python scripts/gowalla_logreg_comparison.py
```

## 4. Llama Fine-Tuning

The `*_llama_prepare.py` scripts output JSONL files ready for fine-tuning. To fine-tune Llama 3.1 8B Instruct with QLoRA on a GPU:

```bash
python scripts/llama_finetune_nextloc.py      # next-location task
python scripts/llama_zeroshot_friendship.py    # zero-shot friendship evaluation
```

These require `unsloth`, `torch`, and a CUDA-capable GPU (tested on A100 40GB). Adjust paths at the top of each script for your data directory.

## 5. Configuration

Each dataset has a YAML config controlling data paths, granularity levels, models, and experiment parameters:

- `gowalla_config.yaml` — Gowalla friendship + next-location
- `nyc_taxi_config.yaml` — NYC Taxi ranking + duration
- `yelp_config.yaml` — Yelp star rating
- `yelp_ranking_config.yaml` — Yelp next-business ranking

## Citation

```bibtex
@article{kumar2026enrichment,
  title={The Enrichment Crossover Effect: Context Granularity and Knowledge Enrichment in Geospatial Prediction},
  year={2026}
}
```

## License

MIT
