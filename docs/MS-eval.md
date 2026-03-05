# CLAUDE.md — Multi-Stakeholder RecBole Evaluation Framework

## Role and Mission

You are a specialised coding assistant for a PhD project that builds a **multi-stakeholder evaluation framework** on top of RecBole. The framework evaluates recommendation quality not just at the system level, but across stakeholder groups (consumers, providers, system) and stakeholder values (usefulness, fairness, novelty, monetary reward, etc.).

Your primary mission for this session is to **design and implement the ms_eval module**: a post-hoc evaluation layer that sits downstream of RecBole's standard inference pipeline, consumes RecBole output plus metadata, and produces a structured multi-stakeholder evaluation report.

---

## Strict Behaviour Rules

1. **Read before writing.** Before designing or coding anything, read the existing codebase thoroughly. Use `find`, `cat`, and `ls` to understand what already exists. Never assume structure.
2. **Never modify RecBole internals.** All new code must live outside the RecBole package directory. RecBole is a dependency, not a codebase to edit. Never edit 'recbole/' and never edit the files 'data_config_convert/conversion/src/base_dataset.py' and 'data_config_convert/conversion/src/kg_dataset.py'
3. **Prefer extension over reimplementation.** Reuse RecBole's dataset loading, config system, and existing metrics where they already work. Only add what is genuinely missing.
4. **Explicit over implicit.** Configuration drives behaviour. No hardcoded dataset names, metric lists, or stakeholder fields inside logic modules.
5. **Incremental complexity.** Start with the minimal working structure. Add complexity only when the simpler version is verified.
6. **Explain decisions.** When you make a non-obvious architectural choice, include a short comment explaining why.

---

## Project Context

### What this project is

A PhD research project building a generic multi-stage evaluation framework for multi-stakeholder recommender systems. The first deliverable (this session) is the offline evaluation component for a single RecBole model, with evaluation at three levels:

- **System level** — aggregate metrics over all users and items
- **Stakeholder level** — metrics grouped by stakeholder role (consumer, provider, system) with optional filtering by metadata attributes
- **Value level** — metrics organised by stakeholder value (usefulness, fairness, novelty, monetary reward, recognition, diversity, etc.)

### Datasets being used (start here)

- **MovieLens-1M** — has user metadata (age, gender, occupation) and item metadata (genres), but NO provider data. Use for consumer-side and system-side evaluation.
- **Steam** — has item metadata (price, genres, developer/publisher as provider), but NO user demographics. Use for provider-side and system-side evaluation.

### What RecBole already provides

RecBole's built-in evaluator computes these metrics after training:

```
auc, averagepopularity, gauc, giniindex, hit, itemcoverage,
logloss, mae, map, mrr, ndcg, precision, recall, rmse,
shannonentropy, tailpercentage
```

These are system-level aggregates only. They do not support stakeholder-level breakdowns or value-level organisation. The ms_eval module must extend these.

---

## Step 1: Read the Codebase (Do This First)

Before writing any code, execute the following reads in order:

```bash
# 1. Understand the top-level repo structure
ls -la

# 2. Find all Python files in the project (excluding RecBole internals)
find . -name "*.py" -not -path "*/recbole/*" -not -path "*/.venv/*" | sort

# 3. Find config files
find . -name "*.yaml" -o -name "*.yml" -o -name "*.cfg" | grep -v ".venv" | sort

# 4. Read the main entry point(s)
# Look for run.py, main.py, train.py, or similar

# 5. Read any existing evaluation-related code
find . -name "*eval*" -o -name "*metric*" -o -name "*assess*" | grep -v ".venv" | grep -v "recbole/"

# 6. Read the dataset configuration to understand how provider fields are currently handled
# Look for how item fields like developer/publisher/artist are specified in configs
```

After reading, summarise:
- Where does training/inference currently run from?
- How are dataset configs structured (which fields map to users, items, interactions)?
- Is there any existing provider field configuration, even if ad-hoc?
- Where do RecBole's evaluation results currently get saved?

---

## Step 2: Understand the Provider Field Problem

This is a critical design decision. Currently, a token field in the item data (e.g., `developer` in Steam, or `genres` in MovieLens) is being used informally as a provider proxy. This is fragile. 

**Your job:** After reading the config files, propose a clean provider configuration scheme. The options are:

**Option A — Item field designation in YAML config**
```yaml
# In the ms_eval config section
stakeholder_config:
  provider:
    field: developer        # column in item feature file
    type: token             # token (one provider per item) or token_seq (multiple)
    filter_fields:          # optional metadata fields for filtering
      - country
      - category
  consumer:
    filter_fields:
      - age
      - gender
      - occupation
```
*Pros:* Simple, config-driven, no new files needed. *Cons:* One field only, no rich provider metadata.

**Option B — Separate provider mapping file**
```
# provider.csv (sits alongside item feature file)
item_id, provider_id, provider_name, provider_country, provider_size_tier
```
With a config entry:
```yaml
stakeholder_config:
  provider_file: provider.csv
  provider_id_field: provider_id
```
*Pros:* Richer provider metadata, separates concerns cleanly, supports multiple providers per item. *Cons:* Requires a new file for each dataset.

**Option C — Augment existing item feature file**
Add provider columns directly to the `.item` atomic file RecBole already loads.
*Pros:* RecBole already loads item features, no new file handling. *Cons:* Pollutes the item feature namespace.

**Recommendation to evaluate:** Option A for MVP (MovieLens, Steam already have the field in item files). Option B for later when richer provider metadata is needed. Read the actual configs before deciding.

---

## Step 3: Design the ms_eval Module

### Target directory structure

```
ms_eval/
├── __init__.py
├── config.py              # loads and validates ms_eval YAML config
├── loader.py              # loads RecBole output + metadata into evaluation objects
├── metrics/
│   ├── __init__.py
│   ├── consumer.py        # consumer-side metrics (accuracy, diversity, novelty, calibration)
│   ├── provider.py        # provider-side metrics (exposure share, Gini, hit rate)
│   ├── system.py          # system-side metrics (coverage, revenue-weighted NDCG, popularity bias)
│   └── registry.py        # metric name → function mapping, value → metric mapping
├── evaluator.py           # orchestrates metric computation across all stakeholders
├── report.py              # structures results into the report object
└── output/
    ├── csv_writer.py      # writes results to CSV
    └── dashboard.py       # Streamlit/Dash interactive dashboard
```

### Core data flow

```
RecBole inference
      ↓
topk_scores (users × items tensor or dict)   ← from Trainer.evaluate()
      ↓
ms_eval Loader
  - loads topk_scores
  - loads item metadata (from RecBole dataset object or file)
  - loads user metadata (if available)
  - resolves provider field per config
      ↓
ms_eval Evaluator
  - computes consumer metrics (per user, then aggregated + grouped)
  - computes provider metrics (per provider, then aggregated)
  - computes system metrics (global)
      ↓
EvaluationReport object
  {
    system: {metric_name: value, ...},
    consumer: {
      aggregate: {metric_name: value},
      by_group: {group_field: {group_value: {metric_name: value}}}
    },
    provider: {
      aggregate: {metric_name: value},
      by_provider: {provider_id: {metric_name: value}},
      by_group: {group_field: {group_value: {metric_name: value}}}
    },
    value_index: {value_name: {metric_name: value, ...}}   ← cross-cutting view
  }
      ↓
Output: CSV / Dashboard / Python dict
```

---

## Step 4: Metric Implementation Specification

### Metrics to implement (priority order)

#### Consumer metrics (`consumer.py`)

These operate on the per-user top-N list.

**From RecBole (reuse or replicate):**
- `ndcg@K`, `precision@K`, `recall@K`, `hit@K`, `map@K`, `mrr@K`
- `average_popularity` (mean popularity of recommended items)
- `tail_percentage` (fraction of recommended items that are long-tail)

**New — must implement:**

`intra_list_diversity(topk_items, item_features, feature_field)`:
- For each user, compute mean pairwise dissimilarity among top-N items
- Dissimilarity = 1 - Jaccard similarity on categorical feature sets (e.g., genres)
- Return mean ILD across users
- Requires: item descriptive features (category/genre)

`novelty(topk_items, item_popularity)`:
- For each recommended item, compute `-log2(p(item))` where `p(item) = interaction_count / total_interactions`
- Return mean novelty across users and items
- Requires: item interaction counts (derivable from training data)

`calibration(topk_items, user_history, item_features, feature_field)`:
- For each user, compute KL divergence between genre distribution in recommendations and genre distribution in user's history
- Lower = better calibrated
- Requires: item features + user history

`performance_disparity(per_user_metric, user_metadata, group_field)`:
- Given a per-user metric (e.g., NDCG per user), compute mean per demographic group
- Return dict of {group_value: mean_metric} and the max-min gap as a disparity score
- Requires: user metadata with group_field

#### Provider metrics (`provider.py`)

These operate on the per-provider view of recommendations.

`provider_exposure_share(topk_items, item_provider_map)`:
- For each provider, count total appearances across all users' top-N lists
- Normalise to share of total recommendation slots
- Return dict {provider_id: share}

`provider_exposure_gini(provider_exposure_shares)`:
- Gini coefficient over the provider_exposure_share distribution
- 0 = perfectly equal, 1 = one provider gets everything
- Return scalar

`provider_hit_rate(topk_items, test_interactions, item_provider_map)`:
- For each provider, among items that appear in test set, what fraction appear in top-N for that user?
- This is recall at the provider level
- Return dict {provider_id: hit_rate}

`min_provider_exposure(provider_exposure_shares, percentile=10)`:
- Return the exposure share at the given percentile (worst-case fairness)

`provider_performance_disparity(per_provider_metric, provider_metadata, group_field)`:
- Same structure as consumer disparity but over providers grouped by metadata field
- e.g., exposure by provider country or size tier

#### System metrics (`system.py`)

`item_coverage(topk_items, total_items)`:
- Fraction of catalogue that appears at least once across all recommendations
- Reuse RecBole's ItemCoverage if accessible, else reimplement

`revenue_weighted_ndcg(topk_items, item_prices, test_interactions, K)`:
- Standard NDCG but gain = price × relevance instead of binary relevance
- Requires: item price field

`popularity_bias_delta(topk_items, test_interactions, item_popularity)`:
- Mean popularity of recommended items minus mean popularity of relevant items
- Positive = system over-recommends popular items
- Requires: item interaction counts

`catalog_coverage_by_group(topk_items, item_features, group_field)`:
- ItemCoverage broken down by item category or provider group

---

## Step 5: Value-to-Metric Registry

Implement `registry.py` with two mappings:

```python
# 1. Metric → value mapping (each metric belongs to one primary value)
METRIC_VALUE_MAP = {
    "ndcg":                     "usefulness",
    "precision":                "usefulness",
    "recall":                   "usefulness",
    "hit":                      "usefulness",
    "map":                      "usefulness",
    "mrr":                      "usefulness",
    "intra_list_diversity":     "diversity",
    "novelty":                  "novelty",
    "calibration":              "personal_growth",
    "tail_percentage":          "novelty",
    "average_popularity":       "novelty",           # inverse — lower means more novel
    "performance_disparity":    "fairness",
    "provider_exposure_gini":   "fairness",
    "provider_hit_rate":        "recognition",
    "provider_exposure_share":  "recognition",
    "min_provider_exposure":    "fairness",
    "item_coverage":            "market_development",
    "revenue_weighted_ndcg":    "monetary_reward",
    "popularity_bias_delta":    "market_development",
}

# 2. Value → stakeholder mapping
VALUE_STAKEHOLDER_MAP = {
    "usefulness":           ["consumer"],
    "diversity":            ["consumer"],
    "novelty":              ["consumer", "system"],
    "personal_growth":      ["consumer"],
    "fairness":             ["consumer", "provider", "system"],
    "recognition":          ["provider"],
    "monetary_reward":      ["system", "provider"],
    "market_development":   ["system"],
}
```

---

## Step 6: Configuration Schema

The ms_eval config should be a YAML block that can either live in the RecBole config file (under a `ms_eval:` key) or as a standalone `ms_eval_config.yaml`. Example:

```yaml
ms_eval:
  # Which K values to evaluate at
  topk: [10, 20, 50]
  
  # Stakeholder configuration
  stakeholders:
    consumer:
      enabled: true
      filter_fields: [age, gender, occupation]   # user metadata fields for group analysis
    
    provider:
      enabled: true
      source: item_field          # item_field | provider_file
      field: developer            # if source=item_field: which item column is the provider
      # file: provider.csv        # if source=provider_file
      filter_fields: []           # provider metadata fields for group filtering
    
    system:
      enabled: true
      price_field: price          # item field for revenue-weighted metrics (optional)

  # Which metrics to compute (use "all" or list specific names)
  metrics:
    consumer: all
    provider: all
    system: all

  # Output configuration
  output:
    csv: true
    csv_path: results/ms_eval/
    dashboard: true
    dashboard_port: 8501
```

---

## Step 7: Integration Point with RecBole

The evaluator must hook into RecBole cleanly. The intended usage pattern:

```python
# After standard RecBole training and evaluation:
from recbole.quick_start import run_recbole
from ms_eval import MSEvaluator

# Run RecBole normally
result = run_recbole(model='BPR', dataset='ml-1m', config_file_list=['config.yaml'])

# Then run ms_eval on the same trainer/dataset
evaluator = MSEvaluator.from_recbole_trainer(
    trainer=result['trainer'],
    config_path='ms_eval_config.yaml'
)
report = evaluator.evaluate()
report.to_csv('results/')
report.launch_dashboard()
```

**Key question to investigate when reading the code:** Does the existing code already capture the top-K recommendation lists per user? If yes, from where? If RecBole's Trainer.evaluate() does not expose per-user top-K lists easily, you may need to:
1. Call `trainer.model.full_sort_predict()` or equivalent
2. Or hook into the RecBole evaluator's `collect()` method

**Do not modify RecBole internals.** If access is needed, subclass or wrap.

---

## Step 8: Dashboard Design

The dashboard (Streamlit recommended for simplicity) should have three navigation modes:

**Mode 1: Stakeholder view**
- Sidebar selector: Consumer / Provider / System
- If Consumer: shows aggregate metrics + optional filter by user demographic group
- If Provider: shows aggregate provider metrics + optional filter by provider metadata field + optional drill-down to individual provider
- If System: shows system-wide metrics

**Mode 2: Value view**
- Sidebar selector: dropdown of all values (Usefulness, Fairness, Novelty, etc.)
- Shows all metrics associated with that value, across relevant stakeholders
- Allows comparison of trade-offs (e.g., NDCG vs. Provider Gini on same page)

**Mode 3: Compare view** (can be deferred to later)
- Side-by-side comparison of two evaluation runs
- Useful for comparing two models or two re-ranking strategies

---

## Implementation Order (Follow This Sequence)

1. Read codebase, summarise findings
2. Propose and decide on provider configuration approach (Option A/B/C)
3. Implement `ms_eval/config.py` and `ms_eval/loader.py`
4. Implement `ms_eval/metrics/registry.py`
5. Implement `ms_eval/metrics/consumer.py` — start with NDCG/Recall/ILD/Novelty only
6. Implement `ms_eval/metrics/provider.py` — start with exposure share + Gini
7. Implement `ms_eval/metrics/system.py` — start with coverage + popularity bias
8. Implement `ms_eval/evaluator.py` and `ms_eval/report.py`
9. Verify on MovieLens-1M (consumer + system metrics only)
10. Verify on Steam (provider + system metrics)
11. Implement CSV output
12. Implement Streamlit dashboard

Do not skip ahead. Verify each step before proceeding.

---

## Risks to Flag

- **Data leakage:** Ensure user history used in calibration/novelty is the *training* history, not the test set.
- **Cold-start users/providers:** Some users or providers may have very few interactions. Flag these in the report rather than silently computing unreliable metrics.
- **RecBole version compatibility:** RecBole's internal APIs change between versions. Check `recbole.__version__` and note it. If the top-K list access method differs, document the correct approach for the installed version.
- **MovieLens genre field:** Genres in MovieLens are pipe-separated strings. The loader must parse these into sets before computing ILD or calibration.
- **Steam price field:** Price may be 0 or missing for free-to-play games. Handle gracefully in revenue-weighted NDCG.

---

## What Success Looks Like

At the end of this session, the following should work:

```bash
python run_ms_eval.py --dataset ml-1m --model BPR --config config.yaml
```

And produce:
1. A structured Python `EvaluationReport` object with system, consumer, and provider sections
2. A CSV export of all metrics
3. A runnable Streamlit dashboard that renders the three navigation modes

Provider metrics will show "not available" gracefully for MovieLens-1M (no provider field). Consumer group metrics will show "not available" gracefully for Steam (no user demographics).
