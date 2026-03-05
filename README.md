# Strategy Evolver v2 (Expanded Genome + Champion High Score)

This service reads:
- `Screener` (current snapshot)
- `history` (append-only snapshots)

…and continuously:
- evolves a large population of **random program strategies** (expression trees) using validation fitness,
- tracks a **Champion** as the **best TEST (holdout) score** to beat ("high score"),
- injects fully random strategies every generation (immigrants),
- writes:
  - `strategies` (append rows each cycle; includes best_val, best_test, champion stats)
  - `signals` (rewritten each cycle; from champion by default)
  - `champion` (rewritten only when champion improves)

## Quick start (Railway)
Env vars:
- DATA_SOURCE=google
- SHEET_ID=...
- GOOGLE_SERVICE_ACCOUNT_JSON=... (raw JSON or base64 JSON)
- SCREENER_TAB=Screener
- HISTORY_TAB=history
- STRATEGIES_TAB=strategies
- SIGNALS_TAB=signals
- CHAMPION_TAB=champion
- STATE_PATH=/data/state.json   (recommended)
- SIGNALS_SOURCE=champion       (or best_val)

Tuning:
- SLEEP_SECONDS=60
- POP_SIZE=96
- GENERATIONS_PER_CYCLE=6
- ELITE_FRAC=0.22
- RANDOM_IMMIGRANT_RATE=0.07
- MAX_EXPR_DEPTH=6
- TRAIN_FRAC=0.70
- VAL_FRAC=0.15      (TEST is the remaining 0.15)
- MIN_TRADES=25

**Important:** attach a Railway Volume mounted at `/data` so the population and champion persist.

## Local test (XLSX)
```bash
pip install -r requirements.txt
export DATA_SOURCE=excel
export EXCEL_PATH=./Oanda.xlsx
python main.py --once
```

## How "Champion" works
- The data is split by time into Train / Val / Test.
- **Evolution** selects by Validation fitness.
- **Champion** is updated ONLY by **Test fitness**, and is the all-time "high score" stored in state + written to the `champion` tab.
- Every cycle, every genome gets a shot at champion: the best TEST performer in the current population is compared against the all-time champion.


## v2.1 patch notes
- Fix: avoid `ValueError: assignment destination is read-only` by copying numpy arrays before in-place masking.
- Add: retry/backoff on Google Sheets 429 quota errors.
