#!/usr/bin/env bash
# run_eval.sh — Full dissertation evaluation run (RAGAS + 4 ablation configs)
#
# Run from project root on homeserver:
#   chmod +x run_eval.sh && ./run_eval.sh
#
# Results saved to:
#   results/ragas_scores.json
#   results/ablation_*.json
#   ~/eval_results.md   ← copy this file to laptop and paste to Claude when done
#
# Total runtime: ~3.5 hrs

set -euo pipefail

PROJ="$(cd "$(dirname "$0")" && pwd)"
VENV="$PROJ/.venv/bin/python3"
ENV_FILE="$PROJ/.env"
ORCH_SCRIPT="$PROJ/orchestrator.py"
RESULTS_DIR="$PROJ/results"
MD_OUT="$HOME/eval_results.md"
ORCH_LOG="$PROJ/orchestrator_eval.log"
ORCH_PORT=8000   # from PORT=8000 in .env

mkdir -p "$RESULTS_DIR"

# ── install eval dependencies if missing ───────────────────────────────────────
log_plain() { echo "$*"; }
log_plain "Checking eval dependencies..."
"$VENV" -c "import ragas, datasets, pandas" 2>/dev/null || {
    log_plain "Installing missing packages (ragas datasets pandas)..."
    "$PROJ/.venv/bin/pip" install -q ragas datasets pandas
}
log_plain "Dependencies ok."

# ── helpers ────────────────────────────────────────────────────────────────────

log() {
    local msg="[$(date '+%H:%M:%S')] $*"
    echo "$msg"
    echo "$msg" >> "$MD_OUT"
}

wait_for_orch() {
    log "  Waiting for orchestrator on :$ORCH_PORT..."
    for i in $(seq 1 60); do
        if curl -sf "http://localhost:$ORCH_PORT/health" >/dev/null 2>&1; then
            log "  Orchestrator ready (${i}s)"
            return 0
        fi
        sleep 2
    done
    log "  ERROR: orchestrator did not come up after 120s — aborting."
    exit 1
}

stop_orch() {
    log "  Stopping orchestrator..."
    pkill -f "python.*orchestrator.py" 2>/dev/null || true
    sleep 4
}

start_orch() {
    log "  Starting orchestrator..."
    nohup "$VENV" "$ORCH_SCRIPT" >> "$ORCH_LOG" 2>&1 &
    wait_for_orch
}

# Use | as sed delimiter to handle values containing /
set_env() {
    local key="$1" val="$2"
    if grep -q "^${key}=" "$ENV_FILE" 2>/dev/null; then
        sed -i "s|^${key}=.*|${key}=${val}|" "$ENV_FILE"
    else
        echo "${key}=${val}" >> "$ENV_FILE"
    fi
}

restore_env() {
    local key="$1" orig_val="$2"
    if [[ -n "$orig_val" ]]; then
        sed -i "s|^${key}=.*|${key}=${orig_val}|" "$ENV_FILE"
    else
        sed -i "/^${key}=/d" "$ENV_FILE"
    fi
}

capture_run() {
    # Run a command, tee stdout to terminal + MD file, return output
    "$@" 2>&1 | tee -a "$MD_OUT"
}

append_json() {
    local label="$1" file="$2"
    {
        echo ""
        echo "### JSON: $label"
        echo '```json'
        cat "$file" 2>/dev/null || echo "(file not found: $file)"
        echo '```'
        echo ""
    } >> "$MD_OUT"
}

# ── read original .env values before any changes ──────────────────────────────

ORIG_JUDGE=$(grep "^JUDGE_ENABLED=" "$ENV_FILE" | cut -d= -f2)
ORIG_ARANGO=$(grep "^ARANGODB_HOST=" "$ENV_FILE" | cut -d= -f2-)
ORIG_MEDGEMMA=$(grep "^MEDGEMMA_API_URL=" "$ENV_FILE" | cut -d= -f2-)
ORIG_MAX_AGENTS=$(grep "^MAX_CONCURRENT_AGENTS\s*=" "$ORCH_SCRIPT" | head -1 | grep -o '[0-9]*')

# ── init markdown output file ──────────────────────────────────────────────────

cat > "$MD_OUT" << HEADER
# MediCortex AI 2.0 — Evaluation Results

- **Generated:** $(date)
- **Host:** $(hostname)
- **Project:** $PROJ

---

HEADER

log "=== Evaluation suite starting ==="
log "RAGAS + 4 ablation configs. Estimated total: ~3.5 hrs"
log "Markdown output: $MD_OUT"
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — RAGAS  (~40 min, no env changes needed)
# ══════════════════════════════════════════════════════════════════════════════

{
    echo ""
    echo "---"
    echo "## PHASE 1 — RAGAS Evaluation (Table 6.1)"
    echo ""
} >> "$MD_OUT"

log "PHASE 1/5: RAGAS (~40 min)..."
capture_run "$VENV" "$PROJ/tests/evaluation/run_ragas.py" \
    --output "$RESULTS_DIR/ragas_scores.json"
append_json "ragas_scores.json" "$RESULTS_DIR/ragas_scores.json"
log "PHASE 1 complete."
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Ablation: no_judge  (~39 min)
# JUDGE_ENABLED=False  → judge node skipped entirely
# ══════════════════════════════════════════════════════════════════════════════

{
    echo ""
    echo "---"
    echo "## PHASE 2 — Ablation: no_judge"
    echo ""
} >> "$MD_OUT"

log "PHASE 2/5: ablation no_judge (~39 min)..."
stop_orch
set_env "JUDGE_ENABLED" "False"
start_orch
capture_run "$VENV" "$PROJ/tests/evaluation/run_ablation.py" --configs no_judge
stop_orch
restore_env "JUDGE_ENABLED" "$ORIG_JUDGE"
append_json "ablation_no_judge.json" "$RESULTS_DIR/ablation_no_judge.json"
log "PHASE 2 complete."
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — Ablation: no_kg  (~37 min)
# ARANGODB_HOST=""  → KB retrieval returns empty context, agents run on prompt only
# ══════════════════════════════════════════════════════════════════════════════

{
    echo ""
    echo "---"
    echo "## PHASE 3 — Ablation: no_kg"
    echo ""
} >> "$MD_OUT"

log "PHASE 3/5: ablation no_kg (~37 min)..."
start_orch   # restore first, then stop cleanly
stop_orch
set_env "ARANGODB_HOST" ""
start_orch
capture_run "$VENV" "$PROJ/tests/evaluation/run_ablation.py" --configs no_kg
stop_orch
restore_env "ARANGODB_HOST" "$ORIG_ARANGO"
append_json "ablation_no_kg.json" "$RESULTS_DIR/ablation_no_kg.json"
log "PHASE 3 complete."
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — Ablation: sequential  (~66 min)
# MAX_CONCURRENT_AGENTS patched to 1 in orchestrator.py
# ══════════════════════════════════════════════════════════════════════════════

{
    echo ""
    echo "---"
    echo "## PHASE 4 — Ablation: sequential"
    echo ""
} >> "$MD_OUT"

log "PHASE 4/5: ablation sequential (~66 min)..."
start_orch
stop_orch
sed -i "s|^MAX_CONCURRENT_AGENTS = .*|MAX_CONCURRENT_AGENTS = 1|" "$ORCH_SCRIPT"
start_orch
capture_run "$VENV" "$PROJ/tests/evaluation/run_ablation.py" --configs sequential
stop_orch
sed -i "s|^MAX_CONCURRENT_AGENTS = .*|MAX_CONCURRENT_AGENTS = $ORIG_MAX_AGENTS|" "$ORCH_SCRIPT"
append_json "ablation_sequential.json" "$RESULTS_DIR/ablation_sequential.json"
log "PHASE 4 complete."
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 5 — Ablation: no_adapt  (~37 min)
# MEDGEMMA_API_URL pointed at dead port → synthesis falls back to gemma4:31b-cloud
# (no domain adaptation — general-purpose model for synthesis)
# ══════════════════════════════════════════════════════════════════════════════

{
    echo ""
    echo "---"
    echo "## PHASE 5 — Ablation: no_adapt (MedGemma disabled → gemma4:31b-cloud fallback)"
    echo ""
} >> "$MD_OUT"

log "PHASE 5/5: ablation no_adapt (~37 min)..."
start_orch
stop_orch
set_env "MEDGEMMA_API_URL" "http://localhost:9999/predict"
start_orch
capture_run "$VENV" "$PROJ/tests/evaluation/run_ablation.py" --configs no_adapt
stop_orch
restore_env "MEDGEMMA_API_URL" "$ORIG_MEDGEMMA"
append_json "ablation_no_adapt.json" "$RESULTS_DIR/ablation_no_adapt.json"
log "PHASE 5 complete."
echo ""

# ── restore orchestrator to normal production config ──────────────────────────

log "Restoring orchestrator to production config..."
start_orch
log "Orchestrator running normally."

# ══════════════════════════════════════════════════════════════════════════════
# DONE
# ══════════════════════════════════════════════════════════════════════════════

{
    echo ""
    echo "---"
    echo "## Run Summary"
    echo ""
    echo "- Finished: $(date)"
    echo "- Results dir: $RESULTS_DIR"
    echo ""
    echo "| File | Size |"
    echo "|------|------|"
    for f in "$RESULTS_DIR"/ragas_scores.json "$RESULTS_DIR"/ablation_*.json; do
        [[ -f "$f" ]] && echo "| $(basename "$f") | $(wc -c < "$f") bytes |"
    done
} >> "$MD_OUT"

log ""
log "=== All done ==="
log "Copy ~/eval_results.md to your laptop and paste to Claude."
