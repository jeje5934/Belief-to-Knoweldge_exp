#!/usr/bin/env bash
# ============================================================
# safe_sweep.sh — vanilla fixed-sigma branch sweep runner
# ============================================================
# 목적: 서버 crash / hang 방지, 재실행 안전성 보장.
#       성능보다 안정성 우선.
#
# 권장 실행:
#   tmux new -s vanilla_sweep
#   cd /home/LJH/onlyextrinsic_vanilla
#   bash safe_sweep.sh
#
# 환경변수 override:
#   PY=python3  BATCH=64  ROUNDS=5
#   RUN_TIMEOUT=7200      # run 1개 최대 실행 시간(초)
#   GPU_TEMP_LIMIT=80     # 이 온도(°C) 이상이면 추가 대기
#   CSV=results/vanilla_sweep.csv
#   SLEEP_MIN=10  SLEEP_MAX=20   # between-run 슬립 범위(초)
#
# --dry-run 플래그: 실제 실행 없이 큐 확인
# ============================================================

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# ── CUDA / 스레드 환경변수 ────────────────────────────────
export CUDA_VISIBLE_DEVICES=""        # vanilla branch: CPU 안정 실행
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export TF_NUM_INTEROP_THREADS=1
export TF_NUM_INTRAOP_THREADS=4
export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export TF_FORCE_GPU_ALLOW_GROWTH=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

# ── 파라미터 ─────────────────────────────────────────────
PY="${PY:-python3}"
BATCH="${BATCH:-64}"
ROUNDS="${ROUNDS:-5}"
RUN_TIMEOUT="${RUN_TIMEOUT:-7200}"
GPU_TEMP_LIMIT="${GPU_TEMP_LIMIT:-80}"
CSV="${CSV:-results/vanilla_sweep.csv}"
LOG_DIR="${LOG_DIR:-logs}"
SLEEP_MIN="${SLEEP_MIN:-10}"
SLEEP_MAX="${SLEEP_MAX:-20}"
DRY_RUN=0

mkdir -p results "$LOG_DIR"

# ── 파일 디스크립터 상한 ────────────────────────────────
ulimit -n 4096 2>/dev/null || true

# ── 플래그 파싱 ──────────────────────────────────────────
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        *) echo "[WARN] unknown arg: $arg" ;;
    esac
done

# ── 로그 파일 ────────────────────────────────────────────
LOG="$LOG_DIR/safe_sweep_$(date -u '+%Y%m%dT%H%M%SZ').log"
log() { echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') $*" | tee -a "$LOG"; }

# ── 자식 PID 추적 + 종료 trap ────────────────────────────
_CHILD_PID=""
cleanup_on_exit() {
    local sig="${1:-EXIT}"
    log "=== TRAP $sig: cleaning up ==="
    if [[ -n "$_CHILD_PID" ]] && kill -0 "$_CHILD_PID" 2>/dev/null; then
        log "Killing child PID $_CHILD_PID ..."
        kill -TERM "$_CHILD_PID" 2>/dev/null || true
        sleep 3
        kill -KILL "$_CHILD_PID" 2>/dev/null || true
    fi
    "$PY" cuda_cleanup.py 2>&1 | tee -a "$LOG" || true
    log "=== safe_sweep exiting due to $sig ==="
}
trap 'cleanup_on_exit EXIT'  EXIT
trap 'cleanup_on_exit INT;  exit 130' INT
trap 'cleanup_on_exit TERM; exit 143' TERM

# ── GPU 스냅샷 ───────────────────────────────────────────
smi_log() {
    local label="$1"
    if command -v nvidia-smi &>/dev/null; then
        # Wrap in subshell so pipefail cannot propagate out of smi_log.
        ( nvidia-smi \
              --query-gpu=index,memory.used,memory.free,temperature.gpu,utilization.gpu \
              --format=csv,noheader,nounits 2>/dev/null \
          | while IFS= read -r line; do log "[smi/$label] $line"; done
        ) || true
    fi
}

# ── GPU 냉각 대기 ────────────────────────────────────────
wait_for_cool() {
    if ! command -v nvidia-smi &>/dev/null; then return; fi
    while true; do
        local temp
        temp=$(nvidia-smi --query-gpu=temperature.gpu \
                          --format=csv,noheader,nounits 2>/dev/null \
               | head -1 | tr -d ' ' || echo "0")
        if [[ "$temp" -lt "$GPU_TEMP_LIMIT" ]] 2>/dev/null; then break; fi
        log "[thermal] GPU ${temp}°C >= ${GPU_TEMP_LIMIT}°C. 30s 대기..."
        sleep 30
    done
}

# ── 실행 간 슬립 + 메모리 해제 ──────────────────────────
between_runs() {
    local s=$((SLEEP_MIN + RANDOM % (SLEEP_MAX - SLEEP_MIN + 1)))
    log "--- sleeping ${s}s (context cooldown) ---"
    sleep "$s"
    log "--- cuda_cleanup.py ---"
    "$PY" cuda_cleanup.py 2>&1 | tee -a "$LOG" || true
    smi_log "after_cleanup" || true
    wait_for_cool || true
}

# ── tag 이미 완료됐는지 확인 ─────────────────────────────
tag_done() {
    local tag="$1"
    [[ ! -f "$CSV" ]] && return 1
    "$PY" - "$tag" << 'PYEOF'
import csv, sys
tag = sys.argv[1]
with open('results/vanilla_sweep.csv') as f:
    rows = [r for r in csv.DictReader(f) if r.get('tag') == tag]
if rows:
    print(f"[SKIP] tag={tag!r} — already {len(rows)} row(s) in CSV.")
    sys.exit(0)
sys.exit(1)
PYEOF
}

# ── 단일 실험 실행 ───────────────────────────────────────
# ── plot_comparison.py 전용 실행 (PNG 존재 시 SKIP) ─────────
run_plot() {
    local out_png="$1"; shift

    echo ""
    log "========================================================"
    log "  run_plot -> $out_png"
    log "========================================================"

    if [[ -f "$out_png" ]]; then
        log "[SKIP] $out_png already exists."
        return 0
    fi

    if [[ $DRY_RUN -eq 1 ]]; then
        log "[DRY-RUN] $PY plot_comparison.py --out $out_png $*"
        return 0
    fi

    local tag
    tag="plot_$(basename "$out_png" .png)"
    local logfile="$LOG_DIR/${tag}.log"
    log "[START] $(date -u '+%Y-%m-%dT%H:%M:%SZ')  log -> $logfile"
    smi_log "pre_${tag}" || true

    set +e
    timeout "$RUN_TIMEOUT" "$PY" plot_comparison.py --out "$out_png" "$@" \
        2>&1 | tee "$logfile" | tee -a "$LOG"
    local ec="${PIPESTATUS[0]}"
    set -e

    if [[ "$ec" -eq 124 ]]; then
        log "!!! TIMEOUT (${RUN_TIMEOUT}s) for $out_png"
        "$PY" cuda_cleanup.py 2>&1 | tee -a "$LOG" || true
        exit 1
    elif [[ "$ec" -ne 0 ]]; then
        log "!!! FAILED (exit $ec) for $out_png -- see $logfile"
        exit "$ec"
    fi

    log "[DONE]  $out_png"
    between_runs
}

run_exp() {
    local tag="$1"; shift   # 나머지 인자 → vanilla_sweep.py 에 전달

    echo ""
    log "════════════════════════════════════════════════════════════"
    log "  tag = $tag"
    log "════════════════════════════════════════════════════════════"

    if tag_done "$tag"; then
        return 0
    fi

    if [[ $DRY_RUN -eq 1 ]]; then
        log "[DRY-RUN] $PY vanilla_sweep.py $*"
        return 0
    fi

    local logfile="$LOG_DIR/${tag}.log"
    log "[START] $(date -u '+%Y-%m-%dT%H:%M:%SZ')  log -> $logfile"
    smi_log "pre_${tag}"

    set +e
    timeout "$RUN_TIMEOUT" "$PY" vanilla_sweep.py "$@" 2>&1 | tee "$logfile" | tee -a "$LOG"
    local ec="${PIPESTATUS[0]}"
    set -e

    if [[ "$ec" -eq 124 ]]; then
        log "!!! TIMEOUT (${RUN_TIMEOUT}s) for tag=$tag — 강제 종료"
        "$PY" cuda_cleanup.py 2>&1 | tee -a "$LOG" || true
        exit 1
    elif [[ "$ec" -ne 0 ]]; then
        log "!!! FAILED (exit $ec) for tag=$tag — see $logfile"
        exit "$ec"
    fi

    log "[DONE]  tag=$tag"
    between_runs
}

# ════════════════════════════════════════════════════════════
# 시작 배너
# ════════════════════════════════════════════════════════════
log "======== safe_sweep start ========"
log "PY=$PY  BATCH=$BATCH  ROUNDS=$ROUNDS"
log "RUN_TIMEOUT=${RUN_TIMEOUT}s  GPU_TEMP_LIMIT=${GPU_TEMP_LIMIT}°C"
log "CSV=$CSV  LOG=$LOG"
smi_log "start"

# ════════════════════════════════════════════════════════════
# 실험 큐 — 이 섹션을 편집하여 실험 추가/제거
# run_exp <tag> [vanilla_sweep.py 인자...]
# ════════════════════════════════════════════════════════════

# ── Warm-up ablation (budget-30, beta=0, Eb/N0=0.5~0.7) ──
# 5-call schedules

run_exp warmup_sched_10_4_4_4_4_4 \
    --ebno 0.5 0.6 0.7 --betas 0.00 \
    --alpha-schedules \
        "0.08,0.08,0.08,0.08,0.08" \
        "0.10,0.10,0.10,0.10,0.10" \
        "0.10,0.08,0.06,0.04,0.02" \
    --sigma 0.3 \
    --bp-schedule 10 4 4 4 4 4 \
    --batch "$BATCH" --rounds "$ROUNDS" \
    --tag warmup_sched_10_4_4_4_4_4

# 4-call schedules

run_exp warmup_sched_10_5_5_5_5 \
    --ebno 0.5 0.6 0.7 --betas 0.00 \
    --alpha-schedules \
        "0.08,0.08,0.08,0.08" \
        "0.10,0.10,0.10,0.10" \
        "0.10,0.08,0.06,0.04" \
    --sigma 0.3 \
    --bp-schedule 10 5 5 5 5 \
    --batch "$BATCH" --rounds "$ROUNDS" \
    --tag warmup_sched_10_5_5_5_5

run_exp warmup_sched_6x5 \
    --ebno 0.5 0.6 0.7 --betas 0.00 \
    --alpha-schedules \
        "0.08,0.08,0.08,0.08" \
        "0.10,0.10,0.10,0.10" \
        "0.10,0.08,0.06,0.04" \
    --sigma 0.3 \
    --bp-schedule 6 6 6 6 6 \
    --batch "$BATCH" --rounds "$ROUNDS" \
    --tag warmup_sched_6x5

# ── Paper main figure ────────────────────────────────────────────────────────
# Baseline BP-30 vs Proposed [5x6], sigma=0.3, alpha=[0.10x5], beta=0.1

run_plot results/comparison_5x6_beta01.png \
    --bp-schedule 5 5 5 5 5 5 \
    --alpha-schedule "0.10,0.10,0.10,0.10,0.10" \
    --beta 0.1 --sigma 0.3 \
    --ebno 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 \
    --batch "$BATCH" --rounds "$ROUNDS"

# ════════════════════════════════════════════════════════════
log "======== safe_sweep finished OK ========"
smi_log "finish"
