#!/usr/bin/env bash
# Muse Glimmer 30B on llama.cpp (Mac Studio side). PLAN.md D13/D19, briefs/glimmer.md §2 and §9.
#
# Runs llama-server inside tmux session "glimmer", bound to 127.0.0.1 only. Reach it from the laptop through
# scripts/glimmer-tunnel.sh. Stop it after each session (D19); it also unloads itself after 30 min idle.
#
# Usage (on the Mac):  glimmer-serve.sh start|stop|status|logs [N]
# From the laptop:     scp scripts/glimmer-serve.sh mac-studio:lightwork-scripts/ &&
#                      ssh mac-studio lightwork-scripts/glimmer-serve.sh start
#
# Environment overrides:
#   GLIMMER_DIR               model directory           (default ~/models/muse-glimmer)
#   GLIMMER_PORT              listen port               (default 8080)
#   GLIMMER_CTX               context size, 1 slot      (default 16384)
#   GLIMMER_REASONING_BUDGET  server-wide think cap     (default -1 = unlimited; see sp2-glimmer.md G4)
#   GLIMMER_KEY_FILE          --api-key-file, used only if the file exists (default ~/.config/lightwork/glimmer.key)
#   GLIMMER_SLEEP_IDLE        --sleep-idle-seconds      (default 1800)
#   GLIMMER_EXTRA             extra llama-server flags, word-split (e.g. DFlash: "-md <gguf> --spec-type draft-dflash")
set -euo pipefail

SESSION=glimmer
DIR=${GLIMMER_DIR:-$HOME/models/muse-glimmer}
PORT=${GLIMMER_PORT:-8080}
CTX=${GLIMMER_CTX:-16384}
BUDGET=${GLIMMER_REASONING_BUDGET:--1}
KEY_FILE=${GLIMMER_KEY_FILE:-$HOME/.config/lightwork/glimmer.key}
SLEEP_IDLE=${GLIMMER_SLEEP_IDLE:-1800}
LOG=$DIR/server.log
MODEL=$DIR/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf
MMPROJ=$DIR/mmproj-Muse-Glimmer-30B-Q4_K_M.gguf

die() { echo "glimmer-serve: $*" >&2; exit 2; }

running() { tmux has-session -t "=$SESSION" 2>/dev/null; }

server_pid() { tmux list-panes -t "=$SESSION" -F '#{pane_pid}' 2>/dev/null | head -n1; }

health() { curl -s --noproxy '*' -m 3 "http://127.0.0.1:$PORT/health" || true; }

cmd_start() {
  command -v llama-server >/dev/null || die "llama-server not on PATH"
  command -v tmux >/dev/null || die "tmux not on PATH"
  [[ -f $MODEL && -f $MMPROJ ]] || die "model files missing in $DIR"
  running && die "tmux session '$SESSION' already exists (pid $(server_pid)); use status/stop"
  if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then die "port $PORT already in use"; fi

  local args=(
    -m "$MODEL" --mmproj "$MMPROJ"
    -a muse-glimmer -ngl 99 -c "$CTX" -np 1
    --host 127.0.0.1 --port "$PORT"
    --chat-template-kwargs '{"reasoning_strength":"low"}'
    --temp 1.0 --top-p 0.95 --top-k 64 --min-p 0
    --reasoning-budget "$BUDGET"
    --metrics --no-webui --sleep-idle-seconds "$SLEEP_IDLE"
    --log-file "$LOG"
  )
  [[ -f $KEY_FILE ]] && args+=(--api-key-file "$KEY_FILE")
  # shellcheck disable=SC2206
  [[ -n ${GLIMMER_EXTRA:-} ]] && args+=(${GLIMMER_EXTRA})

  : >"$LOG"
  local cmdline
  cmdline=$(printf '%q ' llama-server "${args[@]}")
  echo "$cmdline" >"$DIR/server.cmdline"
  # exec: the pane process *is* llama-server, so stop can signal exactly it.
  tmux new-session -d -s "$SESSION" "exec $cmdline"
  echo "started in tmux '$SESSION' (pid $(server_pid)); waiting for /health ..."
  local i
  for i in $(seq 1 180); do
    running || die "server exited during load; see $LOG"
    if [[ $(health) == *'"ok"'* ]]; then
      echo "ready after ${i}s on 127.0.0.1:$PORT"
      grep -E 'n_ctx_slot|capping|mmproj|vision|build' "$LOG" | head -n 12 || true
      return 0
    fi
    sleep 1
  done
  die "not healthy after 180 s; see $LOG"
}

cmd_stop() {
  if ! running; then echo "not running"; return 0; fi
  local pid; pid=$(server_pid)
  kill -INT "$pid" 2>/dev/null || true
  local i
  for i in $(seq 1 20); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 0.5
  done
  kill -0 "$pid" 2>/dev/null && kill -TERM "$pid" 2>/dev/null || true
  tmux kill-session -t "=$SESSION" 2>/dev/null || true
  echo "stopped (pid $pid)"
}

cmd_status() {
  if running; then
    local pid; pid=$(server_pid)
    echo "session: $SESSION running, pid $pid, rss $(ps -o rss= -p "$pid" 2>/dev/null | awk '{printf "%.1f GiB", $1/1048576}')"
    echo "health:  $(health)"
    curl -s --noproxy '*' -m 3 "http://127.0.0.1:$PORT/props" 2>/dev/null |
      python3 -c 'import json,sys; p=json.load(sys.stdin); print("props:  ", json.dumps({k:p.get(k) for k in ("build_info","model_path","modalities","total_slots","is_sleeping")}))' 2>/dev/null || true
  else
    echo "session: $SESSION not running"
    if pgrep -x llama-server >/dev/null; then echo "warning: a llama-server outside tmux '$SESSION' is running: $(pgrep -x llama-server | tr '\n' ' ')"; fi
  fi
}

cmd_logs() { tail -n "${1:-50}" "$LOG"; }

case ${1:-} in
  start) cmd_start ;;
  stop) cmd_stop ;;
  status) cmd_status ;;
  logs) shift; cmd_logs "${1:-50}" ;;
  *) echo "usage: $0 start|stop|status|logs [N]" >&2; exit 2 ;;
esac
