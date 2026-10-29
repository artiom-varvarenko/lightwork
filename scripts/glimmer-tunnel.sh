#!/usr/bin/env bash
# Laptop-side SSH tunnel to Glimmer's llama-server on the Mac Studio (PLAN.md D19, briefs/glimmer.md §2).
# Forwards localhost:$GLIMMER_PORT (default 8080; Lightwork itself uses 8000) to 127.0.0.1:$GLIMMER_PORT on the Mac.
#
# Usage: glimmer-tunnel.sh up|down|status
#   GLIMMER_HOST  ssh host (default mac-studio)
#   GLIMMER_PORT  port on both ends (default 8080)
set -euo pipefail

HOST=${GLIMMER_HOST:-mac-studio}
PORT=${GLIMMER_PORT:-8080}
CTL=${XDG_RUNTIME_DIR:-/tmp}/lightwork-glimmer-tunnel-$PORT.sock

case ${1:-} in
  up)
    if ssh -S "$CTL" -O check "$HOST" 2>/dev/null; then echo "tunnel already up on localhost:$PORT"; exit 0; fi
    # -f -N -M: background master with a control socket so `down` can close exactly this tunnel.
    ssh -f -N -M -S "$CTL" -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 \
      -L "127.0.0.1:$PORT:127.0.0.1:$PORT" "$HOST"
    echo "tunnel up: http://127.0.0.1:$PORT -> $HOST:127.0.0.1:$PORT"
    curl -s --noproxy '*' -m 5 "http://127.0.0.1:$PORT/health" && echo || echo "(server not answering yet)"
    ;;
  down)
    ssh -S "$CTL" -O exit "$HOST" 2>/dev/null && echo "tunnel down" || echo "no tunnel"
    ;;
  status)
    if ssh -S "$CTL" -O check "$HOST" 2>/dev/null; then
      echo "tunnel up on localhost:$PORT; health: $(curl -s --noproxy '*' -m 5 "http://127.0.0.1:$PORT/health")"
    else
      echo "tunnel down"
    fi
    ;;
  *) echo "usage: $0 up|down|status" >&2; exit 2 ;;
esac
