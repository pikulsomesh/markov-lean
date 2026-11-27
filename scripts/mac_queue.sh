#!/usr/bin/env bash
# The Mac Mini's run queue (configs/mac_queue.yaml): one stage at a time, in order, over days.
# Stages now: wait for the Putnam search to end, sphere packing, PNT+ and Carleson extractions,
# then the 64-expansion miniF2F search. Guide: docs/mac_queue.md.
#
#   scripts/mac_queue.sh                 # start it in the background (closing the terminal is fine)
#   scripts/mac_queue.sh --status        # every stage's state, times, and the current stage's progress
#   scripts/mac_queue.sh --stop          # stop cleanly after the current stage's job saves its work
#   scripts/mac_queue.sh --retry NAME    # run a failed or done stage again on the next start
#   scripts/mac_queue.sh --skip NAME     # leave a stage out (later stages then go ahead)
#   scripts/mac_queue.sh --list          # the stages and their commands
#   scripts/mac_queue.sh --foreground    # run in this terminal instead (Ctrl-C stops cleanly)
#   scripts/mac_queue.sh --agent-plist   # print an optional launchd agent that starts it at login
#
# Starting it again is always safe: done stages are skipped, an interrupted one resumes, and
# failed or incomplete ones run again. A second copy only reports that the first is running.
# QUEUE=<config> uses another queue file.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"
PY="${PY:-.venv/bin/python}"
QUEUE="${QUEUE:-configs/mac_queue.yaml}"
DIR=$("$PY" -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1])).get("dir", "results/mac_queue"))' "$QUEUE")
MQ=("$PY" -m markov_lean.mac_queue)

[ -x "$PY" ] || { echo "no $PY: run scripts/setup_mac.sh first" >&2; exit 1; }
running() { "$PY" - "$DIR/queue.lock" <<'PYEOF'
import fcntl, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.exists():
    sys.exit(1)
f = open(p, "a+")
try:
    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
except OSError:
    sys.exit(0)
sys.exit(1)
PYEOF
}
qpid() { "$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))["queue"].get("pid") or "")' "$DIR/state.json" 2>/dev/null; }

case "${1:-}" in
  --status) exec "${MQ[@]}" status --config "$QUEUE" ;;
  --list) exec "${MQ[@]}" list --config "$QUEUE" ;;
  --retry|--skip) exec "${MQ[@]}" "${1#--}" "${2:?usage: $0 $1 STAGE}" --config "$QUEUE" ;;
  --stop)
    if running; then
      pid=$(qpid); kill -TERM "$pid"
      echo "asked the queue (pid $pid) to stop; its current stage saves its work first (a few minutes)."
      echo "Check with: scripts/mac_queue.sh --status"
    else
      echo "the queue is not running"
    fi
    exit 0 ;;
  --foreground)
    if command -v caffeinate >/dev/null 2>&1; then exec caffeinate -ims "${MQ[@]}" run --config "$QUEUE"; fi
    exec "${MQ[@]}" run --config "$QUEUE" ;;
  --agent-plist)
    label="com.markov-lean.queue"
    cat <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<!-- Optional: starts the queue once each time you log in, so it resumes after a reboot.
     It does not restart a queue that stopped on its own. The Mac must log you in after a reboot.
     Install:  scripts/mac_queue.sh --agent-plist > ~/Library/LaunchAgents/$label.plist
               launchctl bootstrap gui/\$(id -u) ~/Library/LaunchAgents/$label.plist
     Remove:   launchctl bootout gui/\$(id -u)/$label && rm ~/Library/LaunchAgents/$label.plist -->
<plist version="1.0">
<dict>
  <key>Label</key><string>$label</string>
  <key>ProgramArguments</key>
  <array><string>/bin/bash</string><string>$REPO/scripts/mac_queue.sh</string><string>--foreground</string></array>
  <key>WorkingDirectory</key><string>$REPO</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>QUEUE</key><string>$QUEUE</string>
    <key>PATH</key><string>/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin:$HOME/.elan/bin</string>
  </dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><false/>
  <key>StandardOutPath</key><string>$REPO/$DIR/queue.log</string>
  <key>StandardErrorPath</key><string>$REPO/$DIR/queue.log</string>
</dict>
</plist>
PLIST
    exit 0 ;;
  "") ;;
  -h|--help) sed -n '2,19p' "$0"; exit 0 ;;
  *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
esac

if running; then
  echo "the queue is already running (pid $(qpid)). Status: scripts/mac_queue.sh --status"
  exit 0
fi
"${MQ[@]}" list --config "$QUEUE" >/dev/null   # stops here if the queue file is wrong
mkdir -p "$DIR"
nohup "${MQ[@]}" run --config "$QUEUE" >> "$DIR/queue.log" 2>&1 < /dev/null &
pid=$!
if command -v caffeinate >/dev/null 2>&1; then
  nohup caffeinate -ims -w "$pid" >/dev/null 2>&1 &   # no sleep while the queue lives; the display may sleep
fi
sleep 3
if kill -0 "$pid" 2>/dev/null; then
  echo "queue started in the background (pid $pid); log: $DIR/queue.log"
  echo "Status any time: scripts/mac_queue.sh --status    Stop: scripts/mac_queue.sh --stop"
else
  echo "the queue exited at once; see $DIR/queue.log:"; tail -5 "$DIR/queue.log"; exit 1
fi
