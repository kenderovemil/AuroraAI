#!/usr/bin/env bash
set -euo pipefail

# scripts/push_code.sh
# Safely push code-only changes to the GitHub repository using SSH.
# Usage: run this in the repository root in a separate terminal while long-running uploads run.

LOG_FILE="scripts/push_code.log"

# Ensure log directory exists and create the log file early so 'tail -f' won't fail if script exits early
mkdir -p "$(dirname "$LOG_FILE")"
touch "$LOG_FILE"

echo "Running pre-push safety checks..."

# 1) Ensure no model files or secrets are tracked
TRACKED=$(git ls-files | grep -E '^(models/|secrets/)' || true)
if [ -n "$TRACKED" ]; then
  echo "ERROR: The following model/secret files are still tracked - untrack them before pushing:" >&2
  echo "$TRACKED" >&2
  # Append the tracked list to the log for easier debugging
  echo "ERROR: Tracked model/secret files found:" >> "$LOG_FILE"
  echo "$TRACKED" >> "$LOG_FILE"
  exit 2
fi

# 2) Show status
echo "Git status (short):"
git status --porcelain

# 3) Ensure remote origin exists
ORIGIN_URL=""
if git remote get-url origin >/dev/null 2>&1; then
  ORIGIN_URL=$(git remote get-url origin)
  echo "Origin remote: ${ORIGIN_URL}"
else
  echo "No origin remote configured. Please add one (SSH preferred) and re-run." >&2
  echo "Example: git remote add origin git@github.com:kenderovemil/AuroraAI.git" >&2
  exit 3
fi

# 4) Prefer SSH remote for safety (avoid embedding PATs)
if [[ "$ORIGIN_URL" != git@github.com:* ]]; then
  echo "WARNING: origin remote is not an SSH URL. For safety, configure SSH remote to avoid exposing PATs." >&2
  echo "Current origin: $ORIGIN_URL" >&2
  echo "To switch to SSH (example):" >&2
  echo "  git remote set-url origin git@github.com:kenderovemil/AuroraAI.git" >&2
  echo "Aborting to avoid accidental token exposure. Re-run after configuring SSH remote." >&2
  exit 4
fi

# 5) Create a commit (if there are changes)
if git diff --staged --quiet 2>/dev/null && git diff --quiet 2>/dev/null; then
  echo "No changes to commit. Nothing to push." > "$LOG_FILE"
  echo "No changes to commit. Exiting. Log: $LOG_FILE"
  exit 0
fi

# 6) Start the push in a background session (prefer tmux)
PUSH_CMD='git add . && git commit -m "Publish code-only" || true && git push -u origin main'

if command -v tmux >/dev/null 2>&1; then
  echo "Starting push in detached tmux session 'push-code' (logs -> $LOG_FILE)"
  tmux new-session -d -s push-code "bash -lc '$PUSH_CMD |& tee $LOG_FILE'"
  echo "Attach with: tmux attach -t push-code"
  echo "Or tail the log: tail -f $LOG_FILE"
  tmux list-panes -t push-code || true
  exit 0
else
  echo "tmux not found; starting push in background with nohup (logs -> $LOG_FILE)"
  nohup bash -lc "$PUSH_CMD" > "$LOG_FILE" 2>&1 &
  PID=$!
  echo "Push started in background (PID: $PID). Tail logs with: tail -f $LOG_FILE"
  exit 0
fi

