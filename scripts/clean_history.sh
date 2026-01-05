#!/usr/bin/env bash
set -euo pipefail

# scripts/clean_history.sh
# Safe, guided script to backup and rewrite git history to remove sensitive/large paths.
# Usage (from repository root):
#   bash scripts/clean_history.sh
# This will:
#  - create a bare mirror backup of your repo
#  - clone a working copy
#  - run git-filter-repo to remove `models/` and `secrets/` from history
#  - leave the cleaned repo locally for inspection (no automatic push)

REPO_PATH="$(pwd)"
BACKUP_MIRROR="${REPO_PATH}-backup-mirror.git"
CLEAN_WORKDIR="${REPO_PATH}-cleaned"

echo "Repository path: ${REPO_PATH}"
echo "Backup mirror will be created at: ${BACKUP_MIRROR}"
echo "Cleaned working copy will be created at: ${CLEAN_WORKDIR}"

if [ -d "${BACKUP_MIRROR}" ] || [ -d "${CLEAN_WORKDIR}" ]; then
  echo "Error: backup or cleaned directories already exist. Remove or rename them first." >&2
  exit 1
fi

echo "Creating a bare mirror backup..."
git clone --mirror "${REPO_PATH}" "${BACKUP_MIRROR}"

echo "Cloning a fresh working copy for history rewrite..."
git clone "${REPO_PATH}" "${CLEAN_WORKDIR}"
cd "${CLEAN_WORKDIR}"

# Ensure git-filter-repo is available
if ! command -v git-filter-repo >/dev/null 2>&1; then
  echo "git-filter-repo is not installed. Install with: pip install git-filter-repo" >&2
  echo "On some systems you may prefer to install from the git-filter-repo repo: https://github.com/newren/git-filter-repo" >&2
  exit 1
fi

echo "Rewriting history to remove 'models/' and 'secrets/' paths (this is irreversible for this clone)..."
# Remove these paths from all commits
git filter-repo --path models/ --path secrets/ --invert-paths

# Optional: also remove very large blobs (uncomment if desired)
# git filter-repo --strip-blobs-bigger-than 50M

echo "History rewrite complete. Inspect the cleaned repository at: ${CLEAN_WORKDIR}"

echo "If you are satisfied with the cleaned history, run the following commands to push the cleaned repo (these are examples; replace <your-remote-url> with your remote):"

echo "  cd ${CLEAN_WORKDIR}"
echo "  git remote remove origin || true"
echo "  git remote add origin <your-remote-url>"
echo "  git push --force origin --all"
echo "  git push --force origin --tags"

echo "WARNING: Force-pushing rewritten history will require all collaborators to reclone the repository."
echo "Keep the backup mirror (${BACKUP_MIRROR}) safe until you no longer need the original history."

echo "Done."
