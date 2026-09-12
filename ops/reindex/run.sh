#!/usr/bin/env bash
# Rebuild a deja index on a remote GPU machine, then bring the file home.
#
# Why this is not just `deja index --reindex` on the workstation: the rebuild
# is embedding-bound, and a machine that is busy being worked on measures in
# the tens of chunks per second. Why it is a script and not shell history: the
# next rebuild should not have to rediscover the two things that make the
# result usable here rather than only there.
#
#   1. Path parity. `indexed_files.path` stores absolute paths and
#      `check_needs_reindex` compares mtime and size, so the transcripts must
#      live under the *local* home path inside the container. Otherwise the
#      first local run treats every file as unknown and `gc_orphans` deletes
#      the sessions it cannot find.
#   2. mtime parity. `rsync -a` preserves it. Without that the incremental run
#      after the swap re-indexes everything.
#
# Usage: ops/reindex/run.sh <user>@<host> [remote-staging-dir]
set -euo pipefail

REMOTE=${1:?usage: run.sh <user>@<host> [remote-dir]}
REMOTE_DIR=${2:-/home/${REMOTE%%@*}/deja-reindex}
LOCAL_HOME=$HOME
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SSH_OPTS=${DEJA_SSH_OPTS:--o IdentitiesOnly=yes}
GPU_DEVICE=${DEJA_GPU_DEVICE:-0}
# Attention is quadratic in sequence length, so the batch ceiling is set by
# the longest chunks, not the average. At the model's 512-token limit a batch
# of 256 asks for a 2.4 GB buffer per layer and dies on a 24 GB card; 64 fits
# with room for whatever else is resident.
EMBED_BATCH=${DEJA_EMBED_BATCH:-64}

ssh $SSH_OPTS "$REMOTE" "mkdir -p '$REMOTE_DIR'/home/.claude/projects '$REMOTE_DIR'/home/.codex/sessions '$REMOTE_DIR'/out '$REMOTE_DIR'/cache '$REMOTE_DIR'/src"

echo "[reindex] mirroring transcripts (mtimes preserved)"
for src in .claude/projects .codex/sessions; do
  [ -d "$LOCAL_HOME/$src" ] || continue
  rsync -a --delete -e "ssh $SSH_OPTS" "$LOCAL_HOME/$src/" "$REMOTE:$REMOTE_DIR/home/$src/"
done

echo "[reindex] shipping the source tree"
rsync -a --delete --exclude .git --exclude .venv --exclude '__pycache__' \
  -e "ssh $SSH_OPTS" "$REPO_ROOT/" "$REMOTE:$REMOTE_DIR/src/"

echo "[reindex] building image"
ssh $SSH_OPTS "$REMOTE" "docker build -q -f '$REMOTE_DIR'/src/ops/reindex/Dockerfile -t deja-reindex:local '$REMOTE_DIR'/src"

echo "[reindex] indexing on GPU $GPU_DEVICE"
ssh $SSH_OPTS "$REMOTE" "docker run --rm --runtime=nvidia --gpus 'device=$GPU_DEVICE' \
  -e HOME=$LOCAL_HOME \
  -e DEJA_INDEX_PATH=/out/index.db \
  -e DEJA_EMBED_PROVIDERS=${DEJA_EMBED_PROVIDERS:-CUDAExecutionProvider} \
  -e DEJA_EMBED_BATCH=$EMBED_BATCH \
  -v '$REMOTE_DIR'/home:$LOCAL_HOME \
  -v '$REMOTE_DIR'/out:/out \
  -v '$REMOTE_DIR'/cache:/cache \
  deja-reindex:local index"

echo "[reindex] compacting and checkpointing"
ssh $SSH_OPTS "$REMOTE" "docker run --rm -v '$REMOTE_DIR'/out:/out --entrypoint python3 deja-reindex:local -c \
  \"import sqlite3, sqlite_vec, os; c=sqlite3.connect('/out/index.db'); c.enable_load_extension(True); sqlite_vec.load(c); c.enable_load_extension(False); c.execute('VACUUM INTO \\\"/out/index.compact.db\\\"'); c.close(); print(os.path.getsize('/out/index.compact.db'))\""

echo "[reindex] built. Fetch with:"
echo "  rsync -a --info=progress2 -e 'ssh $SSH_OPTS' $REMOTE:$REMOTE_DIR/out/index.compact.db <destination>"
