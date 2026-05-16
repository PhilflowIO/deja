"""Tests for subagent JSONL discovery + the `kind` column / filter.

The patch adds a second discovery path for subagent threads stored under
`<project>/<session>/subagents/agent-*.jsonl`, tags each chunk with a `kind`
column ("main" or "subagent"), and exposes an `include_subagents` flag on
`hybrid_search` (and the MCP `search` tool) which defaults to False.
"""

import json
import os
import sqlite3
import tempfile
from unittest.mock import patch

import sqlite_vec

from deja import cli, config
from deja.db import init_db
from deja.indexer import get_embedding_model, index_file
from deja.search import hybrid_search


def _write_jsonl(path: str, user_text: str, ts: str = "2026-05-16T10:00:00Z"):
    lines = [
        {
            "type": "user",
            "message": {"content": [{"type": "text", "text": user_text}]},
            "timestamp": ts,
            "uuid": "1",
        },
        {
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": "ack " + user_text}]},
            "timestamp": ts,
            "uuid": "2",
        },
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


def test_find_jsonl_files_discovers_main_and_subagent():
    with tempfile.TemporaryDirectory() as tmp:
        projects = os.path.join(tmp, "projects")
        project_dir = os.path.join(projects, "-home-user-myproject")
        # main session at project root
        _write_jsonl(os.path.join(project_dir, "session-a.jsonl"), "main turn")
        # subagent thread nested under <session>/subagents/
        _write_jsonl(
            os.path.join(project_dir, "session-a", "subagents", "agent-x.jsonl"),
            "subagent turn",
        )

        with patch.object(config, "CLAUDE_PROJECTS_DIR", projects), \
             patch.object(cli, "CLAUDE_PROJECTS_DIR", projects):
            triples = cli._find_jsonl_files()

        kinds = sorted([k for _, _, k in triples])
        assert kinds == ["main", "subagent"]
        for path, _, kind in triples:
            assert kind in {"main", "subagent"}
            if kind == "subagent":
                assert "/subagents/" in path


def test_index_file_writes_kind_column():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test.db")
        conn = init_db(db_path)
        model = get_embedding_model()

        main_path = os.path.join(tmp, "sess-main.jsonl")
        sub_path = os.path.join(tmp, "sess-sub.jsonl")
        _write_jsonl(main_path, "freelancer math discussion")
        _write_jsonl(sub_path, "freelancer math sub-agent detail")

        index_file(conn, model, main_path, "proj", "main")
        index_file(conn, model, sub_path, "proj", "subagent")

        kinds = dict(
            conn.execute("SELECT kind, COUNT(*) FROM chunks GROUP BY kind").fetchall()
        )
        assert kinds.get("main", 0) >= 1
        assert kinds.get("subagent", 0) >= 1
        conn.close()


def test_hybrid_search_excludes_subagents_by_default():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test.db")
        conn = init_db(db_path)
        model = get_embedding_model()

        main_path = os.path.join(tmp, "sess-main.jsonl")
        sub_path = os.path.join(tmp, "sess-sub.jsonl")
        _write_jsonl(main_path, "compound engineering genesis story")
        _write_jsonl(sub_path, "compound engineering details from subagent")

        index_file(conn, model, main_path, "proj", "main")
        index_file(conn, model, sub_path, "proj", "subagent")

        # Default: only main
        results = hybrid_search(conn, model, "compound engineering", limit=10)
        kinds = {r.get("kind") for r in results}
        assert kinds == {"main"}, f"expected only main, got {kinds}"

        # include_subagents=True: both
        results = hybrid_search(
            conn, model, "compound engineering", limit=10, include_subagents=True
        )
        kinds = {r.get("kind") for r in results}
        assert kinds == {"main", "subagent"}, f"expected both, got {kinds}"

        conn.close()
