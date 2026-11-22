"""Durable, account-scoped query telemetry and a single-process UI worker."""
from __future__ import annotations

import json
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

from sqlalchemy import select

from entdec.auth.conversations import ConversationNotFound
from entdec.auth.models import Conversation, QueryEvent, QueryRun
from entdec.chatbot.progress import DELTA, EVENT, OUTCOME

ACTIVE = {"queued", "running"}
PROCESS = uuid.uuid4().hex
POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="chat")
LOCK = threading.RLock()
INITIALIZED = set()


def encode(value):
    return json.dumps(value, ensure_ascii=False, default=str)


class QueryRuns:
    def __init__(self, conversations):
        self.conversations = conversations
        self.sessions = conversations.sessions

    def recover(self):
        """One Streamlit process per database; API readers never recover jobs."""
        key = str(self.sessions.kw["bind"].url)
        with LOCK:
            if key in INITIALIZED:
                return
            with self.sessions() as db, db.begin():
                rows = db.scalars(select(QueryRun).where(
                    QueryRun.status.in_(ACTIVE), QueryRun.owner != PROCESS)).all()
                for row in rows:
                    payload = json.loads(row.payload)
                    payload.update(status="interrupted", error="Worker process stopped")
                    row.status, row.payload = "interrupted", encode(payload)
            INITIALIZED.add(key)

    def list(self, user_id, *, include_events=False, **filters):
        with self.sessions() as db:
            rows = db.scalars(select(QueryRun).join(Conversation).where(
                QueryRun.user_id == user_id, Conversation.deleted_at.is_(None))
                .order_by(QueryRun.created_at.desc())).all()
            items = [json.loads(row.payload) for row in rows]
            if include_events:
                for item in items:
                    item["events"] = [json.loads(event.payload) for event in db.scalars(
                        select(QueryEvent).where(QueryEvent.run_id == item["trace_id"])
                        .order_by(QueryEvent.seq)).all()]
        for key in ("status", "profile", "conversation_id", "answer_source"):
            if filters.get(key):
                items = [i for i in items if i.get(key) == filters[key]]
        if filters.get("domain"):
            items = [i for i in items if filters["domain"] in i.get("domains", [])]
        if filters.get("since"):
            items = [i for i in items if i["created_at"][:10] >= filters["since"]]
        if filters.get("until"):
            items = [i for i in items if i["created_at"][:10] <= filters["until"]]
        return items

    def get(self, user_id, run_id, after_seq=0):
        with self.sessions() as db:
            row = db.scalar(select(QueryRun).join(Conversation).where(
                QueryRun.id == run_id, QueryRun.user_id == user_id,
                Conversation.deleted_at.is_(None)))
            if row is None:
                raise ConversationNotFound("Query not found")
            payload = json.loads(row.payload)
            payload["events"] = [json.loads(e.payload) for e in db.scalars(
                select(QueryEvent).where(QueryEvent.run_id == run_id, QueryEvent.seq > after_seq)
                .order_by(QueryEvent.seq)).all()]
            return payload

    def submit(self, user_id, conversation_id, question, pipeline):
        self.recover()
        accepted = time.perf_counter()
        with LOCK:
            self.conversations.get(user_id, conversation_id)
            active = [r for r in self.list(user_id, conversation_id=conversation_id)
                      if r["status"] in ACTIVE]
            if active:
                return active[0]["trace_id"]
            run_id = str(uuid.uuid4())
            payload = dict(schema_version=1, trace_id=run_id, user_id=user_id,
                           conversation_id=conversation_id, question=question, status="queued",
                           profile=pipeline.cfg.profile.name, model=pipeline.client.gen_model,
                           embedding_model=pipeline.client.embed_model,
                           created_at=datetime.now(UTC).isoformat(), answer="", metrics={},
                           configuration={k: pipeline.cfg.get("llm." + k) for k in
                                          ("temperature", "num_ctx", "max_tokens_answer", "retrieval_k", "rerank_k")})
            payload["metrics"]["first_event_ms"] = (time.perf_counter() - accepted) * 1000
            with self.sessions() as db, db.begin():
                db.add(QueryRun(id=run_id, user_id=user_id, conversation_id=conversation_id,
                                owner=PROCESS, status="queued", payload=encode(payload)))
                db.flush()
                db.add(QueryEvent(run_id=run_id, seq=1, payload=encode(dict(
                    seq=1, stage="queue", status="queued", at=payload["created_at"],
                    elapsed_ms=payload["metrics"]["first_event_ms"]))))
            POOL.submit(self._execute, payload, pipeline, accepted)
            return run_id

    def _execute(self, payload, pipeline, accepted):
        seq = 1
        started = time.perf_counter()
        payload["metrics"]["queue_ms"] = (started - accepted) * 1000
        payload["status"] = "running"

        def event(data):
            nonlocal seq
            seq += 1
            elapsed = (time.perf_counter() - accepted) * 1000
            item = dict(data, seq=seq, elapsed_ms=elapsed, at=datetime.now(UTC).isoformat())
            payload["metrics"].setdefault("first_event_ms", elapsed)
            if data.get("stage") == "answer_line":
                payload["answer"] += data["text"]
                payload["metrics"].setdefault("first_line_ms", elapsed)
            if data.get("stage") == "persist" and data.get("status") == "running":
                payload["metrics"]["pipeline_ms"] = (time.perf_counter() - started) * 1000
            if data.get("status") == "running":
                payload["active_stage"] = data["stage"]
            if data.get("duration_ms") is not None:
                payload["metrics"][data["stage"] + "_ms"] = data["duration_ms"]
            with self.sessions() as db, db.begin():
                row = db.get(QueryRun, payload["trace_id"])
                row.status, row.payload = payload["status"], encode(payload)
                db.add(QueryEvent(run_id=row.id, seq=seq, payload=encode(item)))

        tokens = [EVENT.set(event), DELTA.set(lambda text: event(
            dict(stage="answer_line", status="ok", text=text))), OUTCOME.set(None)]
        try:
            event(dict(stage="start", status="running"))
            turn = self.conversations.run_turn(payload["user_id"], payload["conversation_id"],
                                              payload["question"], pipeline,
                                              trace_id=payload["trace_id"])
            result = turn["result"]
            payload.update(answer=result["answer"], result=result,
                           domains=result.get("domains_queried", result.get("domains", [])),
                           answer_source=OUTCOME.get() or ("llm" if result.get("answer_llm") else "template" if result.get("items") else "no_result"),
                           status="partial" if OUTCOME.get() == "partial" else "complete")
        except Exception as exc:  # retain telemetry even if message transaction rolls back
            payload.update(status="error", error=str(exc))
        finally:
            payload["metrics"]["total_ms"] = (time.perf_counter() - accepted) * 1000
            payload["finished_at"] = datetime.now(UTC).isoformat()
            try:
                event(dict(stage="done", status=payload["status"], error=payload.get("error")))
            finally:
                EVENT.reset(tokens[0])
                DELTA.reset(tokens[1])
                OUTCOME.reset(tokens[2])


def statistics(items):
    """Missing measurements stay null; quantiles use linear interpolation."""
    keys = sorted({key for item in items for key in item.get("metrics", {})})
    result = {"count": len(items), "statuses": {}, "metrics": {}}
    for item in items:
        status = item["status"]
        result["statuses"][status] = result["statuses"].get(status, 0) + 1
    for key in keys:
        values = sorted(float(i["metrics"][key]) for i in items
                        if i.get("metrics", {}).get(key) is not None)
        if not values:
            result["metrics"][key] = dict(n=0, mean=None, p50=None, p95=None)
            continue
        def percentile(p):
            position = (len(values) - 1) * p
            lo = int(position)
            hi = min(lo + 1, len(values) - 1)
            return values[lo] + (values[hi] - values[lo]) * (position - lo)
        result["metrics"][key] = dict(n=len(values), mean=sum(values) / len(values),
                                      p50=percentile(.5), p95=percentile(.95))
    terminal = [i for i in items if i["status"] not in ACTIVE]
    for name, predicate in {
        "success_rate": lambda i: i["status"] == "complete",
        "error_rate": lambda i: i["status"] in {"error", "interrupted"},
        "partial_rate": lambda i: i["status"] == "partial",
        "fallback_rate": lambda i: i.get("answer_source") == "template",
    }.items():
        result[name] = sum(predicate(i) for i in terminal) / len(terminal) if terminal else None
    return result
