"""Streamlit presentation of durable progress; no worker calls Streamlit."""
import csv
import io
import json
from datetime import UTC, datetime

from entdec.auth.query_runs import ACTIVE, statistics

LABELS = {
    "queue": "Chờ xử lý",
    "start": "Tiếp nhận", "embed": "Embedding câu hỏi", "session": "Bộ nhớ hội thoại",
    "intent": "Phân tích ý định", "resolve": "Chuẩn hoá tham chiếu",
    "retrieval": "Truy xuất ứng viên", "rerank": "Xếp hạng lại",
    "evidence": "Tạo bằng chứng", "answer": "Sinh câu trả lời", "persist": "Lưu kết quả",
    "done": "Kết thúc",
}


def render_progress(st, run, *, detailed=False, show_answer=True):
    st.caption(f"{run['trace_id']} · {run['status']}")
    stages = {}
    for event in run["events"]:
        if event["stage"] != "answer_line":
            stages[event["stage"]] = event
    with st.expander("Quá trình xử lý", expanded=run["status"] in ACTIVE or detailed):
        if not stages:
            st.write("Đang chờ worker xử lý…")
        for stage, event in stages.items():
            if stage in {"queue", "start"} and len(stages) > 2:
                event = dict(event, status="ok")
            if run["status"] in {"error", "interrupted"} and stage == run.get("active_stage"):
                event = dict(event, status=run["status"])
            duration = event.get("duration_ms")
            if duration is None and event["status"] == "running":
                duration = (datetime.now(UTC) - datetime.fromisoformat(event["at"])).total_seconds() * 1000
            suffix = f" · {duration / 1000:.1f}s" if duration is not None else ""
            st.write(f"{LABELS.get(stage, stage)}: {event['status']}{suffix}")
    if show_answer:
        st.markdown(run.get("answer") or "Đang chuẩn bị câu trả lời…")
    if run["status"] == "partial":
        st.warning("Câu trả lời chưa đầy đủ. Bạn có thể gửi lại câu hỏi để tạo lượt mới.")
    if run.get("error"):
        st.error(run["error"])
    if detailed:
        rows = [dict(seq=e.get("seq"), stage=e.get("stage"), status=e.get("status"),
                     elapsed_ms=e.get("elapsed_ms"), duration_ms=e.get("duration_ms"),
                     detail=json.dumps(e, ensure_ascii=False)) for e in run["events"]]
        st.dataframe(rows, use_container_width=True)
        st.json(run, expanded=False)


def trace_button(st, conversation_id, trace_id, *, key, in_fragment=False):
    def select_trace():
        st.session_state["open_trace"] = (conversation_id, trace_id)
        st.session_state.pop("closed_trace", None)

    clicked = st.button("Mở rộng chi tiết truy vấn", key=key, on_click=select_trace)
    if clicked and in_fragment:
        st.rerun()


def trace_panel(st, runs, user_id, conversation_id, trace_id, *, height):
    from entdec.auth.conversations import ConversationNotFound

    def close_trace():
        selected = st.session_state.pop("open_trace", None)
        if selected:
            st.session_state["closed_trace"] = selected

    with st.container():
        st.subheader("Chi tiết truy vấn")
        st.button("Đóng chi tiết", on_click=close_trace)

    with st.container(key="entdec_trace_scroll", height=height, border=True):
        def get_run():
            try:
                run = runs.get(user_id, trace_id)
            except ConversationNotFound:
                return None
            return run if run["conversation_id"] == conversation_id else None

        def render(run):
            if run is None:
                st.warning("Không tìm thấy truy vấn trong hội thoại này.")
                return
            st.text(run["question"])
            render_progress(st, run, detailed=True, show_answer=False)

        initial = get_run()
        if initial and initial["status"] in ACTIVE:

            @st.fragment(run_every="1s")
            def poll():
                current = get_run()
                if current is None or current["status"] not in ACTIVE:
                    st.rerun()
                render(current)

            poll()
        else:
            render(initial)


def live_chat(st, runs, user_id, conversation_id):
    @st.fragment(run_every="1s")
    def poll():
        recent = runs.list(user_id, conversation_id=conversation_id)
        if not recent:
            return
        latest = recent[0]
        key = "observed_" + latest["trace_id"]
        if latest["status"] not in ACTIVE:
            if st.session_state.pop(key, False):
                st.rerun()
            if latest["status"] in {"error", "interrupted"}:
                with st.chat_message("user"):
                    st.write(latest["question"])
                    trace_button(st, conversation_id, latest["trace_id"],
                                 key=f"failed_trace_{latest['trace_id']}", in_fragment=True)
                with st.chat_message("assistant"):
                    render_progress(st, runs.get(user_id, latest["trace_id"]))
            return
        st.session_state[key] = True
        with st.chat_message("user"):
            st.write(latest["question"])
            trace_button(st, conversation_id, latest["trace_id"],
                         key=f"live_trace_{latest['trace_id']}", in_fragment=True)
        with st.chat_message("assistant"):
            render_progress(st, runs.get(user_id, latest["trace_id"]))
    poll()


def trace_page(st, runs, user_id):
    st.subheader("Query trace — theo dõi và thống kê")
    filters = {}
    cols = st.columns(3)
    filters["status"] = cols[0].selectbox("Trạng thái", ["", "queued", "running", "complete", "partial", "error", "interrupted"])
    filters["domain"] = cols[1].selectbox("Miền", ["", "movie", "game"])
    filters["answer_source"] = cols[2].selectbox("Nguồn trả lời", ["", "llm", "template", "partial", "no_result"])
    cols = st.columns(3)
    filters["profile"] = cols[0].text_input("Profile (để trống = tất cả)")
    filters["since"] = str(cols[1].date_input("Từ ngày", value=None) or "")
    filters["until"] = str(cols[2].date_input("Đến ngày", value=None) or "")
    filters["conversation_id"] = st.text_input("ID hội thoại (để trống = tất cả)")
    selected = st.text_input("Trace ID để theo dõi", value=st.query_params.get("trace", ""))

    @st.fragment(run_every="1s")
    def dashboard():
        items = runs.list(user_id, include_events=True, **filters)
        stats = statistics(items)
        st.metric("Số lượt trong bộ lọc", len(items))
        st.json(stats, expanded=False)
        rows = [{k: r.get(k) for k in ("trace_id", "created_at", "question", "status", "profile", "answer_source")} | r.get("metrics", {}) for r in items]
        st.dataframe(rows, use_container_width=True)
        metrics = [{"metric": k, **v} for k, v in stats["metrics"].items()]
        if metrics:
            st.dataframe(metrics, use_container_width=True)
            st.bar_chart(metrics, x="metric", y="mean")
        full = items
        snapshot_stats = statistics(full)
        data = json.dumps(dict(filters=filters, stats=snapshot_stats, traces=full), ensure_ascii=False, indent=2)
        st.download_button("Tải JSON chi tiết", data, "query_traces.json", "application/json")
        out = io.StringIO()
        if rows:
            fields = list(dict.fromkeys(k for row in rows for k in row))
            writer = csv.DictWriter(out, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        st.download_button("Tải CSV", "\ufeff" + out.getvalue(), "query_traces.csv", "text/csv")
        report = "# Báo cáo truy vấn\n\nThời gian phía hệ thống (ms), không phải thời điểm trình duyệt vẽ nội dung.\n\n"
        report += "Bộ lọc: `" + json.dumps(filters, ensure_ascii=False) + "`\n\n"
        report += "```json\n" + json.dumps(snapshot_stats, ensure_ascii=False, indent=2) + "\n```\n"
        report += "\n| Trace | Profile | Model | Trạng thái |\n|---|---|---|---|\n"
        for r in full:
            report += f"| {r['trace_id']} | {r.get('profile', '')} | {r.get('model', '')} | {r['status']} |\n"
        st.download_button("Tải báo cáo Markdown", report, "query_trace_report.md", "text/markdown")
        if selected:
            from entdec.auth.conversations import ConversationNotFound
            try:
                render_progress(st, runs.get(user_id, selected), detailed=True)
            except ConversationNotFound:
                st.warning("Không tìm thấy truy vấn thuộc tài khoản này.")
        elif full:
            st.caption("Sao chép Trace ID từ bảng để theo dõi chi tiết; hoặc mở liên kết trong chat.")
    dashboard()
