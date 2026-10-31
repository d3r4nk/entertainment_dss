"""Minimal Streamlit UI (section 8): ``python -m entdec --profile tiny ui``.

Chat box at the bottom, one result card per recommended item with a "why"
expander holding the code-built evidence, plus the assistant's answer. The UI
calls the chatbot pipeline directly in-process (no HTTP dependency); the same
sequential embedding/generation rule applies.
"""

from __future__ import annotations

import sys

TITLE = "🎥🎬 Entertainment chatbot"
DESCRIPTION = (
    "Hỏi về phim hoặc game dưới bằng tiếng Việt/Anh "
)


def run(cfg, args=None) -> int:  # pragma: no cover - interactive
    """Start a Streamlit server via subprocess (the module needs it)."""
    import subprocess
    from pathlib import Path

    from entdec.graph.store import GraphStoreError, ensure_graph

    app_path = Path(__file__)
    port = int(getattr(args, "port", None) or 8501)
    graph_enabled = not (
        bool(getattr(args, "no_graph", False)) or cfg.env.entdec_no_graph
    )
    try:
        ensure_graph(cfg, no_graph=not graph_enabled)
    except GraphStoreError as exc:
        from entdec.logging_setup import get_logger

        get_logger(__name__).error("unable to select graph profile: %s", exc)
        return 1
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(port),
        "--server.headless",
        "true",
        "--",
        "--profile",
        cfg.profile.name,
    ]
    if not graph_enabled:
        command.append("--no-graph")
    print(f"Streamlit UI -> http://localhost:{port}")
    return subprocess.call(command)


def main() -> None:  # pragma: no cover - interactive
    """Streamlit entry point (module run by `streamlit run`)."""
    import streamlit as st

    profile = "tiny"
    if "--profile" in sys.argv:
        idx = sys.argv.index("--profile")
        if idx + 1 < len(sys.argv):
            profile = sys.argv[idx + 1]
    no_graph = "--no-graph" in sys.argv

    from entdec.chatbot.pipeline import ChatPipeline
    from entdec.config import load_config
    from entdec.graph.store import GraphStoreError, ensure_graph

    cfg = load_config(profile)
    graph_enabled = not (no_graph or cfg.env.entdec_no_graph)
    try:
        ensure_graph(cfg, no_graph=not graph_enabled)
    except GraphStoreError as exc:
        st.error(f"Could not select graph profile: {exc}")
        return
    loaded_profile = None
    if graph_enabled:
        from entdec.graph.neo4j_client import Neo4jClient
        from entdec.graph.store import read_graph_marker

        with Neo4jClient.from_config(cfg) as client:
            marker = read_graph_marker(client)
            loaded_profile = marker.get("profile") if marker else None

    st.set_page_config(page_title="Entertainment chatbot", page_icon="🎬")
    st.title(TITLE)
    st.caption(DESCRIPTION)
    st.caption(
        f"profile: {cfg.profile.name} · loaded graph: {loaded_profile or 'no'} · "
        f"graph: {'yes' if graph_enabled else 'no'}"
    )

    if "pipeline" not in st.session_state:
        st.session_state.pipeline = ChatPipeline(cfg, use_graph=graph_enabled)
    if "history" not in st.session_state:
        st.session_state.history = []

    question = st.chat_input("Your question:")
    if question:
        with st.spinner("Thinking..."):
            st.session_state.history.append(("user", question))
            try:
                result = st.session_state.pipeline.run(question)
            except Exception as exc:  # noqa: BLE001
                result = None
                st.session_state.history.append(("assistant", f"(pipeline error: {exc})"))
            if result is not None:
                st.session_state.history.append(("result", result))

    for kind, payload in st.session_state.history:
        if kind == "user":
            st.chat_message("user").write(payload)
        elif kind == "assistant":
            st.chat_message("assistant").write(payload)
        else:
            with st.chat_message("assistant"):
                st.write(payload.answer)
                if payload.items:
                    columns = st.columns(min(3, len(payload.items)))
                    for index, (column, item) in enumerate(
                        zip(columns, payload.items, strict=False)
                    ):
                        with column:
                            shared = ", ".join(item.shared_themes[:2]) or ", ".join(
                                item.shared_genres[:2]
                            )
                            st.markdown(f"**{index + 1}. {item.title}**")
                            st.caption((shared or "evidence") + f" (score {item.scores.get('total', 0):.2f})")
                            with st.expander("why"):
                                st.write(item.evidence_line(payload.language))
                meta = (
                    f"graph: {'yes' if payload.used_graph else 'no (parquet fallback)'} · "
                    f"intent: {'LLM' if payload.intent_llm else 'rules'} · "
                    f"answer: {'LLM' if payload.answer_llm else 'template'}"
                )
                st.caption(meta)


if __name__ == "__main__":  # pragma: no cover
    main()
