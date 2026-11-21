"""Minimal Streamlit UI (section 8): ``python -m entdec --profile tiny ui``.

Chat box at the bottom, one result card per recommended item with a "why"
expander holding the code-built evidence, plus the assistant's answer. The UI
calls the chatbot pipeline directly in-process (no HTTP dependency); the same
sequential embedding/generation rule applies.
"""

from __future__ import annotations

import sys
from typing import Any

TITLE = "🎥🎬 Entertainment chatbot"
DESCRIPTION = (
    "Hỏi về phim hoặc game dưới bằng tiếng Việt/Anh "
)


def _render_saved_message(st: Any, message: dict[str, Any]) -> None:
    st.write(message["content"])
    if message["role"] != "assistant" or not message.get("meta"):
        return

    meta = message["meta"]
    references = meta.get("resolved_references", [])
    if references:
        st.caption("Resolved references: " + ", ".join(references))
    items = meta.get("items", [])
    if items:
        columns = st.columns(min(3, len(items)))
        for index, (column, item) in enumerate(zip(columns, items, strict=False), start=1):
            with column:
                themes = item.get("shared_themes", [])
                genres = item.get("shared_genres", [])
                evidence = ", ".join(themes[:2]) or ", ".join(genres[:2]) or "evidence"
                score = item.get("score")
                score_suffix = f" (score {score:.2f})" if isinstance(score, (int, float)) else ""
                st.markdown(f"**{index}. {item.get('title', 'Recommendation')}**")
                st.caption(evidence + score_suffix)
                if item.get("evidence"):
                    with st.expander("Evidence"):
                        st.write(item["evidence"])
    graph = "yes" if meta.get("used_graph") else "no (parquet fallback)"
    intent = "LLM" if meta.get("intent_llm") else "rules"
    answer = "LLM" if meta.get("answer_llm") else "template"
    st.caption(f"graph: {graph} · intent: {intent} · answer: {answer}")


def run(cfg, args=None) -> int:  # pragma: no cover - interactive
    """Start a Streamlit server via subprocess (the module needs it)."""
    import subprocess
    from pathlib import Path

    from entdec.graph.store import GraphStoreError, ensure_graph

    app_path = Path(__file__)
    port = int(getattr(args, "port", None) or 8501)
    graph_enabled = not (
        bool(getattr(args, "no_graph", False)) or bool(getattr(cfg.env, "entdec_no_graph", False))
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
    import base64
    import io

    import streamlit as st

    st.set_page_config(page_title="Entertainment chatbot", page_icon="🎬")
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
    graph_enabled = not (no_graph or bool(getattr(cfg.env, "entdec_no_graph", False)))
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

    st.title(TITLE)
    st.caption(DESCRIPTION)
    st.caption(
        f"profile: {cfg.profile.name} · loaded graph: {loaded_profile or 'no'} · "
        f"graph: {'yes' if graph_enabled else 'no'}"
    )

    from sqlalchemy import text
    from sqlalchemy.exc import SQLAlchemyError

    from entdec.auth.conversations import ConversationNotFound, ConversationService
    from entdec.auth.database import init_db, make_engine, make_session_factory
    from entdec.auth.service import AuthError, AuthService

    if "auth_backend" not in st.session_state:
        try:
            if cfg.env.entdec_auth_disabled:
                engine = make_engine("sqlite://")
            else:
                if not cfg.env.entdec_db_url:
                    raise RuntimeError("ENTDEC_DB_URL is missing in .env.")
                engine = make_engine(cfg.env.entdec_db_url)
            if engine.dialect.name == "sqlite":
                init_db(engine)
            else:
                with engine.connect() as connection:
                    connection.execute(text("SELECT 1"))
            sessions = make_session_factory(engine)
            auth_service = (
                None
                if cfg.env.entdec_auth_disabled
                else AuthService(
                    sessions,
                    cfg.env.entdec_secret_key or "",
                    session_days=int(cfg.get("auth.session_lifetime_days", 7)),
                )
            )
            st.session_state.auth_backend = (
                engine,
                auth_service,
                ConversationService(sessions, memory_turns=int(cfg.get("llm.memory_turns", 4))),
            )
        except (RuntimeError, ValueError, SQLAlchemyError) as exc:
            message = (
                "Start MySQL with `python -m entdec auth db-up` and initialize it "
                "with `python -m entdec auth init-db`."
                if isinstance(exc, SQLAlchemyError)
                else str(exc)
            )
            st.error(f"Authentication database configuration error: {message}")
            return

    _engine, auth_service, conversations = st.session_state.auth_backend
    auth_disabled = bool(cfg.env.entdec_auth_disabled)
    user = {"id": "dev-anonymous", "email": "local development"}
    if not auth_disabled:
        token = st.session_state.get("auth_token")
        if token:
            try:
                user = auth_service.authenticate(token)
            except AuthError:
                st.session_state.pop("auth_token", None)
                st.session_state.pop("auth_user", None)
                token = None
        if not token:
            st.subheader("Login / Register")
            if st.session_state.get("recovery_codes"):
                codes = st.session_state.recovery_codes
                st.warning("Save these recovery codes now. They are shown only once.")
                st.code("\n".join(codes))
                st.download_button(
                    "Download recovery codes",
                    data="\n".join(codes),
                    file_name="entdec-recovery-codes.txt",
                    mime="text/plain",
                )
                if st.checkbox("I have saved them", key="recovery_codes_saved"):
                    st.session_state.pop("recovery_codes", None)
                    st.session_state.pop("registration_data", None)
                    st.session_state.pop("recovery_codes_saved", None)
                    st.rerun()
                return

            login_tab, register_tab = st.tabs(["Login", "Register"])
            with login_tab:
                preauth = st.session_state.get("preauth_token")
                if preauth:
                    use_recovery = st.checkbox("Use a recovery code")
                    with st.form("login_2fa_form"):
                        code = st.text_input(
                            "Recovery code" if use_recovery else "6-digit code",
                            max_chars=32 if use_recovery else 6,
                        )
                        verify = st.form_submit_button("Verify and sign in")
                    if verify:
                        try:
                            st.session_state.auth_token = auth_service.login_2fa(preauth, code)
                            st.session_state.pop("preauth_token", None)
                            st.rerun()
                        except AuthError as exc:
                            st.error(exc.detail)
                else:
                    with st.form("login_form"):
                        email = st.text_input("Email")
                        password = st.text_input("Password", type="password")
                        submitted = st.form_submit_button("Continue")
                    if submitted:
                        try:
                            st.session_state.preauth_token = auth_service.login(email, password)
                            st.rerun()
                        except AuthError as exc:
                            st.error(exc.detail)
            with register_tab:
                enrollment = st.session_state.get("registration_data")
                if enrollment:
                    st.image(
                        io.BytesIO(base64.b64decode(enrollment["qr_code_png_base64"])),
                        caption="Scan with Google Authenticator or another TOTP app",
                    )
                    st.caption("Manual setup secret")
                    st.code(enrollment["totp_secret"])
                    with st.form("confirm_registration_form"):
                        code = st.text_input("Enter the 6-digit code", max_chars=6)
                        confirmed = st.form_submit_button("Confirm two-factor setup")
                    if confirmed:
                        try:
                            st.session_state.recovery_codes = auth_service.confirm_registration(
                                enrollment["enrollment_token"], code
                            )
                            st.rerun()
                        except AuthError as exc:
                            st.error(exc.detail)
                else:
                    with st.form("register_form"):
                        email = st.text_input("Email address")
                        password = st.text_input("Password", type="password")
                        confirm_password = st.text_input("Confirm password", type="password")
                        submitted = st.form_submit_button("Create account")
                    if submitted:
                        if password != confirm_password:
                            st.error("Passwords do not match.")
                        else:
                            try:
                                st.session_state.registration_data = auth_service.register(
                                    email, password
                                )
                                st.rerun()
                            except AuthError as exc:
                                st.error(exc.detail)
            return

    if "pipeline" not in st.session_state:
        st.session_state.pipeline = ChatPipeline(cfg, use_graph=graph_enabled)

    with st.sidebar:
        st.caption(user["email"])
        if not auth_disabled and st.button("Logout"):
            auth_service.logout(st.session_state.auth_token)
            st.session_state.pop("auth_token", None)
            st.session_state.pop("auth_user", None)
            st.session_state.pop("selected_conversation", None)
            st.rerun()
        if st.button("New chat"):
            created = conversations.create(user["id"])
            st.session_state.selected_conversation = created["id"]
            st.rerun()

        search = st.text_input("Search conversations")
        conversation_page = conversations.list(user["id"], limit=100, search=search)
        for item in conversation_page["items"]:
            active = item["id"] == st.session_state.get("selected_conversation")
            if st.button(
                ("• " if active else "") + item["title"],
                key=f"open_{item['id']}",
                use_container_width=True,
            ):
                st.session_state.selected_conversation = item["id"]
                st.rerun()
            with st.expander("Rename or delete", expanded=False):
                new_title = st.text_input(
                    "Title",
                    value=item["title"],
                    key=f"title_{item['id']}",
                )
                if st.button("Rename", key=f"rename_{item['id']}"):
                    try:
                        conversations.rename(user["id"], item["id"], new_title)
                        st.rerun()
                    except (ConversationNotFound, ValueError) as exc:
                        st.error(str(exc))
                confirm_delete = st.checkbox("Confirm delete", key=f"confirm_delete_{item['id']}")
                if st.button(
                    "Delete conversation",
                    key=f"delete_{item['id']}",
                    disabled=not confirm_delete,
                ):
                    conversations.delete(user["id"], item["id"])
                    if active:
                        st.session_state.pop("selected_conversation", None)
                    st.rerun()

    conversation_id = st.session_state.get("selected_conversation")
    conversation = None
    if conversation_id:
        try:
            conversation = conversations.get(user["id"], conversation_id)
        except ConversationNotFound:
            st.session_state.pop("selected_conversation", None)

    if conversation:
        st.subheader(conversation["title"])
        for message in conversation["messages"]:
            with st.chat_message(message["role"]):
                _render_saved_message(st, message)
    else:
        st.subheader("Ask about movies or games")
        st.caption("Start a new chat or choose one from your conversation history.")

    question = st.chat_input("Your question / Câu hỏi của bạn")
    if question:
        if conversation_id is None:
            conversation = conversations.create(user["id"])
            conversation_id = conversation["id"]
            st.session_state.selected_conversation = conversation_id
        with st.spinner("Thinking… / Đang suy nghĩ…"):
            try:
                conversations.run_turn(
                    user["id"],
                    conversation_id,
                    question,
                    st.session_state.pipeline,
                )
                st.rerun()
            except ConversationNotFound:
                st.error("This conversation is no longer available.")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Chat pipeline error: {exc}")


if __name__ == "__main__":  # pragma: no cover
    main()
