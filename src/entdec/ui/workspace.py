"""Attach browser-local resize controls to the native Streamlit chat layout."""

from functools import lru_cache
from importlib.resources import files

DEFAULT_PANE_HEIGHT = 460


@lru_cache(maxsize=1)
def _register_component():
    import streamlit as st

    assets = files("entdec.ui")
    return st.components.v2.component(
        "entdec_chat_workspace",
        html="<span hidden></span>",
        css=assets.joinpath("workspace.css").read_text(encoding="utf-8"),
        js=assets.joinpath("workspace.js").read_text(encoding="utf-8"),
        isolate_styles=False,
    )


def mount_workspace_controls(st, conversation_id, trace_id):
    if not hasattr(st.components, "v2"):
        return
    _register_component()(
        key="entdec_workspace_controls",
        data={"conversation": conversation_id, "trace": trace_id, "height": DEFAULT_PANE_HEIGHT},
    )
