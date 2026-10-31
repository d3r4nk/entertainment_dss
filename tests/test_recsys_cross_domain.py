"""Cross-domain recommendation tests (section 7 + 11)."""

from __future__ import annotations

import pandas as pd

from entdec.recsys.cross_domain import resolve_items


def test_resolve_items_direct_and_fuzzy() -> None:
    items = pd.DataFrame(
        {
            "app_id": [10, 11, 12],
            "title": ["Portal 2", "Celeste", "Hollow Knight"],
        }
    )
    # exact substring (case-insensitive)
    assert resolve_items(items, ["portal"], "app_id") == {10: "Portal 2"}
    # fuzzy fallback: typo resolves to the closest catalogue title
    assert list(resolve_items(items, ["Celleste"], "app_id")) == [11]
    # gibberish finds nothing
    assert resolve_items(items, ["zzzz-nothing-like-this"], "app_id") == {}
    # seed title that simply does not exist -> no crash, likely skipped
    assert resolve_items(items, [], "app_id") == {}
