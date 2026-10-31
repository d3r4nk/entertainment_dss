"""CLI parser coverage for global and subcommand profile options."""

from entdec.cli import build_parser


def test_profile_is_preserved_before_or_after_subcommand() -> None:
    parser = build_parser()

    before = parser.parse_args(["--profile", "default", "graph", "use"])
    after = parser.parse_args(["graph", "use", "--profile", "default"])

    assert before.profile == "default"
    assert after.profile == "default"
