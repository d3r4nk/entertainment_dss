"""CLI parser coverage for global and subcommand profile options."""

from entdec.cli import build_parser


def test_profile_is_preserved_before_or_after_subcommand() -> None:
    parser = build_parser()

    before = parser.parse_args(["--profile", "default", "graph", "use"])
    after = parser.parse_args(["graph", "use", "--profile", "default"])

    assert before.profile == "default"
    assert after.profile == "default"


def test_crosseval_arguments_work_before_or_after_command() -> None:
    parser = build_parser()

    before = parser.parse_args(
        ["--profile", "default", "crosseval", "--seeds", "20", "--k", "10"]
    )
    after = parser.parse_args(
        ["crosseval", "--profile", "tiny", "--gold-only", "--no-title-mask-check"]
    )

    assert (before.command, before.profile, before.seeds, before.k) == (
        "crosseval",
        "default",
        20,
        10,
    )
    assert (after.command, after.profile, after.gold_only, after.no_title_mask_check) == (
        "crosseval",
        "tiny",
        True,
        True,
    )


def test_chat_memory_commands_parse() -> None:
    parser = build_parser()
    chat = parser.parse_args(["chat", "--profile", "tiny", "--no-memory"])
    evaluation = parser.parse_args(["chateval", "--profile", "tiny"])
    assert chat.no_memory is True
    assert evaluation.command == "chateval"


def test_auth_commands_parse() -> None:
    parser = build_parser()
    db_up = parser.parse_args(["auth", "db-up"])
    init_db = parser.parse_args(["auth", "init-db"])
    reset = parser.parse_args(["auth", "reset-2fa", "--email", "user@example.com"])
    assert (db_up.command, db_up.action) == ("auth", "db-up")
    assert (init_db.command, init_db.action) == ("auth", "init-db")
    assert (reset.action, reset.email) == ("reset-2fa", "user@example.com")
