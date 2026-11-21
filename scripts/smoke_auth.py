"""Run a real MySQL/API authentication and conversation smoke test.

Run from the repository root with ``python scripts/smoke_auth.py``. The script
starts and restarts a tiny-profile API, then removes all throwaway accounts.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

import pyotp
from sqlalchemy import select, text

ROOT = Path(__file__).resolve().parents[1]
API_HOST = "127.0.0.1"


class SmokeFailure(RuntimeError):
    """A named real-service smoke assertion failed."""


def require(name: str, condition: bool) -> None:
    if not condition:
        raise SmokeFailure(name)
    print(f"PASS {name}")


class Api:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def request(
        self,
        method: str,
        path: str,
        *,
        body: dict[str, Any] | None = None,
        token: str | None = None,
    ) -> tuple[int, Any]:
        headers = {"Accept": "application/json"}
        data = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if token is not None:
            headers["Authorization"] = f"Bearer {token}"
        request = urllib.request.Request(
            self.base_url + path, data=data, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                status, raw = response.status, response.read()
        except urllib.error.HTTPError as error:
            status, raw = error.code, error.read()
        except urllib.error.URLError as error:
            raise SmokeFailure(f"HTTP transport failed for {method} {path}") from error
        if not raw:
            return status, None
        try:
            return status, json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return status, raw.decode("utf-8", errors="replace")


class ApiProcess:
    def __init__(self) -> None:
        self.port = _free_port()
        self.base_url = f"http://{API_HOST}:{self.port}"
        self.log_path = Path(tempfile.gettempdir()) / f"entdec-auth-api-{uuid.uuid4().hex}.log"
        self.process: subprocess.Popen[bytes] | None = None

    def start(self) -> None:
        log = self.log_path.open("ab")
        try:
            self.process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "entdec",
                    "api",
                    "--host",
                    API_HOST,
                    "--port",
                    str(self.port),
                    "--profile",
                    "tiny",
                    "--no-graph",
                ],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=os.environ.copy(),
            )
        finally:
            log.close()
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            if self.process is None or self.process.poll() is not None:
                raise SmokeFailure("API process exited before becoming ready")
            try:
                with urllib.request.urlopen(self.base_url + "/docs", timeout=3) as response:
                    if response.status == 200:
                        return
            except (OSError, urllib.error.URLError):
                time.sleep(1)
        raise SmokeFailure("API readiness timeout")

    def stop(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        if os.name == "nt":
            descendants = _windows_process_tree(self.process.pid)
            ids = ",".join(str(process_id) for process_id in reversed(descendants))
            subprocess.run(
                [
                    "powershell.exe",
                    "-NoProfile",
                    "-Command",
                    f"Stop-Process -Id {ids} -ErrorAction SilentlyContinue",
                ],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=15,
            )
        else:
            self.process.terminate()
        try:
            self.process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            if os.name == "nt":
                print("WARN API process did not stop promptly; checking its process IDs")
            else:
                self.process.kill()
                self.process.wait(timeout=10)

    def restart(self) -> None:
        self.stop()
        self.start()


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind((API_HOST, 0))
        return int(sock.getsockname()[1])


def _windows_process_tree(root_pid: int) -> list[int]:
    script = (
        f"$root={root_pid}; $all=Get-CimInstance Win32_Process; $ids=@($root); "
        "do { $new=@($all | Where-Object { $ids -contains $_.ParentProcessId -and "
        "$ids -notcontains $_.ProcessId } | ForEach-Object { $_.ProcessId }); "
        "$ids += $new } while ($new.Count -gt 0); "
        "$ids | ConvertTo-Json -Compress"
    )
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    parsed = json.loads(result.stdout)
    if isinstance(parsed, int):
        return [parsed]
    return [int(process_id) for process_id in parsed]


def _sql_connection() -> Any:
    from entdec.auth.database import make_engine
    from entdec.config import EnvSettings

    url = EnvSettings().entdec_db_url
    if not url:
        raise SmokeFailure("ENTDEC_DB_URL is missing")
    engine = make_engine(url)
    return engine


def _scalar(engine: Any, query: str, params: dict[str, Any]) -> Any:
    with engine.connect() as connection:
        return connection.execute(text(query), params).scalar_one_or_none()


def _fresh_totp(secret: str, after_step: int) -> tuple[str, int]:
    step = int(time.time()) // 30
    if step <= after_step:
        time.sleep((after_step + 1) * 30 - time.time() + 0.1)
        step = int(time.time()) // 30
    return pyotp.TOTP(secret).at(step * 30), step


def _invalid_totp(secret: str) -> str:
    totp = pyotp.TOTP(secret)
    current = int(time.time()) // 30
    valid = {totp.at(step * 30) for step in range(current - 1, current + 2)}
    return next(f"{code:06d}" for code in range(1_000_000) if f"{code:06d}" not in valid)


def _login(api: Api, email: str, password: str) -> tuple[int, Any]:
    return api.request("POST", "/auth/login", body={"email": email, "password": password})


def _delete_user(email: str) -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "entdec",
            "auth",
            "delete-user",
            "--email",
            email,
        ],
        cwd=ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=30,
    )


def _wait_mysql_healthy() -> bool:
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Health.Status}}", "entdec-mysql"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip() == "healthy":
            return True
        time.sleep(2)
    return False


def _check_database_secrets(
    engine: Any,
    *,
    email: str,
    password: str,
    totp_secret: str,
    recovery_codes: list[str],
    preauth_token: str,
    session_token: str,
) -> None:
    from sqlalchemy.orm import Session

    from entdec.auth.models import (
        AuthSession,
        EnrollmentToken,
        PreAuthToken,
        RecoveryCode,
        User,
    )

    with Session(engine) as db:
        user = db.scalar(select(User).where(User.email == email))
        require("password hash is Argon2id", user is not None and user.password_hash.startswith("$argon2id$"))
        require(
            "TOTP seed encrypted at rest",
            user is not None
            and bool(user.totp_secret_encrypted)
            and totp_secret not in user.totp_secret_encrypted
            and user.totp_secret_encrypted != totp_secret,
        )
        recovery_hashes = db.scalars(
            select(RecoveryCode.code_hash).where(RecoveryCode.user_id == user.id)
        ).all()
        expected_recovery_hash = hashlib.sha256(recovery_codes[-1].strip().upper().encode()).hexdigest()
        require(
            "recovery codes stored only as hashes",
            len(recovery_hashes) == 10
            and all(len(code_hash) == 64 for code_hash in recovery_hashes)
            and expected_recovery_hash in recovery_hashes
            and all(code not in recovery_hashes for code in recovery_codes),
        )
        session_hash = hashlib.sha256(session_token.encode()).hexdigest()
        stored_session_hash = db.scalar(
            select(AuthSession.token_hash).where(
                AuthSession.user_id == user.id,
                AuthSession.token_hash == session_hash,
            )
        )
        require(
            "bearer token stored only as SHA-256",
            stored_session_hash == session_hash and session_token not in stored_session_hash,
        )
        preauth_hash = hashlib.sha256(preauth_token.encode()).hexdigest()
        stored_preauth_hash = db.scalar(
            select(PreAuthToken.token_hash).where(
                PreAuthToken.user_id == user.id,
                PreAuthToken.token_hash == preauth_hash,
            )
        )
        require(
            "pre-auth token stored only as SHA-256",
            stored_preauth_hash == preauth_hash and preauth_token not in stored_preauth_hash,
        )
        require(
            "enrollment token consumed after confirmation",
            db.scalar(
                select(EnrollmentToken.id).where(EnrollmentToken.user_id == user.id)
            )
            is None,
        )
        require(
            "password, TOTP secret, recovery codes, and tokens are not stored in plaintext",
            password not in user.password_hash
            and totp_secret not in user.totp_secret_encrypted
            and all(code not in recovery_hashes for code in recovery_codes)
            and preauth_token not in stored_preauth_hash
            and session_token not in stored_session_hash,
        )


def _check_logs(paths: list[Path], secrets_to_check: list[str]) -> None:
    checked = 0
    for path in paths:
        if path.is_file():
            contents = path.read_bytes()
            checked += 1
            if any(value.encode("utf-8") in contents for value in secrets_to_check if value):
                raise SmokeFailure("plaintext credential or token found in a log")
    print(f"PASS no plaintext credentials in {checked} checked log files")


def _stop_mysql() -> None:
    result = subprocess.run(
        ["docker", "compose", "stop", "entdec-mysql"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=90,
    )
    if result.returncode:
        raise SmokeFailure("could not stop entdec-mysql for outage verification")


def _start_mysql() -> None:
    subprocess.run(
        ["docker", "compose", "start", "entdec-mysql"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=90,
    )
    if not _wait_mysql_healthy():
        raise SmokeFailure("entdec-mysql did not recover to healthy")


def run_smoke() -> None:
    password_a = f"Smoke-{uuid.uuid4().hex}-A9!"
    password_b = f"Smoke-{uuid.uuid4().hex}-B8!"
    email_a = f"smoke-{uuid.uuid4().hex}@example.test"
    email_b = f"smoke-{uuid.uuid4().hex}@example.test"
    secrets_to_check = [password_a, password_b]
    server = ApiProcess()
    engine = None
    mysql_stopped = False
    try:
        engine = _sql_connection()
        api = Api(server.base_url)
        server.start()

        status, _ = api.request("GET", "/conversations")
        require("token-less conversation request rejected", status == 401)
        status, _ = api.request("GET", "/auth/me", token="not-a-valid-session")
        secrets_to_check.append("not-a-valid-session")
        require("invalid bearer token rejected", status == 401)

        status, registration = api.request(
            "POST",
            "/auth/register",
            body={"email": email_a, "password": password_a, "confirm_password": password_a},
        )
        require(
            "registration returns TOTP URI, QR, and manual secret",
            status == 201
            and registration["totp_uri"].startswith("otpauth://totp/")
            and registration["qr_code_png_base64"]
            and registration["totp_secret"],
        )
        secret_a = registration["totp_secret"]
        secrets_to_check.extend(
            [secret_a, registration["enrollment_token"], password_a]
        )
        enrollment_hash = hashlib.sha256(registration["enrollment_token"].encode()).hexdigest()
        require(
            "enrollment token stored only as SHA-256",
            _scalar(
                engine,
                "SELECT token_hash FROM enrollment_tokens e JOIN users u ON u.id=e.user_id WHERE u.email=:email",
                {"email": email_a},
            )
            == enrollment_hash,
        )
        status, premature = _login(api, email_a, password_a)
        require(
            "unconfirmed account cannot log in",
            status == 401 and premature.get("detail") == "Invalid email or password.",
        )
        confirmation_code_a = pyotp.TOTP(secret_a).now()
        secrets_to_check.append(confirmation_code_a)
        status, confirmation = api.request(
            "POST",
            "/auth/register/confirm-2fa",
            body={
                "enrollment_token": registration["enrollment_token"],
                "code": confirmation_code_a,
            },
        )
        recovery_a = confirmation.get("recovery_codes", []) if status == 200 else []
        require("2FA enrollment returns ten one-time recovery codes", status == 200 and len(recovery_a) == 10)
        secrets_to_check.extend(recovery_a)
        status, repeated_confirmation = api.request(
            "POST",
            "/auth/register/confirm-2fa",
            body={
                "enrollment_token": registration["enrollment_token"],
                "code": confirmation_code_a,
            },
        )
        require(
            "recovery codes are not returned a second time",
            status == 401 and "recovery_codes" not in repeated_confirmation,
        )

        status, step1 = _login(api, email_a, password_a)
        require(
            "login step 1 returns only a short-lived pre-auth token",
            status == 200 and set(step1) == {"preauth_token", "requires_2fa"} and step1["requires_2fa"],
        )
        preauth_a = step1["preauth_token"]
        secrets_to_check.append(preauth_a)
        preauth_hash = hashlib.sha256(preauth_a.encode()).hexdigest()
        require(
            "pre-auth token is hashed in MySQL",
            _scalar(
                engine,
                "SELECT token_hash FROM preauth_tokens p JOIN users u ON u.id=p.user_id WHERE u.email=:email",
                {"email": email_a},
            )
            == preauth_hash,
        )
        last_step = _scalar(
            engine,
            "SELECT totp_last_used_step FROM users WHERE email=:email",
            {"email": email_a},
        )
        code_a, _ = _fresh_totp(secret_a, int(last_step))
        secrets_to_check.append(code_a)
        status, session_a = api.request(
            "POST",
            "/auth/login/2fa",
            body={"preauth_token": preauth_a, "code": code_a},
        )
        require("login step 2 issues a bearer session", status == 200 and bool(session_a.get("access_token")))
        token_a = session_a["access_token"]
        secrets_to_check.append(token_a)
        status, me = api.request("GET", "/auth/me", token=token_a)
        require("authenticated /auth/me returns user", status == 200 and me.get("email") == email_a)

        bad_password_value = "incorrect-" + uuid.uuid4().hex
        unknown_password_value = "some-invalid-password"
        secrets_to_check.extend([bad_password_value, unknown_password_value])
        status, bad_password = _login(api, email_a, bad_password_value)
        _, unknown_email = _login(
            api,
            f"unknown-{uuid.uuid4().hex}@example.test",
            unknown_password_value,
        )
        require(
            "wrong-password and unknown-email errors are generic",
            status == 401
            and bad_password.get("detail") == "Invalid email or password."
            and unknown_email.get("detail") == bad_password.get("detail"),
        )
        status, bad_code_step1 = _login(api, email_a, password_a)
        secrets_to_check.append(bad_code_step1["preauth_token"])
        invalid_code = _invalid_totp(secret_a)
        secrets_to_check.append(invalid_code)
        status, bad_code = api.request(
            "POST",
            "/auth/login/2fa",
            body={
                "preauth_token": bad_code_step1["preauth_token"],
                "code": invalid_code,
            },
        )
        require(
            "wrong TOTP code rejected generically",
            status == 401 and bad_code.get("detail") == "Invalid authentication code.",
        )
        status, replay_step1 = _login(api, email_a, password_a)
        secrets_to_check.append(replay_step1["preauth_token"])
        status, replay = api.request(
            "POST",
            "/auth/login/2fa",
            body={"preauth_token": replay_step1["preauth_token"], "code": code_a},
        )
        require(
            "same-step TOTP replay rejected generically",
            status == 401 and replay.get("detail") == "Invalid authentication code.",
        )

        with engine.begin() as connection:
            connection.execute(
                text("UPDATE users SET failed_login_count=0, locked_until=NULL WHERE email=:email"),
                {"email": email_a},
            )
        lock_passwords = [f"wrong-{index}-{uuid.uuid4().hex}" for index in range(5)]
        secrets_to_check.extend(lock_passwords)
        lock_statuses = [_login(api, email_a, value)[0] for value in lock_passwords]
        lock_row = _scalar(
            engine,
            "SELECT locked_until FROM users WHERE email=:email",
            {"email": email_a},
        )
        require(
            "five failures trigger a visible 15-minute lockout",
            lock_statuses == [401, 401, 401, 401, 423] and lock_row is not None,
        )
        with engine.begin() as connection:
            connection.execute(
                text("UPDATE users SET failed_login_count=0, locked_until=NULL WHERE email=:email"),
                {"email": email_a},
            )

        status, recovery_step1 = _login(api, email_a, password_a)
        secrets_to_check.append(recovery_step1["preauth_token"])
        status, recovery_session = api.request(
            "POST",
            "/auth/login/2fa",
            body={"preauth_token": recovery_step1["preauth_token"], "code": recovery_a[0]},
        )
        require("recovery code grants one login", status == 200)
        recovery_token = recovery_session["access_token"]
        secrets_to_check.append(recovery_token)
        status, recovery_replay_step1 = _login(api, email_a, password_a)
        status, recovery_replay = api.request(
            "POST",
            "/auth/login/2fa",
            body={
                "preauth_token": recovery_replay_step1["preauth_token"],
                "code": recovery_a[0],
            },
        )
        require("used recovery code cannot be reused", status == 401)
        secrets_to_check.append(recovery_replay_step1["preauth_token"])

        status, registration_b = api.request(
            "POST",
            "/auth/register",
            body={"email": email_b, "password": password_b, "confirm_password": password_b},
        )
        secret_b = registration_b["totp_secret"]
        secrets_to_check.extend([secret_b, registration_b["enrollment_token"]])
        confirmation_code_b = pyotp.TOTP(secret_b).now()
        secrets_to_check.append(confirmation_code_b)
        status, confirmation_b = api.request(
            "POST",
            "/auth/register/confirm-2fa",
            body={
                "enrollment_token": registration_b["enrollment_token"],
                "code": confirmation_code_b,
            },
        )
        require("second account 2FA enrollment succeeds", status == 200)
        recovery_b = confirmation_b["recovery_codes"]
        secrets_to_check.extend(recovery_b)
        status, preauth_b = _login(api, email_b, password_b)
        status, session_b = api.request(
            "POST",
            "/auth/login/2fa",
            body={"preauth_token": preauth_b["preauth_token"], "code": recovery_b[0]},
        )
        require("second account can authenticate independently", status == 200)
        token_b = session_b["access_token"]
        secrets_to_check.extend([preauth_b["preauth_token"], token_b])

        status, first = api.request("POST", "/conversations", body={}, token=token_a)
        conversation_a = first["id"]
        status, second = api.request("POST", "/conversations", body={}, token=token_a)
        conversation_b = second["id"]
        require("two isolated conversations created", status == 201 and conversation_a != conversation_b)
        prompts = [
            (conversation_a, "recommend a game like Portal 2", "something shorter and cheaper than that"),
            (conversation_b, "recommend a movie like The Matrix", "something more recent than that"),
        ]
        for conversation_id, initial, followup in prompts:
            for message in (initial, followup):
                status, turn = api.request(
                    "POST",
                    f"/conversations/{conversation_id}/messages",
                    body={"message": message},
                    token=token_a,
                )
                require(
                    f"chat turn persisted in conversation {conversation_id[:8]}",
                    status == 200 and bool(turn["result"].get("answer")),
                )
        from sqlalchemy.orm import Session

        from entdec.auth.models import Conversation

        with Session(engine) as db:
            state_a = json.loads(db.get(Conversation, conversation_a).state_json)
            state_b = json.loads(db.get(Conversation, conversation_b).state_json)
        require(
            "follow-up memory is retained and conversation state is isolated",
            len(state_a["turns"]) == 2
            and len(state_b["turns"]) == 2
            and state_a["turns"][0]["user_message"] == prompts[0][1]
            and state_b["turns"][0]["user_message"] == prompts[1][1]
            and prompts[0][1] not in json.dumps(state_b)
            and prompts[1][1] not in json.dumps(state_a),
        )
        require(
            "user B cannot read, rename, or delete user A conversation",
            api.request("GET", f"/conversations/{conversation_a}", token=token_b)[0] == 404
            and api.request(
                "PATCH",
                f"/conversations/{conversation_a}",
                body={"title": "unauthorized"},
                token=token_b,
            )[0]
            == 404
            and api.request("DELETE", f"/conversations/{conversation_a}", token=token_b)[0] == 404,
        )
        status, listing_b = api.request("GET", "/conversations", token=token_b)
        require(
            "user B lists only owned conversations",
            status == 200 and listing_b["total"] == 0,
        )

        server.restart()
        api = Api(server.base_url)
        status, me = api.request("GET", "/auth/me", token=token_a)
        status_list, listing_a = api.request("GET", "/conversations", token=token_a)
        status_get, reopened = api.request("GET", f"/conversations/{conversation_a}", token=token_a)
        require(
            "session, conversation list, and messages survive API restart",
            status == 200
            and me["email"] == email_a
            and status_list == 200
            and listing_a["total"] == 2
            and status_get == 200
            and len(reopened["messages"]) == 4,
        )
        status, restored_turn = api.request(
            "POST",
            f"/conversations/{conversation_a}/messages",
            body={"message": "one more follow-up based on the earlier game"},
            token=token_a,
        )
        require(
            "restarted API restores old memory for follow-up",
            status == 200 and bool(restored_turn["result"].get("answer")),
        )
        status, renamed = api.request(
            "PATCH",
            f"/conversations/{conversation_a}",
            body={"title": "Verified smoke conversation"},
            token=token_a,
        )
        status_delete = api.request("DELETE", f"/conversations/{conversation_b}", token=token_a)[0]
        _, list_after_delete = api.request("GET", "/conversations", token=token_a)
        require(
            "rename and soft-delete update visible history",
            status == 200
            and renamed["title"] == "Verified smoke conversation"
            and status_delete == 204
            and list_after_delete["total"] == 1
            and all(item["id"] != conversation_b for item in list_after_delete["items"]),
        )
        _check_database_secrets(
            engine,
            email=email_a,
            password=password_a,
            totp_secret=secret_a,
            recovery_codes=recovery_a,
            preauth_token=recovery_replay_step1["preauth_token"],
            session_token=recovery_token,
        )
        status, _ = api.request("POST", "/auth/logout", token=token_a)
        old_status, _ = api.request("GET", "/auth/me", token=token_a)
        require("logout revokes old session", status == 204 and old_status == 401)
        invalid_status, _ = api.request("GET", "/auth/me", token="invalid-token")
        secrets_to_check.append("invalid-token")
        require("invalid token remains unauthorized", invalid_status == 401)

        _stop_mysql()
        mysql_stopped = True
        outage_status, outage_body = api.request("GET", "/conversations", token=token_b)
        require(
            "MySQL outage returns a clean 503 without traceback",
            outage_status == 503
            and isinstance(outage_body, dict)
            and "temporarily unavailable" in outage_body.get("detail", "").lower()
            and "traceback" not in json.dumps(outage_body).lower(),
        )
        _start_mysql()
        mysql_stopped = False
        status, _ = api.request("GET", "/auth/me", token=token_b)
        require("API recovers after MySQL returns without restart", status == 200)

        user_row = _scalar(
            engine,
            "SELECT password_hash FROM users WHERE email=:email",
            {"email": email_a},
        )
        require("database retains no throwaway password in plaintext", user_row is not None and password_a not in user_row)
        api_log_paths = [server.log_path]
        repo_logs = ROOT / "logs"
        if repo_logs.is_dir():
            api_log_paths.extend(path for path in repo_logs.rglob("*") if path.is_file())
        _check_logs(api_log_paths, secrets_to_check)
    finally:
        if mysql_stopped:
            try:
                _start_mysql()
            except Exception:
                print("FAIL unable to restart entdec-mysql during cleanup")
        server.stop()
        if engine is not None:
            engine.dispose()
        for email in (email_a, email_b):
            try:
                _delete_user(email)
            except Exception:
                print("WARN cleanup via auth delete-user needs manual verification")
        if server.log_path.exists():
            server.log_path.unlink()


if __name__ == "__main__":
    try:
        run_smoke()
    except SmokeFailure as exc:
        print(f"FAIL {exc}")
        raise SystemExit(1) from None
    except Exception:
        print("FAIL unexpected smoke-test error; details withheld to avoid secret disclosure")
        raise SystemExit(1) from None
