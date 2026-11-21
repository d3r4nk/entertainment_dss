"""Account registration, TOTP verification, and bearer-session services."""

from __future__ import annotations

import base64
import hashlib
import hmac
import io
import re
import secrets
import threading
import time
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError
from cryptography.fernet import Fernet, InvalidToken
from email_validator import EmailNotValidError, validate_email
from sqlalchemy import delete, select
from sqlalchemy.orm import Session, sessionmaker

from entdec.auth.models import (
    AuditLog,
    AuthSession,
    EnrollmentToken,
    PreAuthToken,
    RecoveryCode,
    User,
    utc_now,
)

_COMMON_PASSWORDS = {
    "password",
    "password123",
    "1234567890",
    "qwerty12345",
    "letmein123",
    "admin123456",
    "iloveyou123",
    "welcome123",
}
_DUMMY_PASSWORD_HASH = PasswordHasher().hash(secrets.token_urlsafe(24))


class AuthError(Exception):
    """Expected authentication failure with an HTTP-compatible status."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class AuthService:
    """Pure service layer for user and authentication operations."""

    def __init__(
        self,
        sessions: sessionmaker[Session],
        secret_key: str,
        *,
        session_days: int = 7,
        clock: Any = time.time,
    ) -> None:
        if not secret_key:
            raise RuntimeError(
                "ENTDEC_SECRET_KEY is required when authentication is enabled; "
                "set a URL-safe 32-byte Fernet key in .env."
            )
        try:
            self._fernet = Fernet(secret_key.encode("ascii"))
        except (ValueError, UnicodeEncodeError) as exc:
            raise RuntimeError("ENTDEC_SECRET_KEY must be a valid Fernet key.") from exc
        self.sessions = sessions
        self.session_days = max(1, int(session_days))
        self._clock = clock
        self._rate_lock = threading.Lock()
        self._rate_events: dict[str, list[float]] = {}
        self._last_rate_cleanup = self._clock()
        self._password_hasher = PasswordHasher()

    def register(self, email: str, password: str, *, ip: str | None = None) -> dict[str, str]:
        """Create an inactive account and return its one-time 2FA enrollment data."""
        address = self._normalize_email(email)
        self._limit("register", address, ip)
        self._validate_password(address, password)

        import pyotp
        import qrcode

        now = self._now()
        secret = pyotp.random_base32()
        enrollment_token = secrets.token_urlsafe(32)
        user_id = str(uuid.uuid4())
        totp_uri = pyotp.TOTP(secret).provisioning_uri(
            name=address, issuer_name="Entertainment Decision"
        )
        image = qrcode.make(totp_uri)
        output = io.BytesIO()
        image.save(output, format="PNG")

        with self.sessions() as db, db.begin():
            user = db.scalar(select(User).where(User.email == address).with_for_update())
            if user is not None:
                if not user.is_active and _aware(user.created_at) <= now - timedelta(hours=24):
                    db.delete(user)
                    db.flush()
                else:
                    raise AuthError(409, "Unable to register this email address.")
            user = User(
                id=user_id,
                email=address,
                password_hash=self._password_hasher.hash(password),
                totp_secret_encrypted=self._fernet.encrypt(secret.encode()).decode(),
                totp_enabled=False,
                is_active=False,
                created_at=now,
            )
            db.add(user)
            db.flush()
            db.add(
                EnrollmentToken(
                    id=str(uuid.uuid4()),
                    user_id=user_id,
                    token_hash=_token_hash(enrollment_token),
                    expires_at=now + timedelta(hours=24),
                )
            )
            _audit(db, "register", user_id=user_id, email=address, ip=ip)

        return {
            "enrollment_token": enrollment_token,
            "totp_uri": totp_uri,
            "totp_secret": secret,
            "qr_code_png_base64": base64.b64encode(output.getvalue()).decode("ascii"),
        }

    def confirm_registration(
        self, enrollment_token: str, code: str, *, ip: str | None = None
    ) -> list[str]:
        """Activate a pending account after its first valid TOTP confirmation."""
        now = self._now()
        with self.sessions() as db, db.begin():
            enrollment = self._find_one_time_token(db, EnrollmentToken, enrollment_token, now)
            if enrollment is None:
                raise AuthError(401, "Invalid or expired enrollment token.")
            user = db.scalar(select(User).where(User.id == enrollment.user_id).with_for_update())
            if user is None or user.totp_enabled:
                raise AuthError(401, "Invalid or expired enrollment token.")
            self._limit("register_confirm", user.email, ip)
            secret = self._decrypt_secret(user)
            step = _verified_totp_step(secret, code, now, after=-1)
            if step is None:
                raise AuthError(401, "Invalid authentication code.")

            user.totp_enabled = True
            user.is_active = True
            user.totp_last_used_step = step
            recovery_codes = [_recovery_code() for _ in range(10)]
            db.add_all(
                RecoveryCode(
                    id=str(uuid.uuid4()),
                    user_id=user.id,
                    code_hash=_code_hash(recovery_code),
                )
                for recovery_code in recovery_codes
            )
            db.delete(enrollment)
            _audit(db, "register", user_id=user.id, email=user.email, ip=ip)
        return recovery_codes

    def login(self, email: str, password: str, *, ip: str | None = None) -> str:
        """Verify the password and issue a short-lived, single-purpose token."""
        address = self._normalize_email(email)
        self._limit("login", address, ip)
        now = self._now()
        error: AuthError | None = None
        token: str | None = None
        with self.sessions() as db, db.begin():
            user = db.scalar(select(User).where(User.email == address).with_for_update())
            if user is None:
                self._verify_password(password, _DUMMY_PASSWORD_HASH)
                _audit(db, "login_fail", email=address, ip=ip)
                error = AuthError(401, "Invalid email or password.")
            elif user.locked_until and _aware(user.locked_until) > now:
                error = AuthError(423, "Account temporarily locked.")
            else:
                password_valid = self._verify_password(password, user.password_hash)
                if not user.is_active or not user.totp_enabled or not password_valid:
                    self._record_failure(db, user, "login_fail", ip)
                    locked = user.failed_login_count >= 5
                    error = AuthError(
                        423 if locked else 401,
                        "Account temporarily locked" if locked else "Invalid email or password.",
                    )
                else:
                    token = secrets.token_urlsafe(32)
                    db.add(
                        PreAuthToken(
                            id=str(uuid.uuid4()),
                            user_id=user.id,
                            token_hash=_token_hash(token),
                            expires_at=now + timedelta(minutes=5),
                        )
                    )
        if error is not None:
            raise error
        if token is None:
            raise RuntimeError("Password verification did not produce a login token.")
        return token

    def login_2fa(
        self,
        preauth_token: str,
        code: str,
        *,
        user_agent: str | None = None,
        ip: str | None = None,
    ) -> str:
        """Complete a login with TOTP or one unused recovery code."""
        now = self._now()
        error: AuthError | None = None
        raw_token: str | None = None
        with self.sessions() as db, db.begin():
            preauth = self._find_one_time_token(db, PreAuthToken, preauth_token, now)
            if preauth is None:
                error = AuthError(401, "Invalid or expired pre-authentication token.")
            else:
                user = db.scalar(select(User).where(User.id == preauth.user_id).with_for_update())
                if user is None or not user.is_active or not user.totp_enabled:
                    error = AuthError(401, "Invalid authentication code.")
                elif user.locked_until and _aware(user.locked_until) > now:
                    error = AuthError(423, "Account temporarily locked.")
                else:
                    self._limit("login_2fa", user.email, ip)
                    is_recovery = self._consume_recovery_code(db, user.id, code, now)
                    step = (
                        None
                        if is_recovery
                        else _verified_totp_step(
                            self._decrypt_secret(user),
                            code,
                            now,
                            after=user.totp_last_used_step,
                        )
                    )
                    if not is_recovery and step is None:
                        self._record_failure(db, user, "2fa_fail", ip)
                        locked = user.failed_login_count >= 5
                        error = AuthError(
                            423 if locked else 401,
                            "Account temporarily locked"
                            if locked
                            else "Invalid authentication code.",
                        )
                    else:
                        if step is not None:
                            user.totp_last_used_step = step
                        user.failed_login_count = 0
                        user.locked_until = None
                        db.delete(preauth)
                        if is_recovery:
                            _audit(
                                db,
                                "recovery_used",
                                user_id=user.id,
                                email=user.email,
                                ip=ip,
                            )
                        _audit(db, "login_ok", user_id=user.id, email=user.email, ip=ip)
                        raw_token = secrets.token_urlsafe(32)
                        db.add(
                            AuthSession(
                                id=str(uuid.uuid4()),
                                user_id=user.id,
                                token_hash=_token_hash(raw_token),
                                created_at=now,
                                expires_at=now + timedelta(days=self.session_days),
                                last_seen_at=now,
                                user_agent=(user_agent or "")[:512] or None,
                            )
                        )
        if error is not None:
            raise error
        if raw_token is None:
            raise RuntimeError("Second-factor verification did not produce a session.")
        return raw_token

    def authenticate(self, token: str, *, ip: str | None = None) -> dict[str, str]:
        """Validate a bearer token and slide its session expiry."""
        now = self._now()
        token_hash = _token_hash(token)
        with self.sessions() as db, db.begin():
            auth_session = db.scalar(
                select(AuthSession).where(AuthSession.token_hash == token_hash).with_for_update()
            )
            if (
                auth_session is None
                or not hmac.compare_digest(auth_session.token_hash, token_hash)
                or auth_session.revoked_at is not None
                or _aware(auth_session.expires_at) <= now
            ):
                raise AuthError(401, "Authentication required.")
            user = db.get(User, auth_session.user_id)
            if user is None or not user.is_active:
                raise AuthError(401, "Authentication required.")
            self._limit("authenticated_request", user.email, ip)
            auth_session.last_seen_at = now
            auth_session.expires_at = now + timedelta(days=self.session_days)
            return {"id": user.id, "email": user.email}

    def logout(self, token: str, *, ip: str | None = None) -> None:
        """Revoke the current bearer session."""
        now = self._now()
        token_hash = _token_hash(token)
        with self.sessions() as db, db.begin():
            auth_session = db.scalar(
                select(AuthSession).where(AuthSession.token_hash == token_hash).with_for_update()
            )
            if (
                auth_session is None
                or not hmac.compare_digest(auth_session.token_hash, token_hash)
                or auth_session.revoked_at is not None
            ):
                return
            user = db.get(User, auth_session.user_id)
            if user is not None:
                self._limit("logout", user.email, ip)
            auth_session.revoked_at = now
            _audit(
                db,
                "logout",
                user_id=auth_session.user_id,
                email=user.email if user else None,
                ip=ip,
            )

    def list_users(self) -> list[dict[str, Any]]:
        """Return non-secret user details for local administration."""
        with self.sessions() as db:
            return [
                {
                    "id": user.id,
                    "email": user.email,
                    "is_active": user.is_active,
                    "totp_enabled": user.totp_enabled,
                    "created_at": _aware(user.created_at).isoformat(),
                }
                for user in db.scalars(select(User).order_by(User.created_at)).all()
            ]

    def delete_user(self, email: str) -> bool:
        """Delete a user and all dependent account data by normalized email."""
        address = self._normalize_email(email)
        with self.sessions() as db, db.begin():
            user = db.scalar(select(User).where(User.email == address).with_for_update())
            if user is None:
                return False
            db.delete(user)
            return True

    def reset_2fa(self, email: str) -> dict[str, str] | None:
        """Rotate an account's 2FA seed; the user must enroll it again."""
        import pyotp

        address = self._normalize_email(email)
        secret = pyotp.random_base32()
        enrollment_token = secrets.token_urlsafe(32)
        now = self._now()
        with self.sessions() as db, db.begin():
            user = db.scalar(select(User).where(User.email == address).with_for_update())
            if user is None:
                return None
            user.totp_secret_encrypted = self._fernet.encrypt(secret.encode()).decode()
            user.totp_enabled = False
            user.is_active = False
            user.totp_last_used_step = None
            user.failed_login_count = 0
            user.locked_until = None
            db.execute(delete(RecoveryCode).where(RecoveryCode.user_id == user.id))
            db.execute(delete(AuthSession).where(AuthSession.user_id == user.id))
            db.execute(delete(PreAuthToken).where(PreAuthToken.user_id == user.id))
            db.add(
                EnrollmentToken(
                    id=str(uuid.uuid4()),
                    user_id=user.id,
                    token_hash=_token_hash(enrollment_token),
                    expires_at=now + timedelta(hours=24),
                )
            )
            _audit(db, "2fa_reset", user_id=user.id, email=user.email)
        uri = pyotp.TOTP(secret).provisioning_uri(
            name=address, issuer_name="Entertainment Decision"
        )
        return {
            "enrollment_token": enrollment_token,
            "totp_secret": secret,
            "totp_uri": uri,
        }

    def _find_one_time_token(
        self, db: Session, model: Any, token: str, now: datetime
    ) -> Any | None:
        token_hash = _token_hash(token)
        record = db.scalar(select(model).where(model.token_hash == token_hash).with_for_update())
        if (
            record is None
            or not hmac.compare_digest(record.token_hash, token_hash)
            or _aware(record.expires_at) <= now
        ):
            return None
        return record

    def _consume_recovery_code(self, db: Session, user_id: str, code: str, now: datetime) -> bool:
        candidate = _code_hash(code)
        for recovery in db.scalars(
            select(RecoveryCode)
            .where(RecoveryCode.user_id == user_id, RecoveryCode.used_at.is_(None))
            .with_for_update()
        ):
            if hmac.compare_digest(recovery.code_hash, candidate):
                recovery.used_at = now
                return True
        return False

    def _record_failure(self, db: Session, user: User, event: str, ip: str | None) -> None:
        user.failed_login_count += 1
        if user.failed_login_count >= 5:
            user.locked_until = self._now() + timedelta(minutes=15)
        _audit(db, event, user_id=user.id, email=user.email, ip=ip)

    def _verify_password(self, password: str, password_hash: str) -> bool:
        try:
            return self._password_hasher.verify(password_hash, password)
        except (VerifyMismatchError, InvalidHashError, ValueError):
            return False

    def _decrypt_secret(self, user: User) -> str:
        if not user.totp_secret_encrypted:
            raise AuthError(401, "Invalid authentication code.")
        try:
            return self._fernet.decrypt(user.totp_secret_encrypted.encode("ascii")).decode("ascii")
        except (InvalidToken, UnicodeDecodeError) as exc:
            raise RuntimeError("Stored TOTP secret cannot be decrypted.") from exc

    def _limit(self, endpoint: str, email: str, ip: str | None) -> None:
        now = self._clock()
        keys = [f"{endpoint}:email:{email}"]
        if ip:
            keys.append(f"{endpoint}:ip:{ip}")
        with self._rate_lock:
            if now - self._last_rate_cleanup >= 60:
                self._rate_events = {
                    key: [stamp for stamp in events if now - stamp < 60]
                    for key, events in self._rate_events.items()
                }
                self._rate_events = {
                    key: events for key, events in self._rate_events.items() if events
                }
                self._last_rate_cleanup = now
            for key in keys:
                events = [stamp for stamp in self._rate_events.get(key, []) if now - stamp < 60]
                if len(events) >= 30:
                    raise AuthError(429, "Too many requests; try again shortly.")
                self._rate_events[key] = events
            for key in keys:
                self._rate_events[key].append(now)

    def _now(self) -> datetime:
        return datetime.fromtimestamp(self._clock(), UTC)

    @staticmethod
    def _normalize_email(email: str) -> str:
        raw_email = email.strip()
        try:
            address = validate_email(raw_email, check_deliverability=False).normalized.lower()
        except EmailNotValidError as exc:
            domain = raw_email.rpartition("@")[2].casefold()
            if not domain.endswith(".test"):
                raise AuthError(422, "Enter a valid email address.") from exc
            try:
                address = (
                    validate_email(
                        raw_email,
                        check_deliverability=False,
                        test_environment=True,
                    )
                    .normalized.lower()
                )
            except EmailNotValidError as test_exc:
                raise AuthError(422, "Enter a valid email address.") from test_exc
        if len(address) > 320:
            raise AuthError(422, "Enter a valid email address.")
        return address

    @staticmethod
    def _validate_password(email: str, password: str) -> None:
        if len(password) < 10:
            raise AuthError(422, "Password must be at least 10 characters.")
        lowered = password.casefold()
        if lowered == email.casefold() or lowered == email.split("@", 1)[0].casefold():
            raise AuthError(422, "Password cannot match the email address.")
        if lowered in _COMMON_PASSWORDS:
            raise AuthError(422, "Choose a less common password.")


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _code_hash(code: str) -> str:
    return hashlib.sha256(code.strip().upper().encode("utf-8")).hexdigest()


def _recovery_code() -> str:
    raw = secrets.token_hex(8).upper()
    return f"{raw[:4]}-{raw[4:8]}-{raw[8:12]}-{raw[12:]}"


def _verified_totp_step(secret: str, code: str, now: datetime, *, after: int | None) -> int | None:
    import pyotp

    if not re.fullmatch(r"\d{6}", code):
        return None
    current_step = int(now.timestamp()) // 30
    for step in range(current_step - 1, current_step + 2):
        if step <= (after if after is not None else -1):
            continue
        expected = pyotp.TOTP(secret).at(step * 30)
        if hmac.compare_digest(expected, code):
            return step
    return None


def _aware(value: datetime | None) -> datetime:
    if value is None:
        return datetime.min.replace(tzinfo=UTC)
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _audit(
    db: Session,
    event: str,
    *,
    user_id: str | None = None,
    email: str | None = None,
    ip: str | None = None,
) -> None:
    email_hash = hashlib.sha256(email.casefold().encode("utf-8")).hexdigest() if email else None
    db.add(
        AuditLog(
            id=str(uuid.uuid4()),
            user_id=user_id,
            email_hash=email_hash,
            event=event,
            ip=(ip or "")[:64] or None,
            created_at=utc_now(),
        )
    )


__all__ = ["AuthError", "AuthService"]
