"""The RFC 7523 jwt-bearer grant on ``POST /token``: a requesting app that obtained an
ID-JAG (Identity Assertion Authorization Grant, the cross-app access profile) from the
customer's identity provider redeems it for the same proxy-API credential ``lite login``
stores. The assertion is proven with the JWT auth issuer and JWKS plumbing the proxy
already trusts, its claims are checked the way the ID-JAG draft prescribes, the requesting
app must be one the operator allowlisted, and the ``jti`` is single-use until the
assertion expires."""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import math
from collections.abc import Awaitable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Final, Protocol
from urllib.parse import unquote

import jwt
from fastapi import Request

from litellm._logging import verbose_proxy_logger
from litellm.proxy._experimental.mcp_server.gateway_dcr_flow import (
    AssertionIdentity,
    ConsentTeam,
    SubjectIdentity,
    SubjectTokenRefusal,
)
from litellm.proxy._experimental.mcp_server.idp_token_exchange import (
    gateway_outage_refusal,
    log_line,
    read_token_exchange_prerequisites,
)
from litellm.proxy._experimental.mcp_server.oauth_utils import get_request_base_url
from litellm.proxy._types import IdJagRequestingApp
from litellm.proxy.auth.handle_jwt import ID_JAG_TYP, AssertionRejection, JWTHandler

ID_JAG_REQUIRED_CLAIMS: Final = ("iss", "sub", "aud", "exp", "iat", "jti", "client_id")
REJECTED_ASSERTION: Final = "assertion was rejected"
TEAM_HEADER: Final = "x-litellm-team-id"
_REPLAY_KEY_PREFIX: Final = "mcp_gateway_dcr:id_jag:"


@dataclass(frozen=True, slots=True)
class IdJagPrerequisites:
    """The deployment-level gates for redeeming an ID-JAG: the JWT auth whose issuer plumbing
    proves the assertion must be on, backed by a database, and licensed, not mapping IdP tokens to
    virtual keys (whose blocks and budgets a minted credential would sidestep), and the operator
    must have allowlisted the requesting apps. Discovery advertises the grant only when every gate
    holds, and a redemption is refused naming the first one that does not."""

    jwt_auth_enabled: bool
    has_database: bool
    licensed: bool
    maps_jwts_to_virtual_keys: bool
    requesting_apps: tuple[IdJagRequestingApp, ...]

    @property
    def available(self) -> bool:
        return (
            self.jwt_auth_enabled
            and self.has_database
            and self.licensed
            and not self.maps_jwts_to_virtual_keys
            and bool(self.requesting_apps)
        )

    def refusal(self) -> SubjectTokenRefusal | None:
        if not self.jwt_auth_enabled:
            return SubjectTokenRefusal(
                error="unsupported_grant_type",
                description="JWT auth is not enabled on this gateway, so it cannot redeem identity assertions",
            )
        if not self.has_database:
            return SubjectTokenRefusal(
                error="unsupported_grant_type",
                description="this gateway has no database, so it cannot redeem identity assertions",
            )
        if not self.licensed:
            return SubjectTokenRefusal(
                error="unsupported_grant_type",
                description="JWT auth is an enterprise only feature; no license is set",
            )
        if self.maps_jwts_to_virtual_keys:
            return SubjectTokenRefusal(
                error="unsupported_grant_type",
                description="this gateway maps IdP tokens to virtual keys, which the jwt-bearer grant does not serve",
            )
        if not self.requesting_apps:
            return SubjectTokenRefusal(
                error="unsupported_grant_type",
                description="this gateway has no requesting apps allowlisted for identity assertions",
            )
        return None


def read_id_jag_prerequisites() -> IdJagPrerequisites:
    from litellm.proxy.proxy_server import jwt_handler  # noqa: PLC0415  # rebound after startup, so read it per call

    shared: Final = read_token_exchange_prerequisites()
    return IdJagPrerequisites(
        jwt_auth_enabled=shared.jwt_auth_enabled,
        has_database=shared.has_database,
        licensed=shared.licensed,
        maps_jwts_to_virtual_keys=shared.maps_jwts_to_virtual_keys,
        requesting_apps=_requesting_apps(jwt_handler),
    )


def _requesting_apps(jwt_handler: JWTHandler) -> tuple[IdJagRequestingApp, ...]:
    if not hasattr(jwt_handler, "litellm_jwtauth"):
        return ()
    return tuple(jwt_handler.litellm_jwtauth.id_jag_requesting_apps or ())


def jwt_bearer_available() -> bool:
    return read_id_jag_prerequisites().available


class DecodeAssertion(Protocol):
    """Injected signature and claim verification ``(assertion, audience, required_claims)``:
    ``JWTHandler.decode_trusted_assertion`` in production, which returns the verified claims,
    a rejection, or raises when the gateway could not reach the issuer's keys."""

    def __call__(
        self, assertion: str, audience: Sequence[str], required_claims: Sequence[str], /
    ) -> Awaitable[Mapping[str, object] | AssertionRejection]: ...


class ClaimReader(Protocol):
    """The JWT auth user mapping as ``JWTHandler`` exposes it: the litellm user id and email the
    verified claims name through ``user_id_jwt_field`` and ``user_email_jwt_field``."""

    def get_user_id(self, token: dict[str, object], default_value: str | None) -> str | None: ...

    def get_user_email(self, token: dict[str, object], default_value: str | None) -> str | None: ...

    def get_all_jwt_team_ids(self, token: dict[str, object]) -> list[str]: ...


class ResolveSubject(Protocol):
    """Injected subject lookup ``(claims, team_header)``: the litellm user the verified claims
    name through the JWT auth user mapping (``user_id_jwt_field`` or ``user_email_jwt_field``,
    else ``sub``) and the team the credential is minted for (see ``select_team``), ``None`` when
    the gateway knows no such user, or a refusal when the team cannot be settled. Raises on a
    database outage."""

    def __call__(
        self, claims: Mapping[str, object], team_header: str | None, /
    ) -> Awaitable[SubjectIdentity | SubjectTokenRefusal | None]: ...


class LookupUser(Protocol):
    """Injected ``(user_id, user_email)`` lookup: the stored id of the existing litellm user the
    claims name, ``None`` when the gateway knows no such user. Raises on a database outage."""

    def __call__(self, user_id: str, user_email: str | None, /) -> Awaitable[str | None]: ...


class LookupTeams(Protocol):
    """Injected lookup of the teams a user is on, or the failure text when they cannot be read."""

    def __call__(self, user_id: str, /) -> Awaitable[Sequence[ConsentTeam] | str]: ...


def subject_resolver(reader: ClaimReader, lookup_user: LookupUser, lookup_teams: LookupTeams) -> ResolveSubject:
    """``ResolveSubject`` over the JWT auth claim mapping and the proxy's user and team lookups."""

    async def resolve_subject(
        claims: Mapping[str, object], team_header: str | None
    ) -> SubjectIdentity | SubjectTokenRefusal | None:
        token: Final = dict(claims)
        subject: Final = claims.get("sub")
        claimed_user_id: Final = reader.get_user_id(token=token, default_value=None) or (
            subject if isinstance(subject, str) else None
        )
        if claimed_user_id is None:
            return None
        user_id: Final = await lookup_user(claimed_user_id, reader.get_user_email(token=token, default_value=None))
        if user_id is None:
            return None
        teams: Final = await lookup_teams(user_id)
        if isinstance(teams, str):
            return SubjectTokenRefusal(
                error="temporarily_unavailable", description="the gateway could not read this user's teams; retry"
            )
        team: Final = select_team(team_header, tuple(reader.get_all_jwt_team_ids(token=token)), teams)
        if isinstance(team, SubjectTokenRefusal):
            return team
        return SubjectIdentity(user_id=user_id, team_id=team)

    return resolve_subject


def select_team(
    team_header: str | None, claim_teams: Sequence[str], teams: Sequence[ConsentTeam]
) -> str | None | SubjectTokenRefusal:
    """The team an ID-JAG redemption mints for, settled before the assertion is consumed since
    the minter refuses a teamless credential for a user who has a team. The ``x-litellm-team-id``
    header (a team id or alias) wins and must name one of the user's teams, then the assertion's
    mapped team claim, then the user's only team. A user on several teams must name one."""
    if team_header is not None:
        named: Final = next((team for team in teams if team_header in (team.team_id, team.team_alias)), None)
        if named is None:
            return SubjectTokenRefusal(
                error="invalid_request", description=f"{TEAM_HEADER} does not name one of this user's teams"
            )
        return named.team_id
    if claim_teams:
        asserted: Final = tuple(team.team_id for team in teams if team.team_id in claim_teams)
        match asserted:
            case ():
                return _rejected("the assertion's team claims name no team the user is on")
            case (only,):
                return only
            case _:
                return SubjectTokenRefusal(
                    error="invalid_request",
                    description=f"the assertion names several of this user's teams; send the {TEAM_HEADER} header naming the team for this credential",
                )
    if len(teams) == 1:
        return teams[0].team_id
    if teams:
        return SubjectTokenRefusal(
            error="invalid_request",
            description=f"this user belongs to several teams; send the {TEAM_HEADER} header naming the team for this credential",
        )
    return None


@dataclass(frozen=True, slots=True)
class AssertionRedemption:
    """One ``jwt-bearer`` request as the grant sees it: the assertion, how the requesting app
    identified and authenticated itself, the team the credential is asked for, the issuer
    values this gateway answers to (what the assertion's ``aud`` must name), the clock, and the
    leeway the verifier grants ``exp``, so the replay marker outlives every moment the assertion
    would still verify."""

    assertion: str
    client_id: str
    client_secret: str | None
    authorization_header: str | None
    team_id: str | None
    audiences: tuple[str, ...]
    now: float
    leeway_seconds: float


async def redeem_id_jag_assertion(
    assertion: str, client_id: str, client_secret: str | None, request: Request
) -> AssertionIdentity | SubjectTokenRefusal:
    from litellm.proxy._experimental.mcp_server.proxy_api_credentials import (  # noqa: PLC0415  # circular import at module load
        lookup_consent_teams,
    )
    from litellm.proxy.auth.auth_checks import get_user_object  # noqa: PLC0415  # circular import at module load
    from litellm.proxy.proxy_server import (  # noqa: PLC0415  # rebound after startup, so read them per call
        jwt_handler,
        prisma_client,
        proxy_logging_obj,
        user_api_key_cache,
    )

    async def lookup_user(user_id: str, user_email: str | None) -> str | None:
        user: Final = await get_user_object(
            user_id=user_id,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            user_id_upsert=False,
            proxy_logging_obj=proxy_logging_obj,
            user_email=user_email,
            check_db_only=True,
        )
        return None if user is None else user.user_id

    resolve_subject: Final = subject_resolver(_claim_reader(jwt_handler), lookup_user, lookup_consent_teams)
    base_url: Final = get_request_base_url(request)
    return await identity_from_id_jag(
        AssertionRedemption(
            assertion=assertion,
            client_id=client_id,
            client_secret=client_secret,
            authorization_header=request.headers.get("authorization"),
            team_id=request.headers.get(TEAM_HEADER),
            audiences=(f"{base_url}/mcp", base_url),
            now=datetime.now(timezone.utc).timestamp(),
            leeway_seconds=jwt_handler.leeway,
        ),
        prerequisites=read_id_jag_prerequisites(),
        decode=jwt_handler.decode_trusted_assertion,
        resolve_subject=resolve_subject,
    )


def _claim_reader(handler: ClaimReader) -> ClaimReader:
    return handler


async def identity_from_id_jag(
    redemption: AssertionRedemption,
    prerequisites: IdJagPrerequisites,
    decode: DecodeAssertion,
    resolve_subject: ResolveSubject,
) -> AssertionIdentity | SubjectTokenRefusal:
    """Prove an ID-JAG for the grant tail. The requesting app is checked first, the way a token
    endpoint authenticates its client before it looks at the grant, so an app the operator never
    allowlisted learns nothing about the assertion. Everything wrong with the assertion itself is
    ``invalid_grant`` (RFC 7523 section 3.1), a check the gateway could not complete is
    ``temporarily_unavailable``, and the reason stays in the proxy log because this endpoint is
    public and the verifier's wording can quote the assertion."""
    unmet: Final = prerequisites.refusal()
    if unmet is not None:
        return unmet
    app: Final = next((app for app in prerequisites.requesting_apps if app.client_id == redemption.client_id), None)
    if app is None:
        return SubjectTokenRefusal(error="invalid_client", description="client_id is not an allowlisted requesting app")
    if not _client_authenticated(app, redemption.client_secret, redemption.authorization_header):
        return SubjectTokenRefusal(error="invalid_client", description="client authentication failed")
    try:
        header: Final = jwt.get_unverified_header(redemption.assertion)
    except jwt.PyJWTError:
        return SubjectTokenRefusal(error="invalid_grant", description="assertion is not a JWT")
    if header.get("typ") != ID_JAG_TYP:
        return SubjectTokenRefusal(error="invalid_grant", description=f"assertion typ must be {ID_JAG_TYP}")
    try:
        verified: Final = await decode(redemption.assertion, redemption.audiences, ID_JAG_REQUIRED_CLAIMS)
    except Exception as denied:  # noqa: BLE001  # the verifier raises its 503 for a JWKS outage, and anything else is a bad assertion
        return gateway_outage_refusal(denied, denied, what="an assertion", parameter="assertion") or _rejected(
            str(denied)
        )
    if isinstance(verified, AssertionRejection):
        return _rejected(verified.reason)
    if verified.get("client_id") != redemption.client_id:
        return _rejected("the assertion's client_id is not the requesting app that presented it")
    if "cnf" in verified:
        return _rejected("the assertion is proof-bound and this gateway does not check proof of possession")
    issuer: Final = verified.get("iss")
    jti: Final = verified.get("jti")
    exp: Final = verified.get("exp")
    if not isinstance(issuer, str) or not isinstance(jti, str) or not jti or not isinstance(exp, (int, float)):
        return _rejected("the assertion's iss, jti, or exp claim has the wrong shape")
    replay_ttl_seconds: Final = math.ceil(exp + redemption.leeway_seconds - redemption.now)
    if replay_ttl_seconds <= 0:
        return _rejected("the assertion has expired")
    scope: Final = verified.get("scope")
    try:
        subject: Final = await resolve_subject(verified, redemption.team_id)
    except Exception as failed:  # noqa: BLE001  # the user lookup raises for a missing user and for a database outage alike
        return gateway_outage_refusal(
            failed, failed, what="an assertion's subject", parameter="assertion"
        ) or _rejected(f"subject lookup failed: {failed}")
    if subject is None:
        return _rejected("the assertion names no user the gateway knows")
    if isinstance(subject, SubjectTokenRefusal):
        return subject
    return AssertionIdentity(
        user_id=subject.user_id,
        team_id=subject.team_id,
        replay_key=_REPLAY_KEY_PREFIX + hashlib.sha256(f"{issuer}\n{jti}".encode()).hexdigest(),
        replay_ttl_seconds=replay_ttl_seconds,
        scope=scope if isinstance(scope, str) and scope else None,
    )


def _rejected(reason: str) -> SubjectTokenRefusal:
    verbose_proxy_logger.warning("jwt-bearer grant refused an assertion: %s", log_line(reason))
    return SubjectTokenRefusal(error="invalid_grant", description=REJECTED_ASSERTION)


def _client_authenticated(app: IdJagRequestingApp, form_secret: str | None, authorization_header: str | None) -> bool:
    """A public app (no secret configured) is identified by ``client_id`` alone. A confidential
    app must present its secret, by HTTP Basic (RFC 6749 section 2.3.1, the credentials
    form-urlencoded inside the base64) or the ``client_secret`` form field, compared in constant
    time."""
    if app.client_secret is None:
        return True
    basic: Final = _basic_credentials(authorization_header)
    if basic is not None:
        return basic[0] == app.client_id and _same_secret(basic[1], app.client_secret)
    return form_secret is not None and _same_secret(form_secret, app.client_secret)


def _same_secret(presented: str, expected: str) -> bool:
    return hmac.compare_digest(presented.encode(), expected.encode())


def basic_client_id(authorization_header: str | None) -> str | None:
    """The ``client_id`` an HTTP Basic ``Authorization`` header carries, for a confidential app
    that leaves the form field out as RFC 6749 section 2.3.1 allows."""
    credentials: Final = _basic_credentials(authorization_header)
    return None if credentials is None else credentials[0]


def _basic_credentials(authorization_header: str | None) -> tuple[str, str] | None:
    if authorization_header is None:
        return None
    scheme, _, encoded = authorization_header.strip().partition(" ")
    if scheme.lower() != "basic" or not encoded:
        return None
    try:
        decoded: Final = base64.b64decode(encoded.strip(), validate=True).decode()
    except (binascii.Error, UnicodeDecodeError):
        return None
    client_id, separator, secret = decoded.partition(":")
    if not separator:
        return None
    return unquote(client_id), unquote(secret)
