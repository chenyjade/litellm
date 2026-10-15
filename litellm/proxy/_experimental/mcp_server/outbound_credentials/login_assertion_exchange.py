"""Exchange the identity assertion captured at SSO login for a refresh token, at login.

An ``oauth2_id_jag`` server configured with ``id_jag_subject_source: refresh_token`` does not present the
login assertion on ID-JAG leg 1. It presents a refresh token that the IdP issued for that assertion through
one RFC 8693 exchange (``requested_token_type`` refresh_token), which is how a SAML requesting app works at
an IdP whose ID-JAG exchange does not take a ``saml2`` subject directly. That exchange has to happen at
login: a SAML assertion is valid for minutes, the first MCP call that needs it can come hours later, and
the refresh token outlives the assertion by as long as the IdP decides.

The exchange authenticates as the ID-JAG server's own IdP client, since the refresh token is bound to the
client that later spends it. One row holds one refresh token per user, so every server in this mode has
to share one IdP client; a second client is logged and skipped rather than silently handed a token it
cannot redeem. A refused or unreachable IdP never fails the login, the row keeps the plain assertion and
the ID-JAG call reports the missing refresh token as the precondition it is.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Final, Protocol

from litellm._logging import verbose_proxy_logger
from litellm.proxy._experimental.mcp_server.outbound_credentials.result import Error, Ok, Result
from litellm.proxy._experimental.mcp_server.outbound_credentials.sso_assertion_store import (
    SSOIdentityAssertion,
    with_login_refresh_token,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.token_endpoint import ExchangedToken
from litellm.proxy._experimental.mcp_server.outbound_credentials.types import (
    ClientAuth,
    CredError,
    IdJagConfig,
    IdJagSubjectSource,
    ServerSpec,
)

_TOKEN_EXCHANGE_GRANT_TYPE: Final = "urn:ietf:params:oauth:grant-type:token-exchange"
REFRESH_TOKEN_TOKEN_TYPE: Final = "urn:ietf:params:oauth:token-type:refresh_token"
_REFRESH_TOKEN_SCOPE: Final = "openid offline_access"


class TokenExchangeEndpoint(Protocol):
    async def fetch(
        self, endpoint: str, client_id: str, grant_params: dict[str, str], client_auth: ClientAuth
    ) -> Result[ExchangedToken, CredError]: ...


def refresh_token_exchange_targets(specs: Iterable[ServerSpec | None]) -> tuple[IdJagConfig, ...]:
    """The ID-JAG configs that present a login refresh token, one per IdP client, in registration order."""
    configs: Final = tuple(
        spec.config
        for spec in specs
        if spec is not None
        and isinstance(spec.config, IdJagConfig)
        and spec.config.subject_source is IdJagSubjectSource.refresh_token
    )
    by_client: Final = {(config.org_token_endpoint, config.client_id): config for config in reversed(configs)}
    return tuple(reversed(by_client.values()))


async def exchange_login_assertion_for_refresh_token(
    assertion: SSOIdentityAssertion,
    targets: tuple[IdJagConfig, ...],
    token_endpoint: TokenExchangeEndpoint,
    now: datetime,
) -> SSOIdentityAssertion:
    """The assertion as captured when no server wants a refresh token or the login already holds one
    from its own SSO client, otherwise the assertion plus the refresh token the first target's IdP
    client exchanged it for."""
    if not targets or assertion.refresh_token is not None:
        return assertion
    config: Final = targets[0]
    if len(targets) > 1:
        verbose_proxy_logger.warning(
            "ID-JAG: %d IdP clients are configured with id_jag_subject_source refresh_token; the login refresh "
            "token is minted for client_id=%s at %s and servers using another client cannot redeem it.",
            len(targets),
            config.client_id,
            config.org_token_endpoint,
        )
    params: Final = {
        "grant_type": _TOKEN_EXCHANGE_GRANT_TYPE,
        "requested_token_type": REFRESH_TOKEN_TOKEN_TYPE,
        "subject_token": assertion.id_token.get_secret_value(),
        "subject_token_type": assertion.subject_token_type or config.subject_token_type,
        "scope": _REFRESH_TOKEN_SCOPE,
    }
    match await token_endpoint.fetch(config.org_token_endpoint, config.client_id, params, config.client_auth):
        case Error(err):
            verbose_proxy_logger.warning(
                "ID-JAG: the IdP at %s refused to exchange the login assertion for a refresh token (%s); the "
                "login goes on with the assertion alone and servers presenting a refresh token will ask the "
                "user to sign in again.",
                config.org_token_endpoint,
                err.summary,
            )
            return assertion
        case Ok(token):
            verbose_proxy_logger.info(
                "ID-JAG: exchanged the login assertion for a refresh token at %s (expires_in=%s)",
                config.org_token_endpoint,
                token.expires_in,
            )
            return with_login_refresh_token(assertion, token.access_token, token.expires_in, now)


async def exchange_login_assertion_if_configured(
    assertion: SSOIdentityAssertion | None,
) -> SSOIdentityAssertion | None:
    """The SSO-callback hook: consults the registered MCP servers and performs the exchange when one of
    them presents a login refresh token. Nothing here can fail the login."""
    if assertion is None:
        return None
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (  # noqa: PLC0415  # avoids import cycle
        global_mcp_server_manager,
    )
    from litellm.proxy._experimental.mcp_server.outbound_credentials.adapter import (  # noqa: PLC0415  # import cycle
        to_server_spec,
    )
    from litellm.proxy._experimental.mcp_server.outbound_credentials.token_endpoint import (  # noqa: PLC0415  # cycle
        TokenEndpointClient,
    )

    try:
        targets: Final = refresh_token_exchange_targets(
            to_server_spec(server) for server in global_mcp_server_manager.get_registry().values()
        )
        return await exchange_login_assertion_for_refresh_token(
            assertion, targets, TokenEndpointClient(), datetime.now(timezone.utc)
        )
    except Exception as exc:  # noqa: BLE001  # the login itself must not fail on an egress-side exchange
        verbose_proxy_logger.warning("ID-JAG: the login-time refresh token exchange failed unexpectedly: %s", exc)
        return assertion
