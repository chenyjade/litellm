"""The login-time exchange of a captured identity assertion for an IdP refresh token."""

import time
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import SecretStr

from litellm.proxy._experimental.mcp_server.outbound_credentials.login_assertion_exchange import (
    exchange_login_assertion_for_refresh_token,
    refresh_token_exchange_targets,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.result import Error, Ok, Result
from litellm.proxy._experimental.mcp_server.outbound_credentials.sso_assertion_store import (
    SAML2_SUBJECT_TOKEN_TYPE,
    SSOIdentityAssertion,
    assertion_from_saml_login,
    login_refresh_token_expired,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.token_endpoint import ExchangedToken
from litellm.proxy._experimental.mcp_server.outbound_credentials.types import (
    ClientAuth,
    ClientSecretAuth,
    CredError,
    IdJagConfig,
    IdJagResourceClient,
    IdJagSubjectSource,
    NoneConfig,
    ServerSpec,
)

_NOW = datetime(2026, 10, 16, 12, 0, tzinfo=timezone.utc)


def _config(
    *, source: IdJagSubjectSource = IdJagSubjectSource.refresh_token, client_id: str = "agent", endpoint: str = ""
) -> IdJagConfig:
    return IdJagConfig(
        org_token_endpoint=endpoint or "https://idp.example.com/token",
        resource_token_endpoint="https://mcp-as.example.com/token",
        client_id=client_id,
        client_auth=ClientSecretAuth(client_secret=SecretStr("s")),
        resource_client=IdJagResourceClient(
            client_id=client_id, client_auth=ClientSecretAuth(client_secret=SecretStr("s"))
        ),
        subject_source=source,
    )


def _spec(config: IdJagConfig | NoneConfig, server_id: str = "srv") -> ServerSpec:
    return ServerSpec(server_id=server_id, resource="https://mcp.example.com", config=config)


def _saml() -> SSOIdentityAssertion:
    return assertion_from_saml_login(b"<saml:Assertion/>", int(time.time()) + 300, "https://idp.example.com")


class _Endpoint:
    def __init__(self, result: Result[ExchangedToken, CredError]) -> None:
        self._result = result
        self.calls: list[tuple[str, str, dict[str, str], ClientAuth]] = []

    async def fetch(
        self, endpoint: str, client_id: str, grant_params: dict[str, str], client_auth: ClientAuth
    ) -> Result[ExchangedToken, CredError]:
        self.calls.append((endpoint, client_id, dict(grant_params), client_auth))
        return self._result


def test_targets_keep_only_refresh_token_sourced_id_jag_servers_one_per_idp_client():
    direct = _config(source=IdJagSubjectSource.login_assertion)
    first = _config(client_id="agent")
    twin = _config(client_id="agent")
    other = _config(client_id="other-agent")

    targets = refresh_token_exchange_targets(
        [_spec(direct, "a"), None, _spec(NoneConfig(), "b"), _spec(first, "c"), _spec(twin, "d"), _spec(other, "e")]
    )

    assert targets == (first, other)


@pytest.mark.asyncio
async def test_exchange_presents_the_captured_assertion_and_keeps_the_refresh_token():
    endpoint = _Endpoint(Ok(ExchangedToken(access_token="rt-1", expires_in=3600)))
    config = _config()
    saml = _saml()

    exchanged = await exchange_login_assertion_for_refresh_token(saml, (config,), endpoint, _NOW)

    ((url, client_id, params, auth),) = endpoint.calls
    assert (url, client_id, auth) == (config.org_token_endpoint, config.client_id, config.client_auth)
    assert params == {
        "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
        "requested_token_type": "urn:ietf:params:oauth:token-type:refresh_token",
        "subject_token": saml.id_token.get_secret_value(),
        "subject_token_type": SAML2_SUBJECT_TOKEN_TYPE,
        "scope": "openid offline_access",
    }
    assert exchanged.refresh_token is not None
    assert exchanged.refresh_token.get_secret_value() == "rt-1"
    assert exchanged.refresh_token_expires_at == _NOW + timedelta(seconds=3600)
    assert exchanged.id_token == saml.id_token
    assert exchanged.expires_at == saml.expires_at
    assert exchanged.subject_token_type == SAML2_SUBJECT_TOKEN_TYPE
    assert not login_refresh_token_expired(exchanged, _NOW + timedelta(seconds=3599))
    assert login_refresh_token_expired(exchanged, _NOW + timedelta(seconds=3600))


@pytest.mark.asyncio
async def test_exchange_without_a_lifetime_leaves_expiry_to_the_idp():
    endpoint = _Endpoint(Ok(ExchangedToken(access_token="rt-1", expires_in=None)))

    exchanged = await exchange_login_assertion_for_refresh_token(_saml(), (_config(),), endpoint, _NOW)

    assert exchanged.refresh_token_expires_at is None
    assert not login_refresh_token_expired(exchanged, _NOW + timedelta(days=365))


@pytest.mark.asyncio
async def test_a_refused_exchange_keeps_the_plain_assertion(caplog: pytest.LogCaptureFixture):
    endpoint = _Endpoint(Error(CredError.of_upstream_unavailable("token exchange failed with status 400")))
    saml = _saml()

    with caplog.at_level("WARNING"):
        exchanged = await exchange_login_assertion_for_refresh_token(saml, (_config(),), endpoint, _NOW)

    assert exchanged == saml
    assert exchanged.refresh_token is None
    assert any("refused to exchange the login assertion" in message for message in caplog.messages)


@pytest.mark.asyncio
async def test_no_exchange_without_a_refresh_token_sourced_server():
    endpoint = _Endpoint(Ok(ExchangedToken(access_token="rt-1", expires_in=3600)))
    saml = _saml()

    assert await exchange_login_assertion_for_refresh_token(saml, (), endpoint, _NOW) == saml
    assert endpoint.calls == []


@pytest.mark.asyncio
async def test_a_login_that_already_holds_a_refresh_token_is_not_exchanged_again():
    endpoint = _Endpoint(Ok(ExchangedToken(access_token="rt-1", expires_in=3600)))
    oidc = SSOIdentityAssertion(id_token=SecretStr("id-token"), refresh_token=SecretStr("sso-rt"))

    assert await exchange_login_assertion_for_refresh_token(oidc, (_config(),), endpoint, _NOW) == oidc
    assert endpoint.calls == []


@pytest.mark.asyncio
async def test_several_idp_clients_mint_for_the_first_and_warn(caplog: pytest.LogCaptureFixture):
    endpoint = _Endpoint(Ok(ExchangedToken(access_token="rt-1", expires_in=3600)))
    first, second = _config(client_id="agent"), _config(client_id="other-agent")

    with caplog.at_level("WARNING"):
        exchanged = await exchange_login_assertion_for_refresh_token(_saml(), (first, second), endpoint, _NOW)

    assert [call[1] for call in endpoint.calls] == ["agent"]
    assert exchanged.refresh_token is not None
    assert any("2 IdP clients" in message and "client_id=agent" in message for message in caplog.messages)
