import asyncio
import hashlib
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import pytest
from integration._support.client import Gateway, eventually, gateway_from_environment, object_value, string_value
from integration._support.database import read_rows
from integration._support.process import graceful_stop_seconds, owned_proxy
from integration._support.responses_stream import RATE_LIMIT_MESSAGE, openai_error_event
from integration._support.responses_ws_peer import (
    PROVIDER_MODEL,
    USAGE,
    completed,
    drain,
    events,
    responses_peer,
    session,
)
from integration._support.tls import write_self_signed_cert
from pydantic import JsonValue

pytestmark: Final = pytest.mark.timeout(2 * graceful_stop_seconds() + 120)


def _top_level_rate_limit(response_id: str) -> tuple[dict[str, JsonValue], ...]:
    created: Final = events(response_id, "")[0]
    return (created, dict(openai_error_event("rate_limit_exceeded", RATE_LIMIT_MESSAGE)))


@pytest.fixture(scope="module")
def cert(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    return write_self_signed_cert(tmp_path_factory.mktemp("responses-ws-cert"))


@pytest.fixture(scope="module")
def candidate(tmp_path_factory: pytest.TempPathFactory, cert: tuple[Path, Path]) -> Iterator[Gateway]:
    with gateway_from_environment() as base:
        with owned_proxy(base, tmp_path_factory.mktemp("responses-ws"), {"SSL_CERT_FILE": str(cert[0])}) as proxy:
            yield proxy


def test_a_response_create_frame_streams_from_the_provider_socket_back_to_the_client(
    candidate: Gateway, cert: tuple[Path, Path]
) -> None:
    with responses_peer(cert) as peer, candidate.scenario() as scenario:
        model: Final = scenario.model(model=f"openai/{PROVIDER_MODEL}", api_base=peer.url)
        key: Final = scenario.key(models=[model])
        text: Final = f"say seven {uuid.uuid4().hex}"
        (events,) = asyncio.run(session(str(candidate.client.base_url), key, model, (text,)))
        assert [event["type"] for event in events] == [
            "response.created",
            "response.output_text.delta",
            "response.completed",
        ]
        response: Final = completed(events)
        assert response["status"] == "completed"
        assert response["usage"] == USAGE
        assert peer.paths.get_nowait() == f"/v1/responses?model={PROVIDER_MODEL}"
        (forwarded,) = drain(peer.frames)
        assert forwarded["type"] == "response.create"
        assert forwarded["model"] == PROVIDER_MODEL
        assert forwarded["input"][0]["content"][0]["text"] == text


def test_previous_response_id_from_the_first_turn_reaches_the_provider_as_its_own_id(
    candidate: Gateway, cert: tuple[Path, Path]
) -> None:
    with responses_peer(cert) as peer, candidate.scenario() as scenario:
        model: Final = scenario.model(model=f"openai/{PROVIDER_MODEL}", api_base=peer.url)
        key: Final = scenario.key(models=[model])
        texts: Final = (f"remember seven {uuid.uuid4().hex}", f"which number {uuid.uuid4().hex}")
        first, second = asyncio.run(session(str(candidate.client.base_url), key, model, texts))
        assert completed(first)["status"] == "completed"
        assert str(completed(first)["id"]).startswith("resp_") and completed(first)["id"] != "resp_peer_1"
        assert completed(second)["status"] == "completed"
        assert peer.paths.qsize() == 1
        forwarded: Final = drain(peer.frames)
        assert [frame["input"][0]["content"][0]["text"] for frame in forwarded] == list(texts)
        assert "previous_response_id" not in forwarded[0]
        assert forwarded[1]["previous_response_id"] == "resp_peer_1"


def test_a_top_level_error_frame_reaches_the_client_and_logs_the_provider_code_and_message(
    candidate: Gateway, cert: tuple[Path, Path]
) -> None:
    with responses_peer(cert, _top_level_rate_limit) as peer, candidate.scenario() as scenario:
        model: Final = scenario.model(model=f"openai/{PROVIDER_MODEL}", api_base=peer.url)
        key: Final = scenario.key(models=[model])
        text: Final = f"say seven {uuid.uuid4().hex}"
        (events,) = asyncio.run(session(str(candidate.client.base_url), key, model, (text,)))
        assert [event["type"] for event in events] == ["response.created", "error"], events
        assert (events[-1]["code"], events[-1]["message"]) == ("rate_limit_exceeded", RATE_LIMIT_MESSAGE), events
        (forwarded,) = drain(peer.frames)
        assert forwarded["input"][0]["content"][0]["text"] == text
        (row,) = eventually(
            lambda: read_rows(
                'SELECT status, metadata FROM "LiteLLM_SpendLogs" WHERE api_key = %s',
                (hashlib.sha256(key.encode()).hexdigest(),),
            ),
            lambda rows: len(rows) >= 1,
            seconds=70,
        )
        assert row["status"] == "failure", row
        error: Final = object_value(object_value(row["metadata"])["error_information"])
        assert str(error["error_code"]) == "429", error
        assert RATE_LIMIT_MESSAGE in string_value(error["error_message"]), error
