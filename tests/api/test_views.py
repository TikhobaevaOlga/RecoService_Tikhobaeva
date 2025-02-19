import os
from http import HTTPStatus

from dotenv import find_dotenv, load_dotenv
from starlette.testclient import TestClient

from service.settings import ServiceConfig

load_dotenv(find_dotenv())

GET_RECO_PATH = "/reco/{model_name}/{user_id}"
MODEL_NAMES = ["popular", "user_knn"]
TOKEN = os.environ.get("API_KEYS")[0]


def test_health(
    client: TestClient,
) -> None:
    with client:
        response = client.get("/health")
    assert response.status_code == HTTPStatus.OK


def test_get_reco_success(
    client: TestClient,
    service_config: ServiceConfig,
) -> None:
    user_id = 123
    for model_name in MODEL_NAMES:
        path = GET_RECO_PATH.format(model_name=model_name, user_id=user_id)
    with client:
        response = client.get(path, headers={"Authorization": TOKEN})
    assert response.status_code == HTTPStatus.OK
    response_json = response.json()
    assert response_json["user_id"] == user_id
    assert len(response_json["items"]) == service_config.k_recs
    assert all(isinstance(item_id, int) for item_id in response_json["items"])


def test_get_reco_for_unknown_user(
    client: TestClient,
) -> None:
    user_id = 10**10
    for model_name in MODEL_NAMES:
        path = GET_RECO_PATH.format(model_name=model_name, user_id=user_id)
    with client:
        response = client.get(path, headers={"Authorization": TOKEN})
    assert response.status_code == HTTPStatus.NOT_FOUND
    assert response.json()["errors"][0]["error_key"] == "user_not_found"


def test_get_reco_for_unknown_model(
    client: TestClient,
) -> None:
    model_name = "unfound_model"
    user_id = 123
    path = GET_RECO_PATH.format(model_name=model_name, user_id=user_id)
    with client:
        response = client.get(path, headers={"Authorization": TOKEN})
    assert response.status_code == HTTPStatus.NOT_FOUND
    assert response.json()["errors"][0]["error_key"] == "model_not_found"


def test_get_reco_for_invalid_token(
    client: TestClient,
) -> None:
    user_id = 123
    for model_name in MODEL_NAMES:
        path = GET_RECO_PATH.format(model_name=model_name, user_id=user_id)
    with client:
        response = client.get(path, headers={"Authorization": "abc"})
    assert response.status_code == HTTPStatus.FORBIDDEN
    assert response.json()["errors"][0]["error_key"] == "invalid_api_key"


def test_get_reco_for_missing_token(
    client: TestClient,
) -> None:
    user_id = 123
    for model_name in MODEL_NAMES:
        path = GET_RECO_PATH.format(model_name=model_name, user_id=user_id)
    with client:
        response = client.get(path)
    assert response.status_code == HTTPStatus.FORBIDDEN
    assert response.json()["errors"][0]["error_key"] == "http_exception"
