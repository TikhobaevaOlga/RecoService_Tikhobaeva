import os
import random
from typing import Dict, List

from dotenv import find_dotenv, load_dotenv
from fastapi import APIRouter, FastAPI, Request, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

from service.api.exceptions import AuthorizationError, ModelNotFoundError, UserNotFoundError
from service.log import app_logger

load_dotenv(find_dotenv())

token_header = APIKeyHeader(name="Authorization")


def get_api_key(api_key_header: str = Security(token_header)) -> str:
    api_keys = os.environ.get("API_KEYS")
    if api_key_header in api_keys:
        return api_key_header
    raise AuthorizationError()


class RecoResponse(BaseModel):
    user_id: int
    items: List[int]


add_reco_responses: Dict = {
    404: {
        "description": "Error: User or model is unknown",
        "content": {
            "application/json": {
                "example": {"errors": [{"error_key": "string", "error_message": "string", "error_loc": "string"}]}
            }
        },
    },
    403: {
        "description": "Error: Invalid or missing API Key",
        "content": {
            "application/json": {
                "example": {"errors": [{"error_key": "string", "error_message": "string", "error_loc": "string"}]}
            }
        },
    },
}

router = APIRouter()


@router.get(
    path="/health",
    tags=["Health"],
)
async def health() -> str:
    return "I am alive"


@router.get(
    path="/reco/{model_name}/{user_id}",
    tags=["Recommendations"],
    response_model=RecoResponse,
    responses=add_reco_responses,
)
async def get_reco(request: Request, model_name: str, user_id: int, token: str = Security(get_api_key)) -> RecoResponse:
    app_logger.info(f"Request for model: {model_name}, user_id: {user_id}")
    k_recs = request.app.state.k_recs

    if user_id > 10**9:
        raise UserNotFoundError(error_message=f"User {user_id} not found")

    if model_name == "top":
        reco = list(range(k_recs))
    elif model_name == "random":
        reco = random.sample(range(1, 1000), k_recs)
    else:
        raise ModelNotFoundError(error_message=f"Model {model_name} not registered")

    return RecoResponse(user_id=user_id, items=reco)


def add_views(app: FastAPI) -> None:
    app.include_router(router)
