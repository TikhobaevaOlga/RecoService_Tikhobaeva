import os
from typing import Dict, List

from dotenv import find_dotenv, load_dotenv
from fastapi import APIRouter, FastAPI, Request, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

from service.api.exceptions import AuthorizationError, ModelNotFoundError, UserNotFoundError
from service.log import app_logger
from service.recommenders.lightfm import offline_fm_recommend, online_ann_recommend
from service.recommenders.model_loader import load_ready_recos
from service.recommenders.most_popular import MostPopular
from service.recommenders.neural_networks import offline_neural_recommend
from service.recommenders.userknn import UserKnn

load_dotenv(find_dotenv())

AVAILABLE_MODELS = ["popular", "user_knn", "lightfm_warp_64", "lightfm_ann", "dssm", "autoencoder", "rec_vae"]

RECOS_PATH = "service/recommendations"

userknn_model = UserKnn()
popular_model = MostPopular()

userknn_recos = load_ready_recos(f"{RECOS_PATH}/user_knn_recos.json")
popular_recos = load_ready_recos(f"{RECOS_PATH}/popular_recos.json")

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

    if model_name not in AVAILABLE_MODELS:
        raise ModelNotFoundError(error_message=f"Model {model_name} not registered")

    if user_id > 10**9:
        raise UserNotFoundError(error_message=f"User {user_id} not found")

    if model_name == "user_knn":
        reco = userknn_model.offline_recommend(userknn_recos, user_id)
    elif model_name == "popular":
        reco = popular_model.offline_recommend(popular_recos)
    elif model_name == "lightfm_warp_64":
        reco = offline_fm_recommend(user_id)
    elif model_name == "lightfm_ann":
        reco = online_ann_recommend(user_id)
    elif model_name in ["dssm", "autoencoder", "rec_vae"]:
        reco = offline_neural_recommend(model_name, user_id)

    if not reco:
        reco = popular_model.offline_recommend(popular_recos)

    if len(reco) < k_recs:
        popular_recs = [rec for rec in popular_model.offline_recommend(popular_recos) if rec not in reco]
        reco += popular_recs[: (k_recs - len(reco))]

    return RecoResponse(user_id=user_id, items=reco)


def add_views(app: FastAPI) -> None:
    app.include_router(router)
