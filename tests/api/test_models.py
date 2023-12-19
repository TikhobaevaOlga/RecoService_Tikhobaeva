import os

from service.recommenders.lightfm import online_ann_recommend
from service.recommenders.model_loader import load

MODELS_PATH = "service/models"


def test_popular() -> None:
    if os.path.exists(MODELS_PATH):
        k_recs = 10
        popular_model = load(f"{MODELS_PATH}/popular.pkl")
        reco = popular_model.online_recommend(N_recs=k_recs)
        assert isinstance(reco, list)
        assert len(reco) == k_recs
    else:
        pass


def test_user_knn() -> None:
    if os.path.exists(MODELS_PATH):
        k_recs = 10
        user_id = 123
        userknn_model = load(f"{MODELS_PATH}/user_knn.pkl")
        reco = userknn_model.online_recommend(user_id, N_recs=k_recs)
        assert isinstance(reco, list)
    else:
        pass


def test_lightfm() -> None:
    if os.path.exists(MODELS_PATH):
        k_recs = 10
        user_id = 123
        reco = online_ann_recommend(user_id, N_recs=k_recs)
        assert isinstance(reco, list)
    else:
        pass
