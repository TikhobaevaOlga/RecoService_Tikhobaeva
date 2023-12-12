import os
import pickle

from service.recommenders.model_loader import load_ready_recos

MODELS_PATH = "service/models"
RECOS_PATH = "service/recommendations"

if os.path.exists(MODELS_PATH):
    lightfm_ann_model = pickle.load(open(f"{MODELS_PATH}/ann_lightfm_warp_64.pkl", "rb"))
lightfm_recos = load_ready_recos(f"{RECOS_PATH}/lightfm_recos.json")


def online_ann_recommend(user_id, N_recs: int = 10):
    try:
        lightfm_recs = lightfm_ann_model.get_item_list_for_user(user_id, top_n=N_recs).tolist()
        return lightfm_recs
    except KeyError:
        return None


def offline_fm_recommend(user_id):
    try:
        lightfm_recs = lightfm_recos[str(user_id)]
        return lightfm_recs
    except KeyError:
        return None
