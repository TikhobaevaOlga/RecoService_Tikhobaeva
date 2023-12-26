from service.recommenders.model_loader import load_ready_recos

RECOS_PATH = "service/recommendations"

dssm_recos = load_ready_recos(f"{RECOS_PATH}/dssm_recos.json")
autoencoder_recos = load_ready_recos(f"{RECOS_PATH}/autoencoder_recos.json")
rec_vae_recos = load_ready_recos(f"{RECOS_PATH}/rec_vae_recos.json")

models_recos = {"dssm": dssm_recos, "autoencoder": autoencoder_recos, "rec_vae": rec_vae_recos}


def offline_neural_recommend(model, user_id):
    try:
        recs = models_recos[model][str(user_id)]
        return recs
    except KeyError:
        return None
