import json
import os
import pickle


class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "MostPopular":
            from service.recommenders.most_popular import MostPopular

            return MostPopular
        if name == "UserKnn":
            from service.recommenders.userknn import UserKnn

            return UserKnn
        return super().find_class(module, name)


def load(path: str):
    with open(os.path.join(path), "rb") as f:
        return CustomUnpickler(f).load()


def load_ready_recos(path: str):
    with open(path, "r", encoding="utf8") as file:
        recos = json.load(file)
    return recos
