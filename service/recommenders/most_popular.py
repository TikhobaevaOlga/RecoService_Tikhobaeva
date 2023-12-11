import pandas as pd
from rectools.dataset import Dataset
from rectools.models import PopularModel


class MostPopular:
    """Class for fit-perdict Popular model
    based on PopularModel model from rectools.models
    """

    def __init__(self):
        self.model = PopularModel()
        self.train = None
        self.is_fitted = False

    def get_mappings(self, train: pd.DataFrame):
        self.users_inv_mapping = dict(enumerate(train["user_id"].unique()))
        self.users_mapping = {v: k for k, v in self.users_inv_mapping.items()}

        self.items_inv_mapping = dict(enumerate(train["item_id"].unique()))
        self.items_mapping = {v: k for k, v in self.items_inv_mapping.items()}

    def fit(self, train: pd.DataFrame):
        self.train = train
        self.get_mappings(self.train)
        dataset = Dataset.construct(self.train)
        self.model.fit(dataset)
        self.is_fitted = True

    def online_recommend(self, N_recs: int = 10):
        if not self.is_fitted:
            raise ValueError("Please call fit before get recommendations")
        popular_recs = self.model.popularity_list[0][:N_recs]
        popular_recs = [self.items_inv_mapping[rec] for rec in popular_recs]
        return popular_recs

    def offline_recommend(self, ready_recos):
        popular_recs = ready_recos["popular_recos"]
        return popular_recs
