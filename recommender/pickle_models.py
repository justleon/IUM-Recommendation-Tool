import pickle

from .config import ROOT, BEST_PARAMS
from .data_handler import DataHandler
from .model_evaluator import ModelEvaluator
from .models import ContentBasedRecommender, PopularityBasedRecommender


def pickle_models():
    popularity_path = 'models/popularity_model.pkl'
    content_based_path = "models/content_based_model.pkl"
    dh = DataHandler()
    popularity_model = PopularityBasedRecommender(dh)
    content_based_model = ContentBasedRecommender(dh)
    model_evaluator = ModelEvaluator(dh)

    content_based_model.train(BEST_PARAMS[0], BEST_PARAMS[1], BEST_PARAMS[2])

    with open(f'{ROOT}/{popularity_path}', 'wb') as popularity_model_file:
        pickle.dump(popularity_model, popularity_model_file)
    popularity_model_file.close()

    with open(f'{ROOT}/{content_based_path}', 'wb') as content_based_model_file:
        pickle.dump(content_based_model, content_based_model_file)
    content_based_model_file.close()


if __name__ == "__main__":
    pickle_models()
