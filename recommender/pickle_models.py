import pickle

from config import ROOT, PARAMETER_LIST
from data_handler import DataHandler
from model_evaluator import ModelEvaluator
from models import ContentBasedRecommender, PopularityBasedRecommender


def pickle_models():
    popularity_path = 'models\\popularity_model.pkl'
    content_based_path = "models\\content_based_model.pkl"
    dh = DataHandler()
    popularity_model = PopularityBasedRecommender(dh)
    content_based_model = ContentBasedRecommender(dh)
    model_evaluator = ModelEvaluator(dh)

    curr_max = 0
    best_params = None
    for params in PARAMETER_LIST:
        content_based_model.train(params[0], params[1], params[2])
        result = model_evaluator.evaluate(content_based_model, dh.interactions_val_indexed)
        if (result['rate@5'] + result['rate@10']) > curr_max:
            curr_max = result['rate@5'] + result['rate@10']
            best_params = params

    content_based_model.train(best_params[0], best_params[1], best_params[2])

    with open(f'{ROOT}\\{popularity_path}', 'wb') as popularity_model_file:
        pickle.dump(popularity_model, popularity_model_file)
    popularity_model_file.close()

    with open(f'{ROOT}\\{content_based_path}', 'wb') as content_based_model_file:
        pickle.dump(content_based_model, content_based_model_file)
    content_based_model_file.close()


if __name__ == "__main__":
    pickle_models()
