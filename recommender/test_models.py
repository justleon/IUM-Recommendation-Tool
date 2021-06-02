from models import PopularityBasedRecommender, ContentBasedRecommender
from data_handler import DataHandler
from model_evaluator import ModelEvaluator

PARAMETER_LIST = [(0, 0.5, (1, 1)), (0, 0.5, (1, 2)), (0, 0.5, (1, 3))]

if __name__ == "__main__":
    dh = DataHandler()
    popularity_model = PopularityBasedRecommender(dh)
    content_based_model = ContentBasedRecommender(dh)
    model_evaluator = ModelEvaluator(dh)

    print("---POPULARITY BASED MODEL---")
    print(model_evaluator.evaluate(popularity_model, dh.interactions_test_indexed))

    curr_max = 0
    best_params = None
    for params in PARAMETER_LIST:
        content_based_model.train(params[0], params[1], params[2])
        result = model_evaluator.evaluate(content_based_model, dh.interactions_val_indexed)
        if (result['rate@5'] + result['rate@10']) > curr_max:
            curr_max = result['rate@5'] + result['rate@10']
            best_params = params

    content_based_model.train(best_params[0], best_params[1], best_params[2])
    print("\n---CONTENT BASED MODEL---")
    print(model_evaluator.evaluate(content_based_model, dh.interactions_test_indexed))
