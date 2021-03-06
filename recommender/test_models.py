from .config import PARAMETER_LIST
from .models import PopularityBasedRecommender, ContentBasedRecommender
from .data_handler import DataHandler
from .model_evaluator import ModelEvaluator

if __name__ == "__main__":
    dh = DataHandler()
    popularity_model = PopularityBasedRecommender(dh)
    content_based_model = ContentBasedRecommender(dh)
    model_evaluator = ModelEvaluator(dh)

    pop_result = model_evaluator.evaluate(popularity_model, dh.interactions_test_indexed)
    print("---POPULARITY BASED MODEL---")
    print(pop_result)

    curr_max = 0
    best_params = None
    for params in PARAMETER_LIST:
        content_based_model.train(params[0], params[1], params[2])
        result = model_evaluator.evaluate(content_based_model, dh.interactions_val_indexed)
        if (result['rate@5'] + result['rate@10']) > curr_max:
            curr_max = result['rate@5'] + result['rate@10']
            best_params = params
    content_based_model.train(best_params[0], best_params[1], best_params[2])

    cb_result = model_evaluator.evaluate(content_based_model, dh.interactions_test_indexed)
    print("\n---CONTENT BASED MODEL---")
    print(cb_result)
    print({"best_params": best_params})
