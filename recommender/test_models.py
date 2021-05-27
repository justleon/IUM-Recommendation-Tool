from models import PopularityBasedRecommender, ContentBasedRecommender
from data_handler import DataHandler
from model_evaluator import ModelEvaluator

if __name__ == "__main__":
    dh = DataHandler()
    popularity_model = PopularityBasedRecommender(dh)
    content_based_model = ContentBasedRecommender(dh)
    model_evaluator = ModelEvaluator(dh)

    print("---POPULARITY BASED MODEL---")
    print(model_evaluator.evaluate(popularity_model))
    print("\n---CONTENT BASED MODEL---")
    print(model_evaluator.evaluate(content_based_model))
