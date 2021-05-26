from models import PopularityBasedRecommender, ContentBasedRecommender
from data_handler import DataHandler

if __name__ == "__main__":
    dh = DataHandler()
    popularity_model = PopularityBasedRecommender(dh)
    content_based_model = ContentBasedRecommender(dh)

    predict_popularity = popularity_model.predict(102)
    predict_content = content_based_model.predict(102)

    print("---POPULARITY BASED MODEL---")
    print(predict_popularity)
    print("\n---CONTENT BASED MODEL---")
    print(predict_content)
