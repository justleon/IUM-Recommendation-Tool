from flask import Flask
from flask_restful import Api
from service.recommendations import Recommendations

from recommender.models import PopularityBasedRecommender, ContentBasedRecommender
from recommender.data_handler import DataHandler

app = Flask(__name__)
api = Api(app)

api.add_resource(Recommendations, "/models", '/models/user/<int:user_id>')

if __name__ == "__main__":
    app.run(debug=True)
