from datetime import datetime
from flask_restful import Resource

from get_user_model import get_model
from logging import write_to_log


class Recommendations(Resource):
    def get(self, user_id):
        model = get_model(user_id)
        date = datetime.now().strftime('%d-%m-%Y %H:%M:%S')
        products = model.predict(user_id)

        recommendation = {
            "date": date,
            "user_id": user_id,
            "variant": user_id % 2,
            "products": products
        }
        write_to_log(recommendation)
        return recommendation
