from datetime import datetime
from flask_restful import Resource
from flask import request, jsonify

from service.get_model import get_model
from service.logging import write_to_log


class Recommendations(Resource):
    def get(self):
        code: int = 200
        errors: list[str] = []
        user_ids: list = []

        if 'user' in request.args:
            u = request.args.get('user', type=int)
            if u is None:
                code = 400
                errors.append('User parameter has to be an integer')
            elif u < 102 or u > 301:
                if code == 200:
                    code = 404
                errors.append('This user does not exist')

        if 'model' in request.args:
            m = request.args.get('model', type=int)
            if m is None:
                code = 400
                errors.append('Model parameter has to be an integer')
            elif m != 1 and m != 2:
                if code == 200:
                    code = 404
                errors.append('This model does not exist')

        if 'head' in request.args:
            h = request.args.get('head', type=int)
            if h is None:
                code = 400
                errors.append('Head parameter has to be an integer')

        if code != 200:
            message = {
                "status": "error",
                "parameters": [request.args] if len(request.args) != 0 else [],
                "errors": errors,
                "code": code,
                "data": []
            }
            return jsonify(message)

        if 'user' in request.args:
            user_ids.append(request.args.get('user', type=int))
        else:
            user_ids = list(range(102, 302))

        model_ids: list = []
        if 'model' in request.args:
            model_ids.append(request.args.get('model', type=int))
        else:
            model_ids = list(range(1, 3))

        models: dict = {
            1: get_model(1),
            2: get_model(2)
        }
        for model_id in model_ids:
            models[model_id] = get_model(model_id)

        predictions: list = []
        for model_id in model_ids:
            model_predictions = []
            for user_id in user_ids:
                user_predictions = models[model_id].predict(user_id)
                if len(user_predictions) != 0:
                    user_predictions = user_predictions['product_id']
                    if 'head' in request.args:
                        user_predictions = user_predictions.head(request.args.get('head', type=int))
                    user_data = {
                        "user_id": user_id,
                        "user_recommendations": list(user_predictions) if type(user_predictions) != list else []
                    }
                    model_predictions.append(user_data)

            model_data = {
                "model_id": model_id,
                "model_recommendations": model_predictions
            }
            predictions.append(model_data)

        message = {
            "status": "success",
            "parameters": [request.args] if len(request.args) != 0 else [],
            "errors": errors,
            "code": code,
            "data": {
                "timestamp": datetime.now().strftime('%d-%m-%Y %H:%M:%S'),
                "recommendations": predictions
            }
        }
        # write_to_log(recommendation)
        return jsonify(message)
