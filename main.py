from flask import Flask, request
from flask_restful import Api
from service.recommendations import Recommendations

app = Flask(__name__)
api = Api(app)
app.config['JSON_SORT_KEYS'] = False

api.add_resource(Recommendations, '/recommendations')


if __name__ == "__main__":
    app.run(debug=True)
