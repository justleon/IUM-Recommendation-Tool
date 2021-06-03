import pickle

popularity_path = 'models/popularity_model.pkl'
content_based_path = "models/content_based_model.pkl"


def get_model(model_id: int):
    models = {
        0: popularity_path,
        1: content_based_path
    }
    model_file = open(models[model_id], 'rb')
    model = pickle.load(model_file)
    model_file.close()

    return model
