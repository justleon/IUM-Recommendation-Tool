import pickle

popularity_path = 'models\\popularity_model.pkl'
content_based_path = "models\\content_based_model.pkl"


def get_model(user_id: int):
    if user_id % 2 == 0:
        model_file = open(popularity_path, 'rb')
    else:
        model_file = open(content_based_path, 'rb')

    model = pickle.load(model_file)
    model_file.close()

    return model
