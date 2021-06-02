import os
ROOT = os.path.dirname(os.path.abspath(__file__))

SEED = 1
TRAIN_SIZE = 0.2
TEST_SIZE = 0.2
VAL_SIZE = 0.6
VIEW_PRODUCT_STRENGTH = 1
BUY_PRODUCT_STRENGTH = 3
PARAMETER_LIST = [(0, 0.5, (1, 1)), (0, 0.5, (1, 2)), (0, 0.5, (1, 3))]

EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS = 100
