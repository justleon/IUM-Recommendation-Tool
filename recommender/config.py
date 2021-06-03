import itertools
import os

ROOT = os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir)))

SEED = 1
TEST_SIZE = 0.2
VAL_SIZE = 0.2
TRAIN_SIZE = 1 - TEST_SIZE - VAL_SIZE
VIEW_PRODUCT_STRENGTH = 1
BUY_PRODUCT_STRENGTH = 3
MAX_DF_VALUES = [0]
MIN_DF_VALUES = [0.1, 0.2, 0.4, 0.7, 1.0]
NGRAM_RANGE_VALUES = [(1, 3), (1, 5)]
PARAMETERS = [MAX_DF_VALUES, MIN_DF_VALUES, NGRAM_RANGE_VALUES]
PARAMETER_LIST = [(x, y, z) for x, y, z in itertools.product(*PARAMETERS) if x < y]
BEST_PARAMS = (0, 0.6, (1, 5))

EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS = 100
