import os
from typing import Union, Any

import json
from recommender.config import ROOT
import pandas as pd


def write_to_log(data: Union[pd.DataFrame, Any]):
    if not os.path.isfile(f'{ROOT}\\recommendations_log.jsonl'):
        new_log_file = open(f'{ROOT}\\recommendations_log.jsonl', 'w')
        new_log_file.close()

    with open(f'{ROOT}\\recommendations_log.jsonl', mode='a') as f:
        new_log = json.dumps(data)
        f.write(new_log + "\n")
    f.close()
