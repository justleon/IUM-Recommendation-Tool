from typing import Union, Any

import json
from config import ROOT
import pandas as pd


def write_to_log(data: Union[pd.DataFrame, Any]):
    with open(f'{ROOT}/recommendations_log.json', mode='r+') as f:
        log = json.load(f)
        recommendations = log["recommendations"]
        recommendations.append(data)
        new_log = {
            "recommendations": recommendations
        }

        f.seek(0)
        json.dump(new_log, f)
    f.close()
