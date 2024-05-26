import numpy as np
import pandas as pd
from joblib import load


def calculate_time(position, window):
    model = load('best_model.joblib')
    window_mapping = {'a': 0,  'd': 1}
    window = window_mapping[window]
    sum = 0
    for i in range(1, position + 1):
        model_input = pd.DataFrame({
            'Position': [i],
            'window_encoded': [window]
        })
        time = model.predict(model_input)
        if time<11:
            time=11
        sum+=time
    avg = sum / position
    return avg


calculate_time(4, 'd')
