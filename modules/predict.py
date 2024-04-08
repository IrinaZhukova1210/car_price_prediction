
from datetime import datetime
import json
import logging
import os

import dill
import pandas as pd

path = os.environ.get('PROJECT_PATH', '..')


def model_last():
    model_last = os.listdir(f'{path}/data/models/')[-1]

    with open(f'{path}/data/models/{model_last}', 'rb') as file:
        model = dill.load(file)

    return model


def get_predictions(model) -> pd.DataFrame:
    test_cars = os.listdir(f'{path}/data/test/')
    results = []

    for car in test_cars:
        with open(f'{path}/data/test/{car}') as fin:
            form = json.load(fin)
        df = pd.DataFrame.from_dict([form])
        y = model.predict(df)
        results.append({'car_id': form["id"], 'pred': y[0]})
        logging.info(f'id: {form["id"]}, pred: {y[0]}, price: {form["price"]}')

    return pd.DataFrame(results)


def predict() -> None:
    model = model_last()
    predictions = get_predictions(model)
    predictions.to_csv(f'{path}/data/predictions/preds_{datetime.now().strftime("%Y%m%d%H%M")}.csv', sep=',',
                       index=False)


if __name__ == '__main__':
    predict()
