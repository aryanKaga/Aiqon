import pandas as pd
import numpy as np

shipments = pd.read_csv('./data/shipments.csv')
shipments = shipments.drop(columns = ['required_on_time_prob'])
shipments.to_csv('./data/shipments.csv')