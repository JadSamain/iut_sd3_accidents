import pandas as pd
import numpy as np

hot = pd.read_csv('step4/geo_encoding.csv', sep = ",")

y = hot['grav']

features = ['catu','sexe','trajet','secu',
            'catv','an_nais','mois',
            'occutc','obs','obsm','choc','manv',
            'lum','agg','int','atm','col','gps',
            'catr','circ','vosp','prof','plan',
            'surf','infra','situ','hrmn','geo']

X_train_data = pd.get_dummies(hot[features].astype(str))

X_train_data.to_csv('step5/one_hot_encoding.csv', index=False)
