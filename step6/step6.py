import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# On commence par normaliser les données :

X_train = pd.read_csv('step5/one_hot_encoding.csv', sep = ",")

# On commence par normaliser les données :

X_train = normalize(X_train.values)

# On divise la base en bases d'entraînements et de test :

X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_train,y)

# On construit le modèle :

model_rf = RandomForestClassifier(n_estimators=100, 
                                  max_depth=8
)

# L'entrînement commence :

model_rf.fit(X_train_rf, y_train_rf)

# On a maintenant les prédictions pour la base de test

predictions_test = model_rf.predict(X_test_rf)

# On calcul de même les prédictions pour la base train

predictions_train = model_rf1.predict(X_trainrf)

# Les résultats sont calculés de cette manière :

train_acc = accuracy_score(y_trainrf, predictions_train)
print(train_acc)

test_acc = accuracy_score(y_testrf, predictions_test)
print(test_acc)