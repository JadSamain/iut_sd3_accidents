import pandas as pd
import numpy as np
from sklearn.cluster import KMeans



victime = pd.read_csv('step3/missing_values_deleted.csv')

hrmn=pd.cut(victime['hrmn'],24,labels=[str(i) for i in range(0,24)])

victime['hrmn']=hrmn.values

# On extrait du tableau la latitude et la longitude

X_lat = victime['lat']
X_long = victime['long']

# On définit tous nos points à classifier

X_cluster = np.array((list(zip(X_lat, X_long))))

# Kmeans nous donne pour chaque point la catégorie associée

clustering = KMeans(n_clusters=15, random_state=0)
clustering.fit(X_cluster)

# Enfin on ajoute les catégories dans la base d'entraînement

geo = pd.Series(clustering.labels_)
victime['geo'] = geo

victime.to_csv('time_encoding.csv.csv', index=False)