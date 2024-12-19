import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

gps = pd.read_csv('step3/time_encoding.csv.csv', sep = ",")

X_lat = gps['lat']
X_long = gps['long']

# On définit tous nos points à classifier

X_cluster = np.array((list(zip(X_lat, X_long))))

# Kmeans nous donne pour chaque point la catégorie associée

clustering = KMeans(n_clusters=15, random_state=0)
clustering.fit(X_cluster)

# Enfin on ajoute les catégories dans la base d'entraînement

geo = pd.Series(clustering.labels_)
gps['geo'] = geo

gps.to_csv('step4/geo_encoding.csv', index=False)