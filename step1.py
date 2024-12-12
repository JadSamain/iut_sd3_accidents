import pandas as pd

# Load the data
data_carac = pd.read_csv('data/carac.csv', sep=';')
data_lieux = pd.read_csv('data/lieux.csv', sep=';')
data_veh = pd.read_csv('data/veh.csv', sep=';')
data_vict = pd.read_csv('data/vict.csv', sep=';')

# Merge the data
data = pd.merge(data_carac, data_lieux, on = 'Num_Acc')
data2 = pd.merge(data, data_veh, on = 'Num_Acc')
data3 = pd.merge(data2, data_vict, on = 'Num_Acc')

# Save the data
data3.to_csv('data/merged_data.csv', index=False)