import pandas as pd

# Load the data
carac = pd.read_csv("data/carac.csv",sep=';')
lieux = pd.read_csv("data/lieux.csv",sep=';')
veh = pd.read_csv("data/veh.csv",sep=';')
vict = pd.read_csv("data/vict.csv",sep=';')

victime = vict.merge(veh,on=['Num_Acc','num_veh'])
accident = carac.merge(lieux,on = 'Num_Acc')
victime = victime.merge(accident,on='Num_Acc')
victime.to_csv("data/merged_data.csv", index=False, sep=';')


print(victime)