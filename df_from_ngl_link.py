from nglscenes import *
import pandas as pd

# read scene from clipboard, where each layer has one group of neurons
# link containing EPG and PEN neurons in different wedges: https://neuroglancer-demo.appspot.com/#!gs://flyem-user-links/short/2024-06-12.183452.772389.json

sc = Scene.from_clipboard()

dfs = []
for layer in range(len(sc)):
    df = pd.DataFrame({'root_id': sc.layers[layer]['segments']})
    df['name'] = sc.layers[layer]['name']
    dfs.append(df)

df = pd.concat(dfs)
# remove the de-selected ids
df = df[~df.root_id.str.contains('!')]
df
df.to_csv('data/PENandEPG_wedges.csv', index=False)
