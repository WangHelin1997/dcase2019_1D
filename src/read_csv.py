import pandas as pd

data = pd.read_csv("test.csv")
data['filename']='audio/'+data['filename']+'\t'+data['label']
print(data['filename'])

dataframe = pd.DataFrame({'filename\tscene label': data['filename']})
dataframe.to_csv("WangHL_PKU_task1a_1.output.csv", index=False, sep=',')