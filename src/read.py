
# import torch
# model = torch.load('14800_iterations.pth', map_location='cpu')
# print(model['model'])

import pandas as pd

# 任意的多组列表
a = ['da', 2, 3]
b = [4, 5, 6]

# 字典中的key值即为csv中列名
dataframe = pd.DataFrame({'a_name': a, 'b_name': b})

# 将DataFrame存储为csv,index表示是否显示行名，default=True
dataframe.to_csv("test.csv", index=False, sep=',')