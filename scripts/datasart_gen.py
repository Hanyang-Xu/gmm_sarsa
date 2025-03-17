import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib import lines

prepath = "Datasets"
subjects = ['AB01', 'AB02', 'AB03', 'AB04', 'AB05', 'AB06', 'AB07', 'AB08', 'AB09', 'AB10']  
Height = ['h0', 'h75', 'h150', 'h225', 'h300']  
color_dict = {'h0': 'grey', 'h75': 'orange', 'h150': 'dodgerblue', 'h225': 'lime', 'h300': 'violet'}
Cate = ['Left', 'Right']

knee_a = np.empty((101, 0)) 
knee_m = np.empty((101, 0))
ankle_a = np.empty((101, 0))
ankle_m = np.empty((101, 0))

for cate in Cate:
    for sub in subjects:
        file_prepath = os.path.join(prepath, str(sub))
        file_path = os.path.join(file_prepath, str(cate) + ".xlsx")
        
        ankle_angle = pd.read_excel(file_path, sheet_name='ankle_angle').to_numpy()
        ankle_moment = pd.read_excel(file_path, sheet_name='ankle_moment').to_numpy()
        knee_angle = pd.read_excel(file_path, sheet_name='knee_angle').to_numpy()
        knee_moment = pd.read_excel(file_path, sheet_name='knee_moment').to_numpy()
        
        knee_a = np.concatenate((knee_a, knee_angle), axis=1)
        knee_m = np.concatenate((knee_m, knee_moment), axis=1)
        ankle_a = np.concatenate((ankle_a, ankle_angle), axis=1)
        ankle_m = np.concatenate((ankle_m, ankle_moment), axis=1)

print("Knee Angle Shape:", knee_a.shape)
print("Knee Moment Shape:", knee_m.shape)
print("Ankle Angle Shape:", ankle_a.shape)
print("Ankle Moment Shape:", ankle_m.shape)

save_path = "Datasets/merged_data.xlsx"
with pd.ExcelWriter(save_path) as writer:
    pd.DataFrame(knee_a).to_excel(writer, sheet_name='knee_angle', index=False)
    pd.DataFrame(knee_m).to_excel(writer, sheet_name='knee_moment', index=False)
    pd.DataFrame(ankle_a).to_excel(writer, sheet_name='ankle_angle', index=False)
    pd.DataFrame(ankle_m).to_excel(writer, sheet_name='ankle_moment', index=False)
