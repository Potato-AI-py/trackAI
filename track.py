import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import os, copy, zipfile, re
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV


DATA_PATH = ""
if not os.path.exists("./track3_train.parquet"):
    with zipfile.ZipFile(os.path.join(DATA_PATH, "track3_train.zip"), 'r') as zfile:
        file_paths = zfile.namelist()
        wind_sample_list = list(filter(lambda x: re.match("track_3t/wind/.*csv", x)!=None, file_paths))
        feat_sample_list = list(filter(lambda x: re.match("track_3t/features/.*csv", x)!=None, file_paths))
        wind_sample_list.sort()
        feat_sample_list.sort()
        print('wind csv len', len(wind_sample_list))
        print('feat csv len', len(feat_sample_list))
        wind_data = []
        for item_wind_name in tqdm(wind_sample_list, desc='wind'):
            with zfile.open(item_wind_name, "r") as item:
                item_data = pd.read_csv(item)
                wind_data.append(item_data)

        feat_data = []
        for item_feat_name in tqdm(feat_sample_list, desc='feat'):
            with zfile.open(item_feat_name, "r") as item:
                item_data = pd.read_csv(item)
                feat_data.append(item_data)
    wind_data = pd.concat(wind_data, axis=0)
    feat_data = pd.concat(feat_data, axis=0)
    data = pd.merge(wind_data, feat_data, on=['ID'], how='outer')
    data.to_parquet(os.path.join(DATA_PATH, 'track3_train.parquet'), index=False)

train_data = pd.read_parquet(os.path.join(DATA_PATH, 'track3_train.parquet'))
test_data = pd.read_parquet(os.path.join(DATA_PATH, 'track3_a.parquet'))
test_data = test_data.drop(['ID'], axis=1)

x = train_data.drop(['ID', 'wdir_2min', 'spd_2min', 'spd_inst_max'], axis=1)
# y = train_data.loc[:,['wdir_2min', 'spd_2min', 'spd_inst_max']]
ylist = [train_data['wdir_2min'], train_data['spd_2min'], train_data['spd_inst_max']]


for index,y in enumerate(ylist):
#     svr = GridSearchCV(SVR(), param_grid={"kernel": ("linear", 'rbf'), "C": np.logspace(-3, 3, 7), "gamma": np.logspace(-3, 3, 7)})
    svr = SVR()
    svr.fit(x, y)
    y_pred = svr.predict(test_data)
    aDataFrame = pd.DataFrame(y_pred)
    aDataFrame.to_csv(f'./processed_data_{index}.csv', index=False)


