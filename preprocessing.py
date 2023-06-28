##############
''' IMPORT '''
##############
import os
import numpy as np
import pandas as pd

##############################
''' WEATHER LIST TO REMOVE '''
##############################
rain = [272, 373, 415, 657, 725, 761, 783, 846, 1027, 1133, 1420, 1695, 2356, 2480, 2513, 2541, 2596, 2604, 2635, 2654]

snow = [225, 248, 2147]
# snow = [225, 232, 246, 247, 248, 314, 482, 518, 713, 850, 2147]

normal = [4, 17, 21, 56, 108, 126, 144, 149, 172, 190, 221, 234, 238, 263, 283, 306, 332, 341, 349, 374, 375, 400, 402, 419, 488, 495,
          507, 520, 545, 554, 563, 588, 620, 621, 630, 636, 645, 652, 668, 684, 693, 718, 720, 728, 740, 747, 748, 771, 778, 789, 792, 793, 799, 804,
          805, 856, 858, 861, 875, 877, 889, 890, 892, 896, 899, 917, 920, 920, 970, 978, 1018, 1041, 1060, 1061, 1076, 1081, 1086, 1098, 1126,
          1127, 1169, 1172, 1182, 1193, 1224, 1251, 1263, 1270, 1280, 1287, 1301, 1328, 1333, 1423, 1439, 1467, 1488, 1492, 1506, 1531, 1533, 1542,
          1555, 1559, 1581, 1586, 1605, 1636, 1654, 1656, 1698, 1699, 1703, 1718, 1727, 1728, 1771, 1773, 1786, 1795, 1839, 1848, 1849, 1851, 1874,
          1904, 1917, 1923, 1940, 1944, 1949, 1955, 1972, 1975, 2042, 2046, 2051, 2059, 2040, 2180, 2186, 2193, 2198, 2230, 2236,  2239, 2249, 2352,
          2388, 2391, 2449, 2451, 2486, 2532, 2534, 2570, 2571, 2603, 2607, 2615, 2626, 2645, 2647, 2669, 2685]

drop_weather = rain + normal + snow

#################
''' DATA LOAD '''
#################
def load(root, data_dir, filename):
    df = pd.read_csv(os.path.join(root, data_dir, filename)) # Data Load
    df['video_path'] = df['video_path'].apply(lambda x: os.path.join(root, data_dir, x[2:])) # Change Address
    return df

##########################
''' LABEL SEGMENTATION '''
##########################
def separate(df, labelname):
    print("Multi Labels")
    nL = df['label']

    nC = np.count_nonzero(nL == 0)
    print(f"crash\nYes: {len(nL) - nC}, No: {nC}")

    nE = np.count_nonzero((0 < nL) & (nL < 7))
    print(f"ego-involve\nYes: {nE}, No: {len(nL) - nE - nC}")

    nWn = np.count_nonzero((nL == 1) | (nL == 2) | (nL == 7) | (nL == 8))
    nWs = np.count_nonzero((nL == 3) | (nL == 4) | (nL == 9) | (nL == 10))
    print(f"weather\nNormal: {nWn}, Snowy: {nWs}, Rainy: {len(nL) - (nC + nWn + nWs)}")

    nT = np.count_nonzero(nL % 2 != 0)
    print(f"timing\nDay: {nT}, Night: {len(nL) - (nC + nT)}\n")

    ''' separate labels '''
    if labelname == 'crash':
        for i in range(len(df['label'])):
            if df['label'][i] == 0: # No
                df['label'][i] = 0
            else:   # YES
                df['label'][i] = 1
    elif labelname == 'ego_involve':
        delete = []
        for i in range(len(df['label'])):
            if df['label'][i] == 0:
                delete.append(i)
            elif 1 <= df['label'][i] <= 6:  # YES
                df['label'][i] = 0
            elif 7 <= df['label'][i] <= 12:  # NO
                df['label'][i] = 1
        df.drop(delete, axis=0, inplace=True)
    elif labelname == 'weather':
        delete = []
        for i in range(len(df['label'])):
            k = df['label'][i]
            if k == 0:
                delete.append(i)
            else:
                if (1 <= k <= 2) or (7 <= k <= 8):  # Normal
                    df['label'][i] = 0
                elif (3 <= k <= 4) or (9 <= k <= 10):  # Snowy
                    df['label'][i] = 1
                else:  # Rainy
                    df['label'][i] = 2
        for i in range(len(df['label'])):
            if int(df['sample_id'][i][6:]) in drop_weather:
                delete.append(i) 
        df.drop(delete, axis=0, inplace=True)
    elif labelname == 'timing':
        delete = []
        for i in range(len(df['label'])):
            if df['label'][i] == 0:
                delete.append(i)
            else:
                if df['label'][i]%2 != 0:
                    df['label'][i] = 0
                else:
                    df['label'][i] = 1
        df.drop(delete, axis=0, inplace=True)

    # separate = []
    # for label in df['label']:
    #     if label == 0:
    #         crash_ego = 0
    #         weather = 0
    #         timing = 0
    #     elif label == 1:
    #         crash_ego = 1
    #         weather = 0
    #         timing = 0
    #     elif label == 2:
    #         crash_ego = 1
    #         weather = 0
    #         timing = 1
    #     elif label == 3:
    #         crash_ego = 1
    #         weather = 1
    #         timing = 0
    #     elif label == 4:
    #         crash_ego = 1
    #         weather = 1
    #         timing = 1
    #     elif label == 5:
    #         crash_ego = 1
    #         weather = 2
    #         timing = 0
    #     elif label == 6:
    #         crash_ego = 1
    #         weather = 2
    #         timing = 1
    #     elif label == 7:
    #         crash_ego = 1
    #         weather = 0
    #         timing = 0
    #     elif label == 8:
    #         crash_ego = 1
    #         weather = 0
    #         timing = 1
    #     elif label == 9:
    #         crash_ego = 1
    #         weather = 1
    #         timing = 0
    #     elif label == 10:
    #         crash_ego = 1
    #         weather = 1
    #         timing = 1
    #     elif label == 11:
    #         crash_ego = 1
    #         weather = 2
    #         timing = 0
    #     else:
    #         crash_ego = 1
    #         weather = 2
    #         timing = 1
    #     separate.append([crash_ego, weather, timing])
    #
    # df[['crash_ego', 'weather', 'timing']] = separate
    return df

####################
''' DELETE LABEL '''
####################
def unlabeled(df, is_file, filename):
    delete = []
    for i in range(len(df['label'])):
        if df['label'][i] != 0:  # Yes
            delete.append(i)
    if is_file == True:
        pseudo_df = pd.read_csv(f'./data/{filename}.csv')
        normal_df = pd.read_csv(f'./data/pseudo_normal1.csv')
        for i in range(len(df['label'])):
            for j in range(len(pseudo_df['sample_id'])):
                if df['sample_id'][i] == pseudo_df['sample_id'][j]:  # Yes
                    delete.append(i)
                    break
            for k in range(len(normal_df['sample_id'])):
                if df['sample_id'][i] == normal_df['sample_id'][k]:
                    delete.append(i)
                    break
    outlier = [2508, 104]
    delete+=outlier
    df.drop(delete, axis=0, inplace=True)
    return df