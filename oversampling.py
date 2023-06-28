##############
''' IMPORT '''
##############
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings(action='ignore')

#################
''' DATA COPY '''
#################
def addition(df, labelname):
    nL = df['label']
    if labelname == 'crash':
        no_crash = np.count_nonzero(nL == 0)
        crash = np.count_nonzero(nL == 1)

        crash_df = df[df.label == 1].copy()
        crash_df['sample_id'] = crash_df['sample_id'].apply(lambda x: x + 'V')

        diff = no_crash // crash
        for _ in range(diff):
            df = pd.concat([df, crash_df], ignore_index=True)
        return df
    elif labelname == 'weather':
        normal = np.count_nonzero(nL == 0)
        snowy = np.count_nonzero(nL == 1)
        rainy = np.count_nonzero(nL == 2)

        snowy_df = df[df.label == 1].copy()
        rainy_df = df[df.label == 2].copy()
        snowy_df['sample_id'] = snowy_df['sample_id'].apply(lambda x: x + 'V')
        rainy_df['sample_id'] = rainy_df['sample_id'].apply(lambda x: x + 'V')

        diff1 = normal // snowy
        for _ in range(diff1):
            df = pd.concat([df, snowy_df], ignore_index=True)
        diff2 = normal // rainy
        for _ in range(diff1):
            df = pd.concat([df, rainy_df], ignore_index=True)
        return df
    elif labelname == 'timing':
        day = np.count_nonzero(nL == 0)
        night = np.count_nonzero(nL == 1)

        night_df = df[df.label == 1].copy()
        night_df['sample_id'] = night_df['sample_id'].apply(lambda x: x + 'V')

        diff = day // night
        for _ in range(diff):
            df = pd.concat([df, night_df], ignore_index=True)
        return df

# crash_df = preprocessing.load('.', 'data', 'train.csv')
# crash_df = preprocessing.separate(crash_df, 'crash')
# # print(crash_df)
#
# cover_df = addition(crash_df, 'crash')
# print(cover_df)
#
# crash_train, crash_val, _, _ = train_test_split(cover_df, cover_df['label'], test_size=0.2,
#                                                         random_state=42)
# print(crash_train)