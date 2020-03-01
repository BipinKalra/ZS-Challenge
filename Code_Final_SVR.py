#!/usr/bin/env python
# coding: utf-8

# In[234]:


import pandas as pd
import numpy as np


# In[235]:


football_df = pd.read_csv('data.csv')


# In[236]:


football_df


# # Data Preprocessing

# In[237]:


cols_to_drop = [
    "type_of_combined_shot",
    "team_id",
    "team_name"
]


# In[238]:


football_df['type_of_shot'] = [ s1 if not pd.isna(s1) else s2 for s1, s2 in zip(football_df['type_of_shot'], football_df['type_of_combined_shot']) ]


# In[239]:


football_df.drop(cols_to_drop, inplace = True, axis = 1)


# In[240]:


home_latitude_map = { home: l for home, l in zip(football_df['home/away'], football_df['lat/lng']) if not pd.isna(home) and not pd.isna(l)}


# In[241]:


match_latitude_map = { match: l for match, l in zip(football_df['match_id'], football_df['lat/lng']) if not pd.isna(match) and not pd.isna(l)}


# In[242]:


football_df['lat/lng'] = football_df.apply(
    lambda x: 
        home_latitude_map[x['home/away']] 
        if not pd.isna(x['home/away']) 
        else (
            match_latitude_map[x['match_id']] 
            if not pd.isna(x['match_id']) 
            else x['lat/lng']
        ),
    axis=1
)


# In[243]:


cols_to_drop2 = [
    "lat/lng",
    "home/away",
    "date_of_game",
]


# In[244]:


football_df.drop(cols_to_drop2, inplace= True, axis =1)


# In[245]:


football_df.columns


# In[246]:


# res = football_df[~football_df['power_of_shot.1'].isnull()][['power_of_shot', 'power_of_shot.1', 'distance_of_shot', 'distance_of_shot.1', 'type_of_shot', 'shot_basics', 'range_of_shot']]


# In[247]:


range_of_shot_unique = list(set(football_df['range_of_shot']))


# In[248]:


range_of_shot_unique.pop(0)


# In[249]:


#Required unique classes of shot ranges
range_of_shot_unique


# In[250]:


range_map = {}


# In[251]:


football_df_clean = football_df[football_df['distance_of_shot'] == football_df['distance_of_shot.1']]


# In[252]:


for val in range_of_shot_unique:
    d1 = football_df_clean[football_df_clean['range_of_shot'] == val]['distance_of_shot.1']
    d2 = football_df_clean[football_df_clean['range_of_shot'] == val]['distance_of_shot']
    range_map[val] = range(int(min(d1.min(), d2.min())), int(max(d1.max(), d2.max())))


# In[253]:


range_map


# In[254]:


def range_return(dist): 
    for l, r in range_map.items():
        if dist in r:
            return l


# In[255]:


def range_update(x):
    if not pd.isna(x['range_of_shot']): return x['range_of_shot']
    if x['distance_of_shot'] != x['distance_of_shot.1']:
        if range_return(x['distance_of_shot']):
            return range_return(x['distance_of_shot'])
        else:
            return range_return(x['distance_of_shot.1'])
    else:
        return range_return(x['distance_of_shot'])


# In[256]:


football_df['range_of_shot'] = football_df.apply(range_update, axis=1)


# In[257]:


football_df.drop(["distance_of_shot","distance_of_shot.1"], axis = 1, inplace=True)


# In[260]:


football_df['power_of_shot'] = football_df.apply(
    lambda x:
        x['power_of_shot']
        if not pd.isna(x['power_of_shot'])
        else (
            x['power_of_shot.1']
            if not pd.isna(x['power_of_shot.1'])
            else x['power_of_shot']
        ),
    axis = 1
)


# In[261]:


football_df.columns


# In[262]:


football_df.drop("shot_basics", inplace=True, axis = 1)


# In[263]:


#Mapping area of shot with location x and y since x and y represent the same data
area_map = {}
football_df_clean = football_df[~football_df['area_of_shot'].isnull()]
for val in set(football_df_clean['area_of_shot']):
    x = football_df_clean[football_df_clean['area_of_shot'] == val]
    area_map[val] =         range(int(x['location_x'].min()), int(x['location_x'].max())),         range(int(x['location_y'].min()), int(x['location_y'].max()))


# In[264]:


area_map


# In[265]:


def area_update(x):
    if not pd.isna(x['area_of_shot']): return x['area_of_shot']
    for key, value in area_map.items():
        if not pd.isna(x['location_x']) and not pd.isna(x['location_y']):
            if x['location_x'] in value[0] and x['location_y'] in value[1]:
                return key
        elif not pd.isna(x['location_x']):
            if x['location_x'] in value[0]:
                return key
        elif not pd.isna(x['location_y']):
            if x['location_y'] in value[1]:
                return key
    return x['area_of_shot']


# In[266]:


football_df['area_of_shot'] = football_df.apply(area_update, axis=1)


# In[267]:


football_df.drop(["location_x","location_y"], inplace = True, axis = 1)


# In[268]:


football_df['remaining_min'] = football_df.apply(
    lambda x:
        x['remaining_min']
        if not pd.isna(x['remaining_min'])
        else (
            x['remaining_min.1']
            if x['remaining_min.1'] <= 11
            else x['remaining_min']
        ),
    axis = 1
)


# In[269]:


football_df['remaining_sec'] = football_df.apply(
    lambda x:
        x['remaining_sec']
        if not pd.isna(x['remaining_sec'])
        else (
            x['remaining_sec.1']
            if x['remaining_sec.1'] <= 59
            else x['remaining_sec']
        ),
    axis = 1
)


# In[270]:


football_df['remaining_min'] = football_df.apply(
    lambda x:
        x['remaining_min']
        if not pd.isna(x['remaining_min'])
        else x['remaining_min.1'] % 11,
    axis = 1
)


# In[271]:


football_df['remaining_sec'] = football_df.apply(
    lambda x:
        x['remaining_sec']
        if not pd.isna(x['remaining_sec'])
        else x['remaining_sec.1'] % 60,
    axis = 1
)


# In[272]:


football_df['remaining_min'] = football_df.apply(
    lambda x:
        x['remaining_min']
        if not pd.isna(x['remaining_min'])
        else 0,
    axis = 1
)


# In[273]:


football_df['remaining_sec'] = football_df.apply(
    lambda x:
        x['remaining_sec']
        if not pd.isna(x['remaining_sec'])
        else 0,
    axis = 1
)


# In[274]:


cols_to_drop3 = [
    "remaining_min.1",
    "remaining_sec.1",
    "Unnamed: 0",
    "power_of_shot.1"
]


# In[275]:


football_df.drop(cols_to_drop3, axis = 1, inplace = True)


# In[276]:


football_df.columns


# In[277]:


#encoding categorical data using enumeration method

area_of_shot_mapping = {}
for i, val in enumerate(set(football_df['area_of_shot'])):
    area_of_shot_mapping[val] = i

type_of_shot_mapping = {}
for i, val in enumerate(set(football_df['type_of_shot'])):
    type_of_shot_mapping[val] = i
    
range_of_shot_mapping = {}
for i, val in enumerate(set(football_df['range_of_shot'])):
    range_of_shot_mapping[val] = i

game_season_mapping = {}
for i, val in enumerate(set(football_df['game_season'])):
    game_season_mapping[val] = i


# In[278]:


def mapping_categorical_values(c, d):
    def func(x):
        return d[x[c]]
    return func


# In[279]:


df_back = football_df


# In[280]:


mapping_list = [
    ("game_season", game_season_mapping),
    ("area_of_shot", area_of_shot_mapping),
    ("range_of_shot", range_of_shot_mapping),
    ("type_of_shot", type_of_shot_mapping) 
]


# In[281]:


for col, d in mapping_list:
    football_df[col] = football_df.apply(mapping_categorical_values(col, d), axis = 1)


# In[283]:


unique_powers = football_df.power_of_shot.unique()
unique_powers = unique_powers[~np.isnan(unique_powers)]


# In[284]:


count_powers = football_df.groupby("power_of_shot").size()
count_powers /= count_powers.sum()


# In[285]:


football_df['power_of_shot'] = football_df.apply(
    lambda x:
        x['power_of_shot']
        if not pd.isna(x['power_of_shot'])
        else np.random.choice(unique_powers, p = count_powers),
    axis = 1
)


# In[286]:


def knockout_match_update(x):
    if not pd.isna(x['knockout_match']): return x['knockout_match']
    if not pd.isna(x['knockout_match.1']) and int(x['knockout_match.1']) in [0, 1]:
        return x['knockout_match.1']
    return x['knockout_match']


# In[287]:


football_df['knockout_match'] = football_df.apply(knockout_match_update, axis = 1)


# In[288]:


count_knockout = football_df.groupby("knockout_match").size()
count_knockout /= count_knockout.sum()


# In[289]:


football_df["knockout_match"] = football_df.knockout_match.map(
    lambda x:
        np.random.choice([0,1], p = count_knockout) 
        if np.isnan(x) 
        else x 
    )


# In[290]:


intermediate = football_df[~football_df['game_season'].isnull()][['game_season', 'match_id']]


# In[291]:


match_id_game_season_map = dict(intermediate.groupby('game_season')['match_id'].apply(list))


# In[292]:


def game_season_mapping(x):
    if not pd.isna(x['game_season']): return x['game_season']
    for key, value in match_id_game_season_map.items():
        if x['match_id'] in value:
            return key
    return np.random.choice(list(match_id_season_map.keys()))


# In[293]:


football_df['game_season'] = football_df.apply(game_season_mapping, axis = 1)


# In[294]:


football_df.drop("knockout_match.1", axis = 1, inplace = True)


# In[295]:


football_df.head()


# # Model Training and Prediction

# In[296]:


from sklearn.svm import SVR


# In[297]:


clf = SVR()


# In[298]:


headers = list(football_df.columns)[1:]


# In[299]:


headers.remove('is_goal')
headers.remove('shot_id_number')


# In[300]:


condition = football_df["is_goal"].isnull()


# In[301]:


X = np.array(football_df[~condition][headers])
Y = np.array(football_df[~condition]['is_goal'])


# In[302]:


clf.fit(X, Y)


# In[303]:


condition = football_df['is_goal'].isnull() & ~football_df['shot_id_number'].isnull()


# In[304]:


id_numbers = np.array(football_df[condition]['shot_id_number'])
X_pred = np.array(football_df[condition][headers])


# In[305]:


Y_pred = clf.predict(X_pred)


# In[307]:


Y_pred[1]


# # Writing to CSV File

# In[308]:


import csv


# In[309]:


with open("bipin_kalra_110597_code_7.csv", "w") as file:
    writer = csv.writer(file)
    headers = ['shot_id_number','is_goal']
    writer.writerow(headers)
    writer.writerows(zip(id_numbers, Y_pred))

