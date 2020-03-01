#!/usr/bin/env python
# coding: utf-8

# In[36]:


import pandas as pd
import numpy as np


# In[37]:


football_df = pd.read_csv('data.csv')


# In[38]:


football_df


# # Data Preprocessing

# In[39]:


cols_to_drop = [
    "type_of_combined_shot",
    "team_id",
    "team_name"
]


# In[40]:


football_df['type_of_shot'] = [ s1 if not pd.isna(s1) else s2 for s1, s2 in zip(football_df['type_of_shot'], football_df['type_of_combined_shot']) ]


# In[41]:


football_df.drop(cols_to_drop, inplace = True, axis = 1)


# In[42]:


home_latitude_map = { home: l for home, l in zip(football_df['home/away'], football_df['lat/lng']) if not pd.isna(home) and not pd.isna(l)}


# In[43]:


match_latitude_map = { match: l for match, l in zip(football_df['match_id'], football_df['lat/lng']) if not pd.isna(match) and not pd.isna(l)}


# In[44]:


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


# In[45]:


cols_to_drop2 = [
    "lat/lng",
    "home/away",
    "date_of_game",
]


# In[46]:


football_df.drop(cols_to_drop2, inplace= True, axis =1)


# In[47]:


football_df.columns


# In[48]:


# res = football_df[~football_df['power_of_shot.1'].isnull()][['power_of_shot', 'power_of_shot.1', 'distance_of_shot', 'distance_of_shot.1', 'type_of_shot', 'shot_basics', 'range_of_shot']]


# In[49]:


range_of_shot_unique = list(set(football_df['range_of_shot']))


# In[50]:


range_of_shot_unique.pop(0)


# In[51]:


#Required unique classes of shot ranges
range_of_shot_unique


# In[52]:


range_map = {}


# In[53]:


football_df_clean = football_df[football_df['distance_of_shot'] == football_df['distance_of_shot.1']]


# In[54]:


for val in range_of_shot_unique:
    d1 = football_df_clean[football_df_clean['range_of_shot'] == val]['distance_of_shot.1']
    d2 = football_df_clean[football_df_clean['range_of_shot'] == val]['distance_of_shot']
    range_map[val] = range(int(min(d1.min(), d2.min())), int(max(d1.max(), d2.max())))


# In[55]:


range_map


# In[56]:


def range_return(dist): 
    for l, r in range_map.items():
        if dist in r:
            return l


# In[57]:


def range_update(x):
    if not pd.isna(x['range_of_shot']): return x['range_of_shot']
    if x['distance_of_shot'] != x['distance_of_shot.1']:
        if range_return(x['distance_of_shot']):
            return range_return(x['distance_of_shot'])
        else:
            return range_return(x['distance_of_shot.1'])
    else:
        return range_return(x['distance_of_shot'])


# In[58]:


football_df['range_of_shot'] = football_df.apply(range_update, axis=1)


# In[59]:


football_df.drop(["distance_of_shot","distance_of_shot.1"], axis = 1, inplace=True)


# In[60]:


# set(football_df[~football_df['power_of_shot'].isnull()]['power_of_shot'])


# In[61]:


# football_df[football_df['power_of_shot'].isnull() & football_df['power_of_shot.1'] > 7][['power_of_shot', 'power_of_shot.1', 'type_of_shot', 'area_of_shot']]


# In[62]:


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


# In[63]:


football_df.columns


# In[64]:


football_df.drop("shot_basics", inplace=True, axis = 1)


# In[65]:


#Mapping area of shot with location x and y since x and y represent the same data
area_map = {}
football_df_clean = football_df[~football_df['area_of_shot'].isnull()]
for val in set(football_df_clean['area_of_shot']):
    x = football_df_clean[football_df_clean['area_of_shot'] == val]
    area_map[val] =         range(int(x['location_x'].min()), int(x['location_x'].max())),         range(int(x['location_y'].min()), int(x['location_y'].max()))


# In[66]:


area_map


# In[69]:


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


# In[70]:


football_df['area_of_shot'] = football_df.apply(area_update, axis=1)


# In[71]:


football_df.drop(["location_x","location_y"], inplace = True, axis = 1)


# In[73]:


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


# In[74]:


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


# In[75]:


football_df['remaining_min'] = football_df.apply(
    lambda x:
        x['remaining_min']
        if not pd.isna(x['remaining_min'])
        else x['remaining_min.1'] % 11,
    axis = 1
)


# In[76]:


football_df['remaining_sec'] = football_df.apply(
    lambda x:
        x['remaining_sec']
        if not pd.isna(x['remaining_sec'])
        else x['remaining_sec.1'] % 60,
    axis = 1
)


# In[77]:


football_df['remaining_min'] = football_df.apply(
    lambda x:
        x['remaining_min']
        if not pd.isna(x['remaining_min'])
        else 0,
    axis = 1
)


# In[78]:


football_df['remaining_sec'] = football_df.apply(
    lambda x:
        x['remaining_sec']
        if not pd.isna(x['remaining_sec'])
        else 0,
    axis = 1
)


# In[79]:


cols_to_drop3 = [
    "remaining_min.1",
    "remaining_sec.1",
    "Unnamed: 0",
    "power_of_shot.1"
]


# In[80]:


football_df.drop(cols_to_drop3, axis = 1, inplace = True)


# In[81]:


football_df.columns


# In[82]:


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


# In[83]:


def mapping_categorical_values(c, d):
    def func(x):
        return d[x[c]]
    return func


# In[84]:


df_back = football_df


# In[85]:


mapping_list = [
    ("game_season", game_season_mapping),
    ("area_of_shot", area_of_shot_mapping),
    ("range_of_shot", range_of_shot_mapping),
    ("type_of_shot", type_of_shot_mapping) 
]


# In[87]:


for col, d in mapping_list:
    football_df[col] = football_df.apply(mapping_categorical_values(col, d), axis = 1)


# In[88]:


# for col, d in [
#     ('game_season', game_season_map), 
#     ('area_of_shot', area_shot_map), 
#     ('range_of_shot', range_shot_map), 
#     ('type_of_shot', type_shot_map)
# ]:
#     football_df[col] = football_df.apply(map_vals(col, d), axis = 1)


# In[89]:


# football_df['power_of_shot'] = football_df.apply(
#     lambda row:
#         row['power_of_shot']
#         if not pd.isna(row['power_of_shot'])
#         else 0,
#     axis = 1
# )


# In[90]:


unique_powers = football_df.power_of_shot.unique()
unique_powers = unique_powers[~np.isnan(unique_powers)]


# In[91]:


count_powers = football_df.groupby("power_of_shot").size()
count_powers /= count_powers.sum()


# In[92]:


football_df['power_of_shot'] = football_df.apply(
    lambda x:
        x['power_of_shot']
        if not pd.isna(x['power_of_shot'])
        else np.random.choice(unique_powers, p = count_powers),
    axis = 1
)


# In[93]:


def knockout_match_update(x):
    if not pd.isna(x['knockout_match']): return x['knockout_match']
    if not pd.isna(x['knockout_match.1']) and int(x['knockout_match.1']) in [0, 1]:
        return x['knockout_match.1']
    return x['knockout_match']


# In[94]:


football_df['knockout_match'] = football_df.apply(knockout_match_update, axis = 1)


# In[95]:


count_knockout = football_df.groupby("knockout_match").size()
count_knockout /= count_knockout.sum()


# In[96]:


football_df["knockout_match"] = football_df.knockout_match.map(
    lambda x:
        np.random.choice([0,1], p = count_knockout) 
        if np.isnan(x) 
        else x 
    )


# In[97]:


intermediate = football_df[~football_df['game_season'].isnull()][['game_season', 'match_id']]


# In[98]:


match_id_game_season_map = dict(intermediate.groupby('game_season')['match_id'].apply(list))


# In[99]:


def game_season_mapping(x):
    if not pd.isna(x['game_season']): return x['game_season']
    for key, value in match_id_game_season_map.items():
        if x['match_id'] in value:
            return key
    return np.random.choice(list(match_id_season_map.keys()))


# In[100]:


football_df['game_season'] = football_df.apply(game_season_mapping, axis = 1)


# In[101]:


football_df.drop("knockout_match.1", axis = 1, inplace = True)


# In[102]:


football_df.head()


# # Model Training and Prediction

# In[123]:


from sklearn.svm import SVR


# In[124]:


clf = SVR()


# In[105]:


headers = list(football_df.columns)[1:]


# In[106]:


headers.remove('is_goal')
headers.remove('shot_id_number')


# In[107]:


condition = football_df["is_goal"].isnull()


# In[108]:


X = np.array(football_df[~condition][headers])
Y = np.array(football_df[~condition]['is_goal'])


# In[125]:


clf.fit(X, Y)


# In[110]:


condition = football_df['is_goal'].isnull() & ~football_df['shot_id_number'].isnull()


# In[111]:


id_numbers = np.array(football_df[condition]['shot_id_number'])
X_pred = np.array(football_df[condition][headers])


# In[130]:


Y_pred = clf.predict(X_pred)


# In[133]:


for i in range(0,len(Y_pred)):
    if Y_pred[i] >= 0.5:
        Y_pred[i] = 1
    else:
        Y_pred[i] = 0


# In[134]:


Y_pred[1]


# In[116]:


# from sklearn.ensemble import RandomForestClassifier


# In[117]:


# RF = RandomForestClassifier(n_estimators = 1000)


# In[118]:


# RF.fit(X,Y)


# In[119]:


# Y_pred2 = RF.predict(X_pred)


# # Writing to CSV File

# In[121]:


import csv


# In[135]:


with open("bipin_kalra_110597_code_3.csv", "w") as file:
    writer = csv.writer(file)
    headers = ['shot_id_number','is_goal']
    writer.writerow(headers)
    writer.writerows(zip(id_numbers, Y_pred))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




