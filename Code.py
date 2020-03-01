#!/usr/bin/env python
# coding: utf-8

# In[706]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


# In[707]:


football_df = pd.read_csv("data.csv")


# In[708]:


football_df.head()


# In[709]:


football_df.columns


# # Data Cleaning

# In[714]:


unique_mins = football_df["remaining_min"].unique()
unique_mins = unique_mins[~np.isnan(unique_mins)]

count = football_df.groupby("remaining_min").size()
count /= count.sum()


# In[716]:


football_df["remaining_min"] = football_df.remaining_min.map(lambda x: np.random.choice(unique_mins, p=count) if np.isnan(x) else x)


# In[718]:


#Dropping unecessary and redundant columns


# In[717]:


columns_to_drop = [
    "team_name",
    "home/away",
    "date_of_game",
    "team_id",
    "remaining_min.1",
    "power_of_shot.1",
    "knockout_match.1",
    "remaining_sec.1",
    "distance_of_shot.1",
    "Unnamed: 0",
]


# In[719]:


football_df.drop(columns_to_drop, axis=1, inplace=True)


# In[720]:


football_df.columns


# In[721]:


football_df.match_event_id.fillna(value = -1, inplace = True)


# In[722]:


unique_pos = football_df["power_of_shot"].unique()
unique_pos = unique_pos[~np.isnan(unique_pos)]

count_pos = football_df.groupby("power_of_shot").size()
count_pos /= count_pos.sum()


# In[723]:


football_df["power_of_shot"] = football_df.power_of_shot.map(lambda x: np.random.choice(unique_pos, p = count_pos) if np.isnan(x) else x )


# In[724]:


count_knockout = football_df.groupby("knockout_match").size()
count_knockout /= count_knockout.sum()


# In[725]:


football_df["knockout_match"] = football_df.knockout_match.map(lambda x: np.random.choice([0,1], p = count_knockout) if np.isnan(x) else x )


# In[726]:


# football_df.drop(["type_of_shot","type_of_combined_shot"], axis = 1, inplace=True)


# In[727]:


football_df[["lat","lng"]] = football_df["lat/lng"].str.split(",",expand = True)


# In[728]:


football_df.drop("lat/lng", inplace=True, axis = 1)


# In[729]:


football_df["game_season"] = football_df["game_season"].fillna(method = 'backfill')


# In[730]:


football_df["range_of_shot"] = football_df["range_of_shot"].astype('category')
football_df["range_of_shot"] = football_df["range_of_shot"].cat.codes

football_df["shot_basics"] = football_df["shot_basics"].astype('category')
football_df["shot_basics"] = football_df["shot_basics"].cat.codes

football_df["area_of_shot"] = football_df["area_of_shot"].astype('category')
football_df["area_of_shot"] = football_df["area_of_shot"].cat.codes

football_df["game_season"] = football_df["game_season"].astype('category')
football_df["game_season"] = football_df["game_season"].cat.codes


# In[731]:


football_df["shot_id_number"] = football_df.index
football_df.shot_id_number = football_df.shot_id_number.map(lambda x: x+1)


# In[732]:


football_df.type_of_shot.fillna(football_df.type_of_combined_shot, inplace=True)
football_df.drop("type_of_combined_shot", axis = 1, inplace=True)


# In[733]:


football_df["type_of_shot"] = football_df.type_of_shot.map(lambda x: int(str(x).split(" - ")[1]), na_action="ignore")


# In[734]:


unique_types_of_shot = football_df["type_of_shot"].unique()
unique_types_of_shot = unique_types_of_shot[~np.isnan(unique_types_of_shot)]

count_types_of_shot = football_df.groupby("type_of_shot").size()
count_types_of_shot /= count_types_of_shot.sum()


# In[735]:


football_df["type_of_shot"] = football_df.type_of_shot.map(lambda x: np.random.choice(unique_types_of_shot,p=count_types_of_shot) if np.isnan(x) else x)


# In[736]:


imputer = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)
imputer = imputer.fit(football_df.iloc[:,6:7])
football_df.iloc[:,6:7] = imputer.transform(football_df.iloc[:,6:7])


# In[737]:


imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(football_df.iloc[:,1:3])
football_df.iloc[:,1:3] = imputer.transform(football_df.iloc[:,1:3])


# In[738]:


imputer = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)
imputer = imputer.fit(football_df.iloc[:,15:])
football_df.iloc[:,15:] = imputer.transform(football_df.iloc[:,15:])


# In[739]:


imputer = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
imputer = imputer.fit(football_df.iloc[:,7:8])
football_df.iloc[:,7:8] = imputer.transform(football_df.iloc[:,7:8])


# In[768]:


unique_dist_of_shot = football_df["distance_of_shot"].unique()
unique_dist_of_shot = unique_dist_of_shot[~np.isnan(unique_dist_of_shot)]

count_dist_of_shot = football_df.groupby("distance_of_shot").size()
count_dist_of_shot /= count_dist_of_shot.sum()


# In[769]:


football_df["distance_of_shot"] = football_df.distance_of_shot.map(lambda x: np.random.choice(unique_types_of_shot,p=count_types_of_shot) if np.isnan(x) else x)


# In[770]:


football_df.head()


# In[771]:


football_df.isnull().sum()


# # Splitting data into Training and Testing data

# In[772]:


condition = np.isnan(football_df["is_goal"])


# In[773]:


unlabelled_data = football_df[condition]


# In[774]:


unlabelled_data.shape


# In[775]:


labelled_data = football_df[~condition]


# In[776]:


labelled_data.shape


# In[777]:


labelled_data = shuffle(labelled_data)
labelled_data.head()


# In[778]:


Y = labelled_data["is_goal"]


# In[779]:


Y.shape


# In[780]:


X = labelled_data.drop(["is_goal","shot_id_number"], axis = 1)


# In[781]:


X.shape


# In[782]:


shot_id_number = unlabelled_data.shot_id_number


# In[783]:


X_pred = unlabelled_data.drop(["is_goal","shot_id_number"], axis = 1)


# In[784]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


# In[785]:


from sklearn.ensemble import RandomForestClassifier


# In[786]:


RF = RandomForestClassifier(n_estimators = 500)


# In[787]:


RF.fit(X_train,Y_train)


# In[788]:


RF.score(X_test,Y_test)


# In[789]:


Y_pred = RF.predict(X_pred)


# In[790]:


d = {
    "shot_id_number" : shot_id_number,
    "is_goal" : Y_pred
}


# In[791]:


final = pd.DataFrame(d)


# In[792]:


final.set_index('shot_id_number', inplace=True)


# In[793]:


final


# In[794]:


final.to_csv("bipin_kalra_110597_code_1.csv")


# In[ ]:




