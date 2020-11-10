import pandas as pd
import numpy as np
from sklearn import tree
#from sklearn.ensemble import AdaBoostClassifier
#from progress.bar import Bar
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
#from sklearn.ensemble import BaggingClassifier
# Load in the Data
data = pd.read_csv("movie_metadata.csv", header=0, skipinitialspace = False, na_values=["?"])
# =============================================================================
# Pre-Precessing
# =============================================================================
data = data.drop(columns=['color', 'num_critic_for_reviews', 'duration',
       'director_facebook_likes', 'actor_3_facebook_likes', 'language',
       'actor_1_facebook_likes', 'num_voted_users', 'cast_total_facebook_likes',
       'facenumber_in_poster', 'num_user_for_reviews', 'movie_imdb_link',
       'actor_2_facebook_likes', 'aspect_ratio', 'country',
       'movie_facebook_likes', 'movie_title'])
# Drop the null for the following - they weren't valuable anyway (we checked)
data = data.dropna(subset=['plot_keywords'])
data = data.dropna(subset=['actor_1_name'])
# Fill the null with the actor_1_name because it provides more data without
#  sacrificing valuable info (there weren't a ton with this issue)
data.actor_2_name = data.actor_2_name.fillna(data.actor_1_name)
data.actor_3_name = data.actor_3_name.fillna(data.actor_1_name)
data.director_name = data.director_name.fillna(data.actor_1_name)
# Fill the null with "Not Rated" - we checked
data.content_rating = data.content_rating.fillna("Not Rated")
# Fill the null with the mean value (dates and budgets and gross shouldn't
#  be a deciding factor in this algorithm anyway)
data.gross = data.gross.fillna(data.gross.mean())
data.budget = data.budget.fillna(data.budget.mean())
data.title_year = data.title_year.fillna(data.title_year.mean())

# Now we need to make new row items for every plot key word and genre
#  A basic list will make this faster (trust me... I tried everything else)
#  We will store each different key word in its own array to find 5 keywords
data_list_1 = []
data_list_2 = []
data_list_3 = []
data_list_4 = []
data_list_5 = []

# For each row in our data grame...
for index, row in data.iterrows():
    # ...split the plot_keywords and genres columns by the '|' delimeter
    kw = row.plot_keywords
    genre = row.genres
    kw_list = kw.split("|")
    genre_list = genre.split("|")
    new_row = data.loc[index].values
    new_row[3] = genre_list[0]
    if (len(kw_list) == 5):
        new_row[6] = kw_list[0]
        data_list_1.append(new_row)
        new_row[6] = kw_list[1]
        data_list_2.append(new_row)
        new_row[6] = kw_list[2]
        data_list_3.append(new_row)
        new_row[6] = kw_list[3]
        data_list_4.append(new_row)
        new_row[6] = kw_list[4]
        data_list_5.append(new_row)
    elif (len(kw_list) == 4):
        new_row[6] = kw_list[0]
        data_list_1.append(new_row)
        new_row[6] = kw_list[1]
        data_list_2.append(new_row)
        new_row[6] = kw_list[2]
        data_list_3.append(new_row)
        new_row[6] = kw_list[3]
        data_list_4.append(new_row)
    elif (len(kw_list) == 3):
        new_row[6] = kw_list[0]
        data_list_1.append(new_row)
        new_row[6] = kw_list[1]
        data_list_2.append(new_row)
        new_row[6] = kw_list[2]
        data_list_3.append(new_row)
    elif (len(kw_list) == 2):
        new_row[6] = kw_list[0]
        data_list_1.append(new_row)
        new_row[6] = kw_list[1]
        data_list_2.append(new_row)
    elif (len(kw_list) == 1):
        new_row[6] = kw_list[0]
        data_list_1.append(new_row)

# =============================================================================
# Data_1
# =============================================================================
target = pd.DataFrame(np.array([['Jon Favreau', 'Tom Hanks', 800_000_000, 
                                'Family', 'Brad Pitt', 'Spencer Breslin', 
                                'G', 10_000_000, 2001, 6.2],
                                ['Steven Spielberg', 'Sylvester Stallone', 1_000_000_000, 
                                'Comedy', 'Adam Sandler', 'Thomas F. Wilson', 
                                'R', 100_000, 1995, 5.6],
                                ['Christopher Nolan', 'Philip Seymour Hoffman', 9_000_000_000, 
                                'Crime', 'Christian Bale', 'Minnie Driver', 
                                'PG-13', 100_000_000, 2009, 8.9],
                                ['Kenneth Branagh', 'Leonardo DiCaprio', 1_000_000, 
                                'Drama', 'Leonardo DiCaprio', 'Colin Farrell', 
                                'PG-13', 100_000_000, 2018, 7.4]]))

print(target)        


# Convert to data pandaframe to make encoding easier
data_1 = pd.DataFrame(data_list_1, columns=data.columns.values)
# Make label encoders for data_1 for later reference
le_director_1 = LabelEncoder()
le_actor1_1 = LabelEncoder()
le_actor2_1 = LabelEncoder()
le_actor3_1 = LabelEncoder()
le_keywords_1 = LabelEncoder()
le_genres_1 = LabelEncoder()
le_rating_1 = LabelEncoder()

# Label Encode all data for data_1
le_director_1.fit(data_1.director_name.unique())
data_1.director_name = le_director_1.transform(data_1.director_name)
target.loc[:,0] = le_director_1.transform(target.loc[:,0])

le_actor1_1.fit(data_1.actor_1_name.unique())
data_1.actor_1_name = le_actor1_1.transform(data_1.actor_1_name)
target[4] = le_actor1_1.transform(target[4])

le_actor2_1.fit(data_1.actor_2_name.unique())
data_1.actor_2_name = le_actor2_1.transform(data_1.actor_2_name)
target[1] = le_actor2_1.transform(target[1])

le_actor3_1.fit(data_1.actor_3_name.unique())
data_1.actor_3_name = le_actor3_1.transform(data_1.actor_3_name)
target[5] = le_actor3_1.transform(target[5])

le_keywords_1.fit(data_1.plot_keywords.unique())
data_1.plot_keywords = le_keywords_1.transform(data_1.plot_keywords)

le_genres_1.fit(data_1.genres.unique())
data_1.genres = le_genres_1.transform(data_1.genres)
target[3] = le_genres_1.transform(target[3])

le_rating_1.fit(data_1.content_rating.unique())
data_1.content_rating = le_rating_1.transform(data_1.content_rating)
target[6] = le_rating_1.transform(target[6])

# Train Test Split
x = data_1.drop(columns=["plot_keywords"]).to_numpy()
y = data_1["plot_keywords"].to_numpy()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1)

# Standard ID3
id3 = tree.DecisionTreeClassifier()
id3.fit(x_train, y_train)
y_pred = id3.predict(x_test)
print("Accuracy 1: {}".format(accuracy_score(y_test, y_pred)))
y_target = id3.predict(target)
print(le_keywords_1.inverse_transform(y_target))
## AdaBoost
#ada = AdaBoostClassifier(n_estimators = 100)
#ada.fit(x_train, y_train)
#y_pred = ada.predict(x_test)
#accuracy = accuracy_score(y_test, y_pred)
#print("Ada Accuracy: {}".format(accuracy))
#y_target = ada.predict(target)
#print(le_keywords_1.inverse_transform(y_target))
## Bagging Classifier
#bag = BaggingClassifier(base_estimator=tree.DecisionTreeClassifier(),
#                        n_estimators=50, random_state=0).fit(x_train, y_train)
#y_pred = bag.predict(x_test)
#accuracy = accuracy_score(y_test, y_pred)
#print("Bagging Accuracy: {}".format(accuracy))
#y_target = bag.predict(target)
#print(le_keywords_1.inverse_transform(y_target))
# =============================================================================
# Data_2
# =============================================================================
target = pd.DataFrame(np.array([['Jon Favreau', 'Tom Hanks', 800_000_000, 
                                'Family', 'Brad Pitt', 'Spencer Breslin', 
                                'G', 10_000_000, 2001, 6.2],
                                ['Steven Spielberg', 'Sylvester Stallone', 1_000_000_000, 
                                'Comedy', 'Adam Sandler', 'Thomas F. Wilson', 
                                'R', 100_000, 1995, 5.6],
                                ['Christopher Nolan', 'Philip Seymour Hoffman', 9_000_000_000, 
                                'Crime', 'Christian Bale', 'Minnie Driver', 
                                'PG-13', 100_000_000, 2009, 8.9],
                                ['Kenneth Branagh', 'Leonardo DiCaprio', 1_000_000, 
                                'Drama', 'Leonardo DiCaprio', 'Colin Farrell', 
                                'PG-13', 100_000_000, 2018, 7.4]]))

# Convert to data pandaframe to make encoding easier
data_2 = pd.DataFrame(data_list_2, columns=data.columns.values)

# Make label encoders for data_2 for later reference
le_director_2 = LabelEncoder()
le_actor1_2 = LabelEncoder()
le_actor2_2 = LabelEncoder()
le_actor3_2 = LabelEncoder()
le_keywords_2 = LabelEncoder()
le_genres_2 = LabelEncoder()
le_rating_2 = LabelEncoder()

# Label Encode all data for data_2
le_director_2.fit(data_2.director_name.unique())
data_2.director_name = le_director_2.transform(data_2.director_name)
target.loc[:,0] = le_director_2.transform(target.loc[:,0])

le_actor1_2.fit(data_2.actor_1_name.unique())
data_2.actor_1_name = le_actor1_2.transform(data_2.actor_1_name)
target[4] = le_actor1_2.transform(target[4])

le_actor2_2.fit(data_2.actor_2_name.unique())
data_2.actor_2_name = le_actor2_2.transform(data_2.actor_2_name)
target[1] = le_actor2_2.transform(target[1])

le_actor3_2.fit(data_2.actor_3_name.unique())
data_2.actor_3_name = le_actor3_2.transform(data_2.actor_3_name)
target[5] = le_actor3_2.transform(target[5])

le_keywords_2.fit(data_2.plot_keywords.unique())
data_2.plot_keywords = le_keywords_2.transform(data_2.plot_keywords)

le_genres_2.fit(data_2.genres.unique())
data_2.genres = le_genres_2.transform(data_2.genres)
target[3] = le_genres_2.transform(target[3])

le_rating_2.fit(data_2.content_rating.unique())
data_2.content_rating = le_rating_2.transform(data_2.content_rating)
target[6] = le_rating_2.transform(target[6])

# Train Test Split
x = data_2.drop(columns=["plot_keywords"]).to_numpy()
y = data_2["plot_keywords"].to_numpy()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1)
        
# Standard ID3
id3 = tree.DecisionTreeClassifier()
id3.fit(x_train, y_train)
y_pred = id3.predict(x_test)
print("Accuracy 2: {}".format(accuracy_score(y_test, y_pred)))
y_target = id3.predict(target)
print(le_keywords_1.inverse_transform(y_target))
## AdaBoost
#ada = AdaBoostClassifier(n_estimators = 100)
#ada.fit(x_train, y_train)
#y_pred = ada.predict(x_test)
#accuracy = accuracy_score(y_test, y_pred)
#print("Ada Accuracy: {}".format(accuracy))
#y_target = ada.predict(target)
#print(le_keywords_1.inverse_transform(y_target))
## Bagging Classifier
#bag = BaggingClassifier(base_estimator=tree.DecisionTreeClassifier(),
#                        n_estimators=50, random_state=0).fit(x_train, y_train)
#y_pred = bag.predict(x_test)
#accuracy = accuracy_score(y_test, y_pred)
#print("Bagging Accuracy: {}".format(accuracy))
#y_target = bag.predict(target)
#print(le_keywords_1.inverse_transform(y_target))
# =============================================================================
# Data_3
# =============================================================================
target = pd.DataFrame(np.array([['Jon Favreau', 'Tom Hanks', 800_000_000, 
                                'Family', 'Brad Pitt', 'Spencer Breslin', 
                                'G', 10_000_000, 2001, 6.2],
                                ['Steven Spielberg', 'Sylvester Stallone', 1_000_000_000, 
                                'Comedy', 'Adam Sandler', 'Thomas F. Wilson', 
                                'R', 100_000, 1995, 5.6],
                                ['Christopher Nolan', 'Philip Seymour Hoffman', 9_000_000_000, 
                                'Crime', 'Christian Bale', 'Minnie Driver', 
                                'PG-13', 100_000_000, 2009, 8.9],
                                ['Kenneth Branagh', 'Leonardo DiCaprio', 1_000_000, 
                                'Drama', 'Leonardo DiCaprio', 'Colin Farrell', 
                                'PG-13', 100_000_000, 2018, 7.4]]))
    
# Convert to data pandaframe to make encoding easier
data_3 = pd.DataFrame(data_list_3, columns=data.columns.values)

# Make label encoders for data_3 for later reference
le_director_3 = LabelEncoder()
le_actor1_3 = LabelEncoder()
le_actor2_3 = LabelEncoder()
le_actor3_3 = LabelEncoder()
le_keywords_3 = LabelEncoder()
le_genres_3 = LabelEncoder()
le_rating_3 = LabelEncoder()

# Label Encode all data for data_3
le_director_3.fit(data_3.director_name.unique())
data_3.director_name = le_director_3.transform(data_3.director_name)
target.loc[:,0] = le_director_3.transform(target.loc[:,0])

le_actor1_3.fit(data_3.actor_1_name.unique())
data_3.actor_1_name = le_actor1_3.transform(data_3.actor_1_name)
target[4] = le_actor1_3.transform(target[4])

le_actor2_3.fit(data_3.actor_2_name.unique())
data_3.actor_2_name = le_actor2_3.transform(data_3.actor_2_name)
target[1] = le_actor2_3.transform(target[1])

le_actor3_3.fit(data_3.actor_3_name.unique())
data_3.actor_3_name = le_actor3_3.transform(data_3.actor_3_name)
target[5] = le_actor3_3.transform(target[5])

le_keywords_3.fit(data_3.plot_keywords.unique())
data_3.plot_keywords = le_keywords_3.transform(data_3.plot_keywords)

le_genres_3.fit(data_3.genres.unique())
data_3.genres = le_genres_3.transform(data_3.genres)
target[3] = le_genres_3.transform(target[3])

le_rating_3.fit(data_3.content_rating.unique())
data_3.content_rating = le_rating_3.transform(data_3.content_rating)
target[6] = le_rating_3.transform(target[6])

# Train Test Split
x = data_3.drop(columns=["plot_keywords"]).to_numpy()
y = data_3["plot_keywords"].to_numpy()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1)
        
# Standard ID3
id3 = tree.DecisionTreeClassifier()
id3.fit(x_train, y_train)
y_pred = id3.predict(x_test)
print("Accuracy 3: {}".format(accuracy_score(y_test, y_pred)))
y_target = id3.predict(target)
print(le_keywords_1.inverse_transform(y_target))
## AdaBoost
#ada = AdaBoostClassifier(n_estimators = 100)
#ada.fit(x_train, y_train)
#y_pred = ada.predict(x_test)
#accuracy = accuracy_score(y_test, y_pred)
#print("Ada Accuracy: {}".format(accuracy))
#y_target = ada.predict(target)
#print(le_keywords_1.inverse_transform(y_target))
## Bagging Classifier
#bag = BaggingClassifier(base_estimator=tree.DecisionTreeClassifier(),
#                        n_estimators=50, random_state=0).fit(x_train, y_train)
#y_pred = bag.predict(x_test)
#accuracy = accuracy_score(y_test, y_pred)
#print("Bagging Accuracy: {}".format(accuracy))
#y_target = bag.predict(target)
#print(le_keywords_1.inverse_transform(y_target))
## =============================================================================
# Data_4
# =============================================================================
target = pd.DataFrame(np.array([['Jon Favreau', 'Tom Hanks', 800_000_000, 
                                'Family', 'Brad Pitt', 'Spencer Breslin', 
                                'G', 10_000_000, 2001, 6.2],
                                ['Steven Spielberg', 'Sylvester Stallone', 1_000_000_000, 
                                'Comedy', 'Adam Sandler', 'Thomas F. Wilson', 
                                'R', 100_000, 1995, 5.6],
                                ['Christopher Nolan', 'Philip Seymour Hoffman', 9_000_000_000, 
                                'Crime', 'Christian Bale', 'Minnie Driver', 
                                'PG-13', 100_000_000, 2009, 8.9],
                                ['Kenneth Branagh', 'Leonardo DiCaprio', 1_000_000, 
                                'Drama', 'Leonardo DiCaprio', 'Colin Farrell', 
                                'PG-13', 100_000_000, 2018, 7.4]]))

# Convert to data pandaframe to make encoding easier
data_4 = pd.DataFrame(data_list_4, columns=data.columns.values)

# Make label encoders for data_4 for later reference
le_director_4 = LabelEncoder()
le_actor1_4 = LabelEncoder()
le_actor2_4 = LabelEncoder()
le_actor3_4 = LabelEncoder()
le_keywords_4 = LabelEncoder()
le_genres_4 = LabelEncoder()
le_rating_4 = LabelEncoder()

# Label Encode all data for data_4
le_director_4.fit(data_4.director_name.unique())
data_4.director_name = le_director_4.transform(data_4.director_name)
target.loc[:,0] = le_director_4.transform(target.loc[:,0])

le_actor1_4.fit(data_4.actor_1_name.unique())
data_4.actor_1_name = le_actor1_4.transform(data_4.actor_1_name)
target[4] = le_actor1_4.transform(target[4])

le_actor2_4.fit(data_4.actor_2_name.unique())
data_4.actor_2_name = le_actor2_4.transform(data_4.actor_2_name)
target[1] = le_actor2_4.transform(target[1])

le_actor3_4.fit(data_4.actor_3_name.unique())
data_4.actor_3_name = le_actor3_4.transform(data_4.actor_3_name)
target[5] = le_actor3_4.transform(target[5])

le_keywords_4.fit(data_4.plot_keywords.unique())
data_4.plot_keywords = le_keywords_4.transform(data_4.plot_keywords)

le_genres_4.fit(data_4.genres.unique())
data_4.genres = le_genres_4.transform(data_4.genres)
target[3] = le_genres_4.transform(target[3])

le_rating_4.fit(data_4.content_rating.unique())
data_4.content_rating = le_rating_4.transform(data_4.content_rating)
target[6] = le_rating_4.transform(target[6])

# Train Test Split
x = data_4.drop(columns=["plot_keywords"]).to_numpy()
y = data_4["plot_keywords"].to_numpy()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1)
        
# Standard ID3
id3 = tree.DecisionTreeClassifier()
id3.fit(x_train, y_train)
y_pred = id3.predict(x_test)
print("Accuracy 4: {}".format(accuracy_score(y_test, y_pred)))
y_target = id3.predict(target)
print(le_keywords_1.inverse_transform(y_target))
## AdaBoost
#ada = AdaBoostClassifier(n_estimators = 100)
#ada.fit(x_train, y_train)
#y_pred = ada.predict(x_test)
#accuracy = accuracy_score(y_test, y_pred)
#print("Ada Accuracy: {}".format(accuracy))
#y_target = ada.predict(target)
#print(le_keywords_1.inverse_transform(y_target))
## Bagging Classifier
#bag = BaggingClassifier(base_estimator=tree.DecisionTreeClassifier(),
#                        n_estimators=50, random_state=0).fit(x_train, y_train)
#y_pred = bag.predict(x_test)
#accuracy = accuracy_score(y_test, y_pred)
#print("Bagging Accuracy: {}".format(accuracy))
#y_target = bag.predict(target)
#print(le_keywords_1.inverse_transform(y_target))
# =============================================================================
# Data_5
# =============================================================================
target = pd.DataFrame(np.array([['Jon Favreau', 'Tom Hanks', 800_000_000, 
                                'Family', 'Brad Pitt', 'Spencer Breslin', 
                                'G', 10_000_000, 2001, 6.2],
                                ['Steven Spielberg', 'Sylvester Stallone', 1_000_000_000, 
                                'Comedy', 'Adam Sandler', 'Thomas F. Wilson', 
                                'R', 100_000, 1995, 5.6],
                                ['Christopher Nolan', 'Philip Seymour Hoffman', 9_000_000_000, 
                                'Crime', 'Christian Bale', 'Minnie Driver', 
                                'PG-13', 100_000_000, 2009, 8.9],
                                ['Kenneth Branagh', 'Leonardo DiCaprio', 1_000_000, 
                                'Drama', 'Leonardo DiCaprio', 'Colin Farrell', 
                                'PG-13', 100_000_000, 2018, 7.4]]))

# Convert to data pandaframe to make encoding easier
data_5 = pd.DataFrame(data_list_5, columns=data.columns.values)

# Make label encoders for data_5 for later reference
le_director_5 = LabelEncoder()
le_actor1_5 = LabelEncoder()
le_actor2_5 = LabelEncoder()
le_actor3_5 = LabelEncoder()
le_keywords_5 = LabelEncoder()
le_genres_5 = LabelEncoder()
le_rating_5 = LabelEncoder()

# Label Encode all data for data_5
le_director_5.fit(data_5.director_name.unique())
data_5.director_name = le_director_5.transform(data_5.director_name)
target.loc[:,0] = le_director_5.transform(target.loc[:,0])

le_actor1_5.fit(data_5.actor_1_name.unique())
data_5.actor_1_name = le_actor1_5.transform(data_5.actor_1_name)
target[4] = le_actor1_5.transform(target[4])

le_actor2_5.fit(data_5.actor_2_name.unique())
data_5.actor_2_name = le_actor2_5.transform(data_5.actor_2_name)
target[1] = le_actor2_5.transform(target[1])

le_actor3_5.fit(data_5.actor_3_name.unique())
data_5.actor_3_name = le_actor3_5.transform(data_5.actor_3_name)
target[5] = le_actor3_5.transform(target[5])

le_keywords_5.fit(data_5.plot_keywords.unique())
data_5.plot_keywords = le_keywords_5.transform(data_5.plot_keywords)

le_genres_5.fit(data_5.genres.unique())
data_5.genres = le_genres_5.transform(data_5.genres)
target[3] = le_genres_5.transform(target[3])

le_rating_5.fit(data_5.content_rating.unique())
data_5.content_rating = le_rating_5.transform(data_5.content_rating)
target[6] = le_rating_5.transform(target[6])

# Train Test Split
x = data_5.drop(columns=["plot_keywords"]).to_numpy()
y = data_5["plot_keywords"].to_numpy()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1)
        
# Standard ID3
id3 = tree.DecisionTreeClassifier()
id3.fit(x_train, y_train)
y_pred = id3.predict(x_test)
print("Accuracy 5: {}".format(accuracy_score(y_test, y_pred)))
y_target = id3.predict(target)
print(le_keywords_1.inverse_transform(y_target))
## AdaBoost
#ada = AdaBoostClassifier(n_estimators = 100)
#ada.fit(x_train, y_train)
#y_pred = ada.predict(x_test)
#accuracy = accuracy_score(y_test, y_pred)
#print("Ada Accuracy: {}".format(accuracy))
#y_target = ada.predict(target)
#print(le_keywords_1.inverse_transform(y_target))
## Bagging Classifier
#bag = BaggingClassifier(base_estimator=tree.DecisionTreeClassifier(),
#                        n_estimators=50, random_state=0).fit(x_train, y_train)
#y_pred = bag.predict(x_test)
#accuracy = accuracy_score(y_test, y_pred)
#print("Bagging Accuracy: {}".format(accuracy))
#y_target = bag.predict(target)
#print(le_keywords_1.inverse_transform(y_target))