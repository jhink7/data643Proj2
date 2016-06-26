from math import*
import pandas as pd
import numpy as np
import random as rd


movie_df = pd.DataFrame(columns=['id', 'title'])
for i in range(1, 101):
    movie_df.loc[len(movie_df)] = [i, 'title' + str(i)]

movie_df[['id']] = movie_df[['id']].astype(int)
#movie_df.to_csv('data/new_movies.csv', index=False)

all_users = pd.read_csv('data/users.csv')

#print all_users

rating_df_train = pd.DataFrame(columns=['id','user_id','movie_id','rating'])
rating_df_eval = pd.DataFrame(columns=['id','user_id','movie_id','rating'])

rat_index = 1
for user in all_users.id.values:
    for movie in movie_df.id.values:
        rand = rd.random()

        if rand < 0.4:
            rating = 5 - abs(user - movie % 10) /2
            rating_df_train.loc[len(rating_df_train)] = [rat_index, user, movie, rating]
            rat_index = rat_index + 1
        elif rand >= 0.4 and rand < 0.5:
            rating = 5 - abs(user - movie % 10) / 2
            rating_df_eval.loc[len(rating_df_train)] = [rat_index, user, movie, rating]
            rat_index = rat_index + 1

rating_df_train[['id', 'user_id', 'movie_id', 'rating']] = rating_df_train[['id', 'user_id', 'movie_id', 'rating']].astype(int)
rating_df_train.to_csv('data/ratings_training.csv', index=False)

rating_df_eval[['id', 'user_id', 'movie_id', 'rating']] = rating_df_eval[['id', 'user_id', 'movie_id', 'rating']].astype(int)
rating_df_eval.to_csv('data/ratings_eval.csv', index=False)