from math import*
import pandas as pd
import numpy as np
import random as rd

from recommendation_engine import RecommendationEngine

engine = RecommendationEngine()
#found, recs = engine.generate_recommendations(7)

score = engine.predict_rating(8,96, sim_mode='jac', rttm=0.5)

eval_ratings = pd.read_csv('data/ratings_eval.csv')

m1_preds = []
m2_preds = []
m3_preds = []
m4_preds = []
m5_preds = []
m6_preds = []
m7_preds = []
m8_preds = []
m9_preds = []
for eval in eval_ratings.values:
    m1_hat = engine.predict_rating(eval[1],eval[2], sim_mode='comb', rttm=0)
    m1_preds.append(m1_hat)

    m2_hat = engine.predict_rating(eval[1],eval[2], sim_mode='jac', rttm=0)
    m2_preds.append(m2_hat)

    m3_hat = engine.predict_rating(eval[1],eval[2], sim_mode='cos', rttm=0)
    m3_preds.append(m3_hat)

    m4_hat = engine.predict_rating(eval[1], eval[2], sim_mode='comb', rttm=0.25)
    m4_preds.append(m4_hat)

    m5_hat = engine.predict_rating(eval[1], eval[2], sim_mode='jac', rttm=0.25)
    m5_preds.append(m5_hat)

    m6_hat = engine.predict_rating(eval[1], eval[2], sim_mode='cos', rttm=0.25)
    m6_preds.append(m6_hat)

    m7_hat = engine.predict_rating(eval[1], eval[2], sim_mode='comb', rttm=0.5)
    m7_preds.append(m7_hat)

    m8_hat = engine.predict_rating(eval[1], eval[2], sim_mode='jac', rttm=0.5)
    m8_preds.append(m8_hat)

    m9_hat = engine.predict_rating(eval[1], eval[2], sim_mode='cos', rttm=0.5)
    m9_preds.append(m9_hat)

eval_ratings['m1_hat'] = pd.Series(np.asarray(m1_preds), index=eval_ratings.index)
eval_ratings['m2_hat'] = pd.Series(np.asarray(m2_preds), index=eval_ratings.index)
eval_ratings['m3_hat'] = pd.Series(np.asarray(m3_preds), index=eval_ratings.index)
eval_ratings['m4_hat'] = pd.Series(np.asarray(m4_preds), index=eval_ratings.index)
eval_ratings['m5_hat'] = pd.Series(np.asarray(m5_preds), index=eval_ratings.index)
eval_ratings['m6_hat'] = pd.Series(np.asarray(m6_preds), index=eval_ratings.index)
eval_ratings['m7_hat'] = pd.Series(np.asarray(m7_preds), index=eval_ratings.index)
eval_ratings['m8_hat'] = pd.Series(np.asarray(m8_preds), index=eval_ratings.index)
eval_ratings['m9_hat'] = pd.Series(np.asarray(m9_preds), index=eval_ratings.index)

eval_ratings.to_csv('out/evals.csv', index=False)

