import pandas as pd
import seaborn as sns
import pandas as pd
from surprise import SVD
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import train_test_split
from surprise import KNNBasic,  KNNWithMeans, KNNBaseline
from surprise.model_selection import KFold
from surprise import Reader
from surprise import NormalPredictor
from surprise.model_selection import cross_validate
import matplotlib.pyplot as plt
import seaborn as sns
from surprise.model_selection import GridSearchCV
from dataset import fetch_ml_ratings
from dataset import VARIANTS

variant = 'ml1m'

print("Obtaining data...")
ratings_ds = fetch_ml_ratings(variant=variant)

USER_ID_COLUMN = 'u_id'
ITEM_ID_COLUMN = 'i_id'
RATING_COLUMN = 'rating'
 
print(ratings_ds)
reader = Reader(rating_scale=(1, 5))
# The columns must correspond to user id, item id and ratings (in that order).
data = Dataset.load_from_df(ratings_ds[[USER_ID_COLUMN, ITEM_ID_COLUMN, RATING_COLUMN]], reader)

print("ok")

movies = ratings_ds[[ITEM_ID_COLUMN]].drop_duplicates([ITEM_ID_COLUMN])
users = ratings_ds[[USER_ID_COLUMN]].drop_duplicates([USER_ID_COLUMN])

print("ok")

print("MSD")

print("KNNBasic")

kf = KFold(n_splits=3)
algo = KNNBasic()
best_algo = None
best_rmse = 1000.0
best_pred = None
for trainset, testset in kf.split(data):
    # train and test algorithm.
    algo.fit(trainset)
    predictions = algo.test(testset)
    # Compute and print Root Mean Squared Error
    rmse = accuracy.rmse(predictions, verbose=True)
    if rmse < best_rmse:
        best_algo = algo
        best_rmse= rmse
        best_pred = predictions
    pass
pass

print("ok")
print(f"best RMSE {best_rmse}")

print("KNNWithMeans")

kf = KFold(n_splits=5)
sim_options = {'name':'cosine'}
algo = KNNWithMeans(sim_options = sim_options)
best_algo = None
best_rmse = 1000.0
best_pred = None
for trainset, testset in kf.split(data):
    # train and test algorithm.
    algo.fit(trainset)
    predictions = algo.test(testset)
    # Compute and print Root Mean Squared Error
    rmse = accuracy.rmse(predictions, verbose=True)
    if rmse < best_rmse:
        best_algo = algo
        best_rmse= rmse
        best_pred = predictions
    pass
pass
print(best_rmse)

print("ok")
print(f"best RMSE {best_rmse}")

print("KNNBaseline")


kf = KFold(n_splits=3)
algo = KNNBaseline(k=3)
best_algo = None
best_rmse = 1000.0
best_pred = None
for trainset, testset in kf.split(data):
    # train and test algorithm.
    algo.fit(trainset)
    predictions = algo.test(testset)
    # Compute and print Root Mean Squared Error
    rmse = accuracy.rmse(predictions, verbose=True)
    if rmse < best_rmse:
        best_rmse = rmse
        best_algo = algo
        best_pred = predictions
    pass
pass

print("ok")
print(f"best RMSE {best_rmse}")

print("cosine")

sim_options = { 'name': 'cosine' ,'user_based':  False}
kf = KFold(n_splits=5)
algo = KNNWithMeans(k =3 , sim_options = sim_options)
best_algo = None
best_rmse = 1000.0
best_pred = None
for trainset, testset in kf.split(data):
    # train and test algorithm.
    algo.fit(trainset)
    predictions = algo.test(testset)
    # Compute and print Root Mean Squared Error
    rmse = accuracy.rmse(predictions, verbose=True)
    if rmse < best_rmse:
        best_rmse= rmse
        best_algo = algo
        best_pred = predictions
    pass
pass

print("ok")
print(f"best RMSE {best_rmse}")
