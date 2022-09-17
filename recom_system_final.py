import numpy as np
import pandas as pd
import time
import argparse
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF as NMF_sklearn




############### ARGPARSE
def ParseArguments():
    parser = argparse.ArgumentParser(description="Project")
    parser.add_argument('--train', default="", required=True, help='train data file (default: %(default)s)')
    parser.add_argument('--test', default="", required=True, help='test data file (default: %(default)s)')
    parser.add_argument('--alg', default="SVD1", required=True, help='which algorithm to run (default: %(default)s)')
    parser.add_argument('--result', default="", required=True, help='output file (default: %(default)s)')
    args = parser.parse_args()

    return args.train, args.test, args.alg, args.result


train, test, alg, result_file = ParseArguments()

################ DEFINING FUNCTIONS

######## data preparation


def add_missing_rows_and_columns(matrix, columnIds, rowIds):
    for columnId in columnIds:
        if columnId not in matrix.columns:
            matrix[columnId] = np.nan
    for rowId in rowIds:
        if rowId not in matrix.index:
            matrix.loc[rowId]=np.nan

    matrix_new = matrix.copy()
    return matrix_new

class Data_preparation:
    def __init__(self, debug=True):
        self.debug = debug
        self.train_with_validation_rating_matrix = None
        self.validation_rating_matrix = None 

    def log(self, *message):
        if self.debug:
            print(*message)

    def read_input_file(self, inputfile):
        self.movie_rating = pd.read_csv(inputfile)
        self.userIds = self.movie_rating["userId"].unique()
        self.movieIds = self.movie_rating["movieId"].unique()
        # logging
        self.log("UserIds (shape, min, max)", self.userIds.shape, self.userIds.min(), self.userIds.max())
        self.log("movieIds (shape, min, max)", self.movieIds.shape, self.movieIds.min(), self.movieIds.max())
        
    def split_train_test(self, test_size=0.1, random_state=None):
        self.train, self.test = train_test_split(self.movie_rating, test_size=test_size, stratify=self.movie_rating["userId"], random_state=random_state)
        # logging
        self.log("Trainset: ", self.train.shape, " Testset: ", self.test.shape)

    def split_train_valid(self, test_size=0.1, random_state=None):
        self.train_with_validation, self.validation = train_test_split(self.train, test_size=test_size, stratify=self.train["userId"], random_state=random_state)
        self.log("Trainset with val: ", self.train_with_validation.shape, " Validation: ", self.validation.shape)

    def save_train_test(self, trainfile_path, testfile_path):
        self.train.to_csv(trainfile_path, index=False)
        self.test.to_csv(testfile_path, index=False)
        self.log(f"Trainset saved in {trainfile_path}")
        self.log(f"Testset saved in {testfile_path}")

    # inputfile - path for the file with movielens data (but any csv data with userId, movieId, rating columns should work)
    # trainfile, testfile - paths for saving processed data
    def split_and_save(self, inputfile, trainfile_path= "data/training_ratings.csv" , testfile_path="data/test_ratings.csv", test_size=0.1, random_state=None):
        self.read_input_file(inputfile)
        self.split_train_test(test_size, random_state= random_state)
        self.save_train_test(trainfile_path, testfile_path)

    def comupte_user_and_movie_ids(self):
        combined = pd.concat([self.train, self.test])
        self.userIds = combined["userId"].unique()
        self.movieIds = combined["movieId"].unique()
        self.log("UserIds (shape, min, max)", self.userIds.shape, self.userIds.min(), self.userIds.max())
        self.log("movieIds (shape, min, max)", self.movieIds.shape, self.movieIds.min(), self.movieIds.max())

    def read_test_train_files(self, trainfile_path, testfile_path):
        self.train = pd.read_csv(trainfile_path)
        self.test = pd.read_csv(testfile_path)
        self.comupte_user_and_movie_ids()
        #logging
        self.log("Trainset: ", self.train.shape, " Testset: ", self.test.shape)


    def create_matrices(self):
        self.log("creating matrices...")
        self.train_rating_matrix = self.train.pivot(index = 'userId', columns='movieId', values='rating')
        self.test_rating_matrix = self.test.pivot(index = 'userId', columns='movieId', values='rating')
        self.log("Train_matrix", self.train_rating_matrix.shape)
        self.log("Test matrix", self.test_rating_matrix.shape)

    def create_matrices_validation(self):
        self.log("creating matrices...")
        self.train_with_validation_rating_matrix = self.train_with_validation.pivot(index = 'userId', columns='movieId', values='rating')
        self.validation_rating_matrix = self.validation.pivot(index = 'userId', columns='movieId', values='rating')
        self.log("Train_matrix", self.train_with_validation_rating_matrix.shape)
        self.log("Test matrix", self.validation_rating_matrix.shape)

    def add_missing_ids(self):
        self.log("Adding missing ids...")
        add_missing_rows_and_columns(self.train_rating_matrix, self.movieIds, self.userIds)
        add_missing_rows_and_columns(self.test_rating_matrix, self.movieIds, self.userIds)
        self.log("Sorting ids...")
        self.train_rating_matrix.sort_index(axis=1, inplace=True)
        self.test_rating_matrix.sort_index(axis=1, inplace=True)
        self.train_rating_matrix.sort_index(axis=0, inplace=True)
        self.test_rating_matrix.sort_index(axis=0, inplace=True)
        self.log("Train_matrix", self.train_rating_matrix.shape)
        self.log("Test matrix", self.test_rating_matrix.shape)

    def add_missing_ids_validation(self):
        self.log("Adding missing ids...")
        add_missing_rows_and_columns(self.train_with_validation_rating_matrix, self.movieIds, self.userIds)
        add_missing_rows_and_columns(self.validation_rating_matrix, self.movieIds, self.userIds)
        self.log("Sorting ids...")
        self.train_with_validation_rating_matrix.sort_index(axis=1, inplace=True)
        self.validation_rating_matrix.sort_index(axis=1, inplace=True)
        self.train_with_validation_rating_matrix.sort_index(axis=0, inplace=True)
        self.validation_rating_matrix.sort_index(axis=0, inplace=True)
        self.log("Train_with_validation_matrix", self.train_with_validation_rating_matrix.shape)
        self.log("Validation matrix", self.validation_rating_matrix.shape)

    def prepare_validation_matrices(self):
        if self.validation_rating_matrix is None:
            self.split_train_valid()
            self.create_matrices_validation()
            self.add_missing_ids_validation()

    def prepare_matrices(self):
        self.create_matrices()
        self.add_missing_ids()

    def get_train_matrix(self):
        return self.train_rating_matrix

    def get_test_matrix(self):
        return self.test_rating_matrix

    def get_train_with_validation(self):
        if self.train_with_validation_rating_matrix is None:
            self.prepare_validation_matrices()
        return self.train_with_validation_rating_matrix

    def get_validation(self):
        if self.validation_rating_matrix is None:
            self.prepare_validation_matrices()
        return self.validation_rating_matrix

############# nmf

def NMF(Z, r, random_state = None, initial_filling_method = lambda x : fill_weighted_mean(x, 0.55), init=None, max_iter=200):
    filled_Z = initial_filling_method(Z)
    model = NMF_sklearn(n_components=r, init=init, random_state=random_state, max_iter=max_iter)
    W = model.fit_transform(filled_Z)
    H = model.components_
    Z_prediction = np.dot(W, H)
    return Z_prediction

############### svd1 and svd2

# Z is filled rating matrix
def SVD_step(Z, r, random_state=None):
    svd = TruncatedSVD(n_components=r, random_state=random_state)
    svd.fit(Z)
    U_Sigma = svd.transform(Z)
    VT=svd.components_
    Z_prediction = U_Sigma @ VT
    return Z_prediction

def SVD(Z, r, random_state, initial_filling_method = lambda x : fill_weighted_mean(x, 0.63)):
    filled_Z = initial_filling_method(Z)
    result = SVD_step(filled_Z, r, random_state)
    return result
    
def SVD2_validation(Z_train_v, Z_validation, r, random_state = None, initial_filling_method = lambda x : fill_weighted_mean(x, 0.75), max_iter=25, log=False):
    Z_filled = initial_filling_method(Z_train_v)
    Z_previous_prediction = None
    previous_rmse = np.inf
    for i in range(max_iter):
        Z_prediction = SVD_step(Z_filled, r, random_state=random_state)
        assert Z_train_v.shape == Z_prediction.shape
        Z_filled = Z_train_v.fillna(pd.DataFrame(Z_prediction, index=Z_filled.index, columns=Z_filled.columns)) # zaobaczyć czy to nie musi być DF zamiast numpy
        rmse = calc_RMSE(Z_validation, Z_prediction)
        if log: print(f"{i}: {rmse}")
        if (rmse > (previous_rmse + 1e-05)) or abs(rmse-previous_rmse) < 1e-05:
            return Z_previous_prediction
        Z_previous_prediction = Z_prediction
        previous_rmse = rmse
    return Z_prediction

def SVD2_validation_from_data(data, r, random_state = None, initial_filling_method = lambda x : fill_weighted_mean(x, 0.75) , max_iter=25, log=False):
    Z_train_v = data.get_train_with_validation()
    Z_validation = data.get_validation() 
    return SVD2_validation(Z_train_v, Z_validation, r, random_state = random_state, initial_filling_method = initial_filling_method, max_iter=max_iter, log=log)

############### SGD
def prepare_df(data):
    userIds = np.sort(data.userIds)
    movieIds = np.sort(data.movieIds)

    userMapping = dict(zip(userIds, np.arange(userIds.size, dtype=int)))
    movieMapping = dict(zip(movieIds, np.arange(movieIds.size, dtype=int)))

    train_prepared = data.train.replace({"userId":userMapping, "movieId":movieMapping})
    test_prepared = data.test.replace({"userId":userMapping, "movieId":movieMapping})

    #train.info()

    train_prepared = train_prepared[["userId", "movieId", "rating"]].values
    test_prepared = test_prepared[["userId", "movieId", "rating"]].values
    return train_prepared, test_prepared, userMapping, movieMapping

#from numba import jit 
#@jit
def SGD_epoch_step(train_permuted, W, H, learning_rate=0.01, lambda_reg = 0.1):
    for sample in train_permuted:
        userId, movieId, rating = int(sample[0]), int(sample[1]), sample[2]
        prediction = np.dot(W[userId], H[:,movieId])
        difference = rating - prediction
        W[userId, :] = (W[userId, :] + learning_rate *(difference*H[:, movieId] - lambda_reg * W[userId, :]))
        H[:, movieId] = H[:, movieId] + learning_rate *(difference*W[userId, :] - lambda_reg * H[:, movieId])
    return W, H

from sklearn.utils import shuffle
def SGD(train, n, d, learning_rate, r, lambda_reg, max_epochs, W=None, H=None):
    if W is None:
        W = np.random.random((n,r))
    if H is None:
        H = np.random.random((r,d))
    for epoch in range(max_epochs):
        train_permuted = shuffle(train)
        W, H = SGD_epoch_step(train_permuted, W, H, learning_rate=learning_rate, lambda_reg=lambda_reg)
    return W @ H

def SGD_from_data(data, learning_rate, r, lambda_reg, max_epochs):
    train_prepared, test_prepared, userMapping, movieMapping = prepare_df(data)
    return SGD(train_prepared, len(userMapping), len(movieMapping), learning_rate, r, lambda_reg, max_epochs)


############### matrix filling 

def fill_weighted_mean(df, row_weight):
    mean = np.nanmean(df)
    col_mean = df.mean(axis = 0).to_numpy()[np.newaxis]
    col_mean[np.isnan(col_mean)] = mean
    row_mean = df.mean(axis = 1).to_numpy()[np.newaxis].T
    row_mean[np.isnan(row_mean)] = mean
    fill = (1-row_weight) * col_mean + row_weight * row_mean
    df_filled = df.fillna(pd.DataFrame(fill, index = df.index, columns=df.columns))
    df_filled_2 = df_filled.copy()
    return df_filled_2

############# calculating rmse 

def calc_RMSE(true_test_mat, pred_test_mat): # it must be given numpy matrices and not pandas data frames !
    true_test_mat = np.asarray(true_test_mat)
    pred_test_mat = np.asarray(pred_test_mat)
    assert true_test_mat.shape == pred_test_mat.shape
    # T_true = true_test_mat.to_numpy()
    mask = ~np.isnan(true_test_mat)
    T = mask.sum()
    RMSE = np.sqrt(((true_test_mat[mask] - pred_test_mat[mask])**2).sum()/T)
    return(RMSE)

########### script
data = Data_preparation(debug = False)
data.read_test_train_files(trainfile_path = train, testfile_path = test)
data.comupte_user_and_movie_ids()
data.prepare_matrices()
train_mat = data.get_train_matrix()
test_mat = data.get_test_matrix()

if alg == "SVD1":
    svd1 = SVD(Z = train_mat, r = 16, random_state = 0, initial_filling_method = lambda x : fill_weighted_mean(x, 0.63))
    rmse_res = calc_RMSE(test_mat,svd1)
elif alg == "SVD2":
    svd2 = SVD2_validation_from_data(data = data, r = 2, random_state = 0, initial_filling_method = lambda x : fill_weighted_mean(x, 0.75), max_iter=2000, log=False)
    rmse_res = calc_RMSE(test_mat, svd2)
elif alg == "NMF":
    nmf = NMF(Z = train_mat, r = 25, random_state = 0, initial_filling_method = lambda x : fill_weighted_mean(x, 0.55), max_iter=1000, init='nndsvda')
    rmse_res = calc_RMSE(test_mat,nmf)
elif alg == "SGD":
    sgd = SGD_from_data(data,  learning_rate=0.005, r=15, lambda_reg=0.1, max_epochs=100)
    rmse_res = calc_RMSE(test_mat,sgd)

outF = open(result_file, "w")
outF.write(str(rmse_res))
outF.close()
## plik result ma się tworzyć nowy w open w+