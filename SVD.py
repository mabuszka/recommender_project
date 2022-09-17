from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF as NMF_sklearn
from scripts.fill_matrix import *
from scripts.model_evaluation import *
import pandas as pd
import numpy as np

##########################################################################################
## Termination conditions for SVD2:
# Termination class should have a check function which returns true 
# if algoritm should continue to work, and false if termination condition is met
##########################################################################################

# stops after set amount of iterations
class TerminationAfterIterations:
    def __init__(self, n):
        self.n = n
        self.current = 0

    def check(self):
        self.current +=1
        return self.current <= self.n

    def update_variables(self, *args):
        pass

    def get_iterations(self):
        return self.current

# stops when predictions in following iterations are not changing much  
class TerminationWhenPredictionsClose:
    def __init__(self, rtol=1e-05, atol=1e-08, max_iter=25, percentage=1.0):
        self.iter_termination = TerminationAfterIterations(max_iter)
        self.Z_filled_matrices = []
        self.Z_prediction_matrices = []
        self.updates = 0
        self.rtol = rtol
        self.atol = atol
        self.percentage = percentage
        self.current = None
        self.previous = None

    def check_closeness(self):
        if self.updates >= 3:
            previous_prediction = self.previous#self.Z_prediction_matrices[-2]
            current_prediction = self.current#self.Z_prediction_matrices[-1]
            return np.isclose(previous_prediction, current_prediction, self.rtol, self.atol).sum() >= current_prediction.size * self.percentage
        else:
            return True

    def update_variables(self, Z_filled, Z_prediction, *args):
        self.updates += 1
        #self.Z_filled_matrices.append(Z_filled)
        #self.Z_prediction_matrices.append(Z_prediction)
        self.previous = self.current
        self.current = Z_prediction
        if self.current is not None:
            print(calc_RMSE(Z_test.to_numpy(copy = True), self.current), calc_RMSE(Z.to_numpy(copy = True), self.current))

    def check(self):
        return self.iter_termination.check() and self.check_closeness()

    def get_iterations(self):
        return self.iter_termination.get_iterations()

#class TerminationByRMSE:
#    def __init__(self)

# stops when prediction is sufficiently close to original Z on filled places
## this condition is much less likely to be met, especially for smaller r
## i think it may also encourage overfitting, i am not sure if it is true tho
class TerminationWhenCloseToZ:
    def __init__(self, Z, rtol=1e-05, atol=1e-08, max_iter=1000):
        self.iter_termination = TerminationAfterIterations(max_iter)
        self.Z_filled_matrices = []
        self.Z_prediction_matrices = []
        self.Z = Z
        self.mask = ~np.isnan(Z)
        self.updates = 0
        self.rtol = rtol
        self.atol = atol

    def check_closeness(self):
        if self.updates >= 2:
            Z_filled = self.Z_filled_matrices[-1]
            return not np.isclose(Z[mask], Z_filled[mask], self.rtol, self.atol).all()
        else:
            return True

    def update_variables(self, Z_filled, Z_prediction, *args):
        self.updates += 1
        self.Z_filled_matrices.append(Z_filled)
        self.Z_prediction_matrices.append(Z_prediction)

    def check(self):
        return self.iter_termination.check() and self.check_closeness()

    def get_iterations(self):
        return self.iter_termination.get_iterations()


# other ideas:
# - stop when matrix is not changing much
# -  stop when RMSE starts to grow (or grows by something not very small?)

# following methods return predictions for all pairs user-movie in matrix Z_prediction
# initial_filling_method(matrix) returns filled matrix - it should not modify original matrix!
# You can use methods from fill matrix that take additional arguments by using lambda expression

######################################################
##  SVD, SVD2
######################################################

# Z is filled rating matrix
def SVD_step(Z, r, random_state=None):
    svd = TruncatedSVD(n_components=r, random_state=random_state)
    svd.fit(Z)
    U_Sigma = svd.transform(Z)
    VT=svd.components_
    Z_prediction = U_Sigma @ VT
    return Z_prediction

def SVD(Z, r, random_state, initial_filling_method = fill_all_mean):
    filled_Z = initial_filling_method(Z)
    result = SVD_step(filled_Z, r, random_state)
    return result
    
# termination condition is class with check and update_variables metchods
def SVD2(Z, r, random_state=None, termination_condition=TerminationAfterIterations(5),initial_filling_method = lambda x : fill_all_stat(x, "mean")):
    Z_filled = initial_filling_method(Z)
    termination_condition.update_variables(Z_filled, None)
    while(termination_condition.check()):
        Z_prediction = SVD_step(Z_filled, r, random_state=random_state)
        assert Z.shape == Z_prediction.shape
        Z_filled = Z.fillna(pd.DataFrame(Z_prediction, index=Z.index, columns=Z.columns)) # zaobaczyć czy to nie musi być DF zamiast numpy
        termination_condition.update_variables(Z_filled, Z_prediction)
    return Z_prediction

def SVD2_validation(Z_train_v, Z_validation, r, random_state = None, initial_filling_method = lambda x : fill_all_stat(x, "mean"), max_iter=25, log=True):
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

def SVD2_validation_from_data(data, r, random_state = None, initial_filling_method = lambda x : fill_all_stat(x, "mean"), max_iter=25, log=True):
    Z_train_v = data.get_train_with_validation()
    Z_validation = data.get_validation() 
    return SVD2_validation(Z_train_v, Z_validation, r, random_state = random_state, initial_filling_method = initial_filling_method, max_iter=max_iter, log=log)

#from data_split import add_missing_rows_and_columns

def SVD2_validation_from_trainset(trainset, testset, r, random_state = None, initial_filling_method = lambda x : fill_all_stat(x, "mean"), max_iter=25, log=True, validation_size=0.1):
    train_with_validation, validation = train_test_split(trainset, test_size=validation_size, stratify=trainset["userId"], random_state=random_state)
    train_with_validation_rating_matrix = train_with_validation.pivot(index = 'userId', columns='movieId', values='rating')
    validation_rating_matrix = validation.pivot(index = 'userId', columns='movieId', values='rating')

    combined = pd.concat([trainset, testset])
    userIds = combined["userId"].unique()
    movieIds = combined["movieId"].unique()

    add_missing_rows_and_columns(train_with_validation_rating_matrix, movieIds, userIds)
    add_missing_rows_and_columns(validation_rating_matrix, movieIds, userIds)
    train_with_validation_rating_matrix.sort_index(axis=1, inplace=True)
    validation_rating_matrix.sort_index(axis=1, inplace=True)
    train_with_validation_rating_matrix.sort_index(axis=0, inplace=True)
    validation_rating_matrix.sort_index(axis=0, inplace=True)

    return SVD2_validation(train_with_validation_rating_matrix, validation_rating_matrix, r, random_state = random_state, initial_filling_method = initial_filling_method, max_iter=max_iter, log=log)

####################################################
## NMF
####################################################

def NMF(Z, r, random_state = None, initial_filling_method = fill_all_mean, init=None, max_iter=200):
    filled_Z = initial_filling_method(Z)
    model = NMF_sklearn(n_components=r, init=init, random_state=random_state, max_iter=max_iter)
    W = model.fit_transform(filled_Z)
    H = model.components_
    Z_prediction = np.dot(W, H)
    return Z_prediction

