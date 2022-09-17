import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def add_missing_rows_and_columns(matrix, columnIds, rowIds):
    for columnId in columnIds:
        if columnId not in matrix.columns:
            matrix[columnId] = np.nan
    for rowId in rowIds:
        if rowId not in matrix.index:
            matrix.loc[rowId]=np.nan
    return matrix

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