import numpy as np

def calc_RMSE(true_test_mat, pred_test_mat): # it must be given numpy matrices and not pandas data frames !
    true_test_mat = np.asarray(true_test_mat)
    pred_test_mat = np.asarray(pred_test_mat)
    assert true_test_mat.shape == pred_test_mat.shape
    # T_true = true_test_mat.to_numpy()
    mask = ~np.isnan(true_test_mat)
    T = mask.sum()
    RMSE = np.sqrt(((true_test_mat[mask] - pred_test_mat[mask])**2).sum()/T)
    return(RMSE)

def round_to_ratings(matrix):
    rounded_matrix = round(matrix * 2) / 2
    # TODO if needed: add casting values below 1 to 1 and above 5 to 5
    return rounded_matrix

def calc_accuracy(Z, Z_prediction):
    Z_rounded_prediction = round_to_ratings(Z_prediction)
    mask = ~np.isnan(Z)
    T = mask.sum()
    return (Z_rounded_prediction[mask]==Z[mask]).sum()/T
