import time
#!pip install numba
from numba import jit # SPEEED!

def prepare_df(train, test):
    combined = pd.concat([train, test])
    userIds = np.sort(combined["userId"].unique())
    movieIds = np.sort(combined["movieId"].unique())

    userMapping = dict(zip(userIds, np.arange(userIds.size, dtype=int)))
    movieMapping = dict(zip(movieIds, np.arange(movieIds.size, dtype=int)))

    train = train.replace({"userId":userMapping, "movieId":movieMapping})
    test = test.replace({"userId":userMapping, "movieId":movieMapping})

    train.info()

    train = train[["userId", "movieId", "rating"]].values
    test = test[["userId", "movieId", "rating"]].values
    return train, test, userMapping, movieMapping

@jit
def SGD_epoch_step(train_permuted, W, H, learning_rate=0.01, lambda_reg = 0.1):
    for sample in train_permuted:
        userId, movieId, rating = int(sample[0]), int(sample[1]), sample[2]
        prediction = np.dot(W[userId], H[:,movieId])
        #print(prediction, prediction2, sample)
        difference = rating - prediction
        W[userId, :] = (W[userId, :] + learning_rate *(difference*H[:, movieId] - lambda_reg * W[userId, :]))
        H[:, movieId] = H[:, movieId] + learning_rate *(difference*W[userId, :] - lambda_reg * H[:, movieId])
    return W, H

def SGD(train, n, d, learning_rate, r, lambda_reg, max_epochs, W=None, H=None, log=5, Z_test=Z_test, Z_train=Z_train):
    start_time = time.time()
    if W is None:
        W = np.random.random((n,r))
    if H is None:
        H = np.random.random((r,d))
    #learning_rate = learning_rate/train.shape[0]
    epoch_log = []
    train_rmse_log = []
    test_rmse_log =[]
    time_log = []
    for epoch in range(max_epochs):
        #train_permuted = train
        train_permuted = shuffle(train)
        #aprox_loss = 0
        W, H = SGD_epoch_step(train_permuted, W, H, learning_rate=learning_rate, lambda_reg=lambda_reg)
        #print(epoch, np.sqrt(aprox_loss))
        if epoch%log == log-1:
            Z_pred_SGD = W @ H
            test_rmse = calc_RMSE(Z_test, Z_pred_SGD)
            train_rmse = calc_RMSE(Z_train, Z_pred_SGD)
            enlapsed_time = time.time() - start_time
            epoch_log.append(epoch)
            test_rmse_log.append(test_rmse)
            train_rmse_log.append(train_rmse)
            time_log.append(enlapsed_time)
            #print(epoch, test_rmse, train_rmse, enlapsed_time)
            if np.isnan(test_rmse) or np.isnan(train_rmse):
                print("Nan encounter")
                break
    return W, H , pd.DataFrame({"epoch":epoch_log, "train_rmse":train_rmse_log, "test_rmse":test_rmse_log, "time":time_log})