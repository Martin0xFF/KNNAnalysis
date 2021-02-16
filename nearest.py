import numpy as np
def shuffle(a, b):
    """
    """
    rng_state = np.random.get_state() # rng is based on a state machine
    np.random.shuffle(a)              # use first state here
    np.random.set_state(rng_state)    # reset state
    np.random.shuffle(b)              # use first state again
    np.random.set_state(rng_state)    # reset state

class knn():
    def __init__(self, dist_type= 'l2', k= 5):
        self.k = k
        self.pred = None
        self.dist_type = dist_type
                                   
    def batch_re(self, x_train, y_train, x_test):
        """
        Batch regression, give it the model data (features and targets) and the 'text' feature data
        It will return predicted values
        x_train - model feature data, used for prediction, should be a matrix
        y_train - model target data, used for prediction, should be a matrix
        x_test - test feature data, used for prediction, should be a matrix
        
        """
        dim = x_test.shape
        diffs = (x_train - x_test.reshape(dim[0], 1, dim[1])) # reshape with additional axis, then broadcast substraction into feature data
        if self.dist_type == 'l2': # if we want l2, use l2
            dist = np.sqrt(np.sum(np.square(diffs), axis=2)) # we calculate distances for all x_test points 
        else:                      # assume anything else is l1  
            dist = np.sum(np.abs(diffs), axis=2)
        i_nn = np.argpartition(dist, kth=self.k)[:, :self.k] # we find nearest neighbour of each x_test point
        return np.mean(y_train[i_nn], axis=1)    # return mean of the nearest neighbour of each x_test point respectively
    
    def test_error(self, xdata, ydata, xtest, ytest):
        """
        Calculate the RMSE error from a test set
        xdata - model feature data, used for prediction
        ydata - model target data, used for prediction
        xtest - test feature data, used for prediction
        ytest - teste target data, used for error calculation
        """
        yhat = self.batch_re(xdata, ydata, xtest)
        error = np.sqrt(np.mean(np.square(yhat - ytest)))
        return error
                                   
    def batch_cv_knn(self, xdata, ydata, folds):
        '''
        Run cross validation on set
        xdata - feature points from the data set
        ydata - target points from the data set
        folds - number of CV folds you would like (folds > 1)
        '''
        score = 0
        prediction = []
        xfolds = np.array_split(xdata, folds, axis=0,)
        yfolds = np.array_split(ydata, folds, axis=0,)
        for i in range(folds):
                validx = xfolds[i]
                trainx = np.vstack([xfolds[j] for j in range(folds) if j != i])
                validy = yfolds[i]
                trainy = np.vstack([yfolds[j] for j in range(folds) if j != i])
                yhat = self.batch_re(trainx, trainy, validx)
                prediction.append(yhat)
                error = np.sqrt(np.mean(np.square(yhat - validy)))
                score += error
        self.pred = np.vstack(prediction)
        return score/folds
   

