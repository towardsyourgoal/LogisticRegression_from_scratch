import numpy as np

class LogisticRegressor:
    """
    Custom implementation of Logistic Regression, a binary classification algorithm
    It models the probablity that a sample belongs to the positive class using the sigmoid function
    """
    def __init__(self, learning_rate=0.01, iterations=10000):
        """
        Initializing the parameters
        """
        self.iterations=iterations
        self.learning_rate=learning_rate
        self.weights=None
        self.bias=None
        self.prob_score=None
        self.cost_history=[]
        
    def linear_op(self,X,w,b):
        """
        Linear model function for the given inputs, weights and bias
        """
        return np.dot(X,self.weights) + self.bias

    def sigmoid(self,z):
        """
        Sigmoid function to apply logistic transformation
        Converts linear output to probablity
        """
        return 1/(1+np.exp(-z))
    
    def compute_cost(self,y,p):
        """
        Computing cost for the given labels and probablities
        """
        m=len(y)
        loss=0
        for i in range(m):
            loss_i=(y[i]*np.log(p[i])) + ((1-y[i]) * np.log(1-p[i]))
            loss+=loss_i
        return -loss/m
    
    def compute_gradient(self,X,y,p):
        """
        Computing gradients for the given inputs, labels and probablities
        """
        m,n=X.shape
        dL_dw=np.dot(np.transpose(X),(p-y)) * (1/m)
        dL_db=np.sum(p-y) * (1/m)
        return dL_dw,dL_db
    

    def fit(self,X,y):
        """
        Training of the model
        Computes gradients using above formula
        Updates weights and bias
        """
        X_t=X.to_numpy()
        y_t=y.to_numpy()
        m,n=X.shape
        self.weights=np.zeros(n)
        self.bias=0
        for _ in range(self.iterations):
            z=self.linear_op(X_t,self.weights,self.bias)
            s_op=self.sigmoid(z)
            dL_dw,dL_db=self.compute_gradient(X_t,y_t,s_op)
            self.cost_history.append(self.compute_cost(y_t,s_op))
            self.weights-=self.learning_rate * dL_dw
            self.bias-=self.learning_rate * dL_db


    def predict(self,X):
        """
        Uses weights and bias from the trained model
        Generates class labels from the probablities
        Uses threhsold value of 0.5
        """
        x_test_t=X.to_numpy()
        m,n=X.shape
        y_pred=np.zeros((m,))
        z=self.linear_op(x_test_t,self.weights,self.bias)
        y_pred=self.sigmoid(z)
        self.prob_score=y_pred
        return (y_pred >= 0.5).astype(int)

    def predict_probs(self,X):
        """
        Uses weights and bias from the trained model
        Outputs the probabilites instead of class labels
        """
        x_test_t=X.to_numpy()
        m,n=X.shape
        y_pred=np.zeros((m,))
        z=self.linear_op(x_test_t,self.weights,self.bias)
        y_pred=self.sigmoid(z)
        return y_pred


    def mse(self, y_true, y_pred):
        """
        This function will calculate Mean Squared Error between true and predicted values
        """
        assert len(y_true)==len(y_pred), "Length of true values and predicted values must be the same"
        y_true_np=y_true.to_numpy().reshape(-1,1)
        m=len(y_true_np)
        mse=0
        for i in range(m):
            mse+=(y_pred[i]-y_true_np[i])**2
        mse=mse/m
        return mse
    
    def get_confusion_matrix(self,y_true,y_pred):
        """
        This function outputs a matrix that is sized 2x2
        The rows are true values and columns are predicted values 
        """
        classes=np.unique(np.concatenate((y_true,y_pred)))
        num_classes=len(classes)
        self.confusion_matrix=np.zeros((num_classes,num_classes),dtype=int)
        class_to_index={cls:i for i,cls in enumerate(classes)}
        for i,j in zip(y_true,y_pred):
            true_idx=class_to_index[i]
            pred_idx=class_to_index[j]
            self.confusion_matrix[true_idx,pred_idx]+=1
        return self.confusion_matrix
    
    def find_metrics_from_con_mat(self,y_true,y_pred,con_mat):
        """
        This function returns core metrics 
        TP- True Positoves,
        TN- True Negatives,
        FP- False Positives
        FN- False Negatives
        These will be used in finding important evaluation metrics for classification
        """
        self.TP=con_mat[1,1]
        self.TN=con_mat[0,0]
        self.FP=con_mat[0,1]
        self.FN=con_mat[1,0]
        return self.TP, self.TN, self.FP, self.FN
    

    def accuracy_score(self,y_true,y_pred):
        """
        This returns accuracy of the classfier 
        How well the model is predicting correctly across the samples
        """
        n_samples=len(y_true)
        con_mat=self.get_confusion_matrix(y_true,y_pred)
        TP, TN, FP, FN=self.find_metrics_from_con_mat(y_true,y_pred,con_mat)
        return (TP+TN)/(TP+TN+FP+FN)
    
        
    def precision_score(self,y_true,y_pred):
        """
        This returns how often a positive label is actually correct
        Our aim is to make the FP to 0 - no false alarms 
        """
        con_mat=self.get_confusion_matrix(y_true,y_pred)
        TP, TN, FP, FN=self.find_metrics_from_con_mat(y_true,y_pred,con_mat)
        return TP/(TP+FP)
    
    def recall_score(self,y_true,y_pred):
        """
        This returns score - "of all the positive instances, how much the model actually identified correctly ?"
        Very important for imbalanced datasets where missing a positive can be very costly
        Our aim is to make the FN to 0 - no missed alarms 
        """
        con_mat=self.get_confusion_matrix(y_true,y_pred)
        TP, TN, FP, FN=self.find_metrics_from_con_mat(y_true,y_pred,con_mat)
        return TP/(TP+FN)
    
    def f1_score(self,y_true,y_pred, average='micro'):
        """
        This calculates hormonice mean of precision and recall
        Again, very important for imbalanced datasets
        Accounts for both False alarms as well as Missed alarms
        """
        precision=self.precision_score(y_true,y_pred)
        recall=self.recall_score(y_true,y_pred)
        return 2 * (precision * recall)/ (precision + recall)
    

    def fpr(self,y_true,y_pred):
        """
        This calculates and returns ratio of total False positives the model calculated to the actual negatives 
        Ideally should be zero, where there are no false alarms
        """
        con_mat=self.get_confusion_matrix(y_true,y_pred)
        TP, TN, FP, FN=self.find_metrics_from_con_mat(y_true,y_pred,con_mat)
        return FP/(FP+TN)
    
    def specificity(self,y_true,y_pred):
        """
        This is 1-fpr, basically True Negative Rate
        This denotes - how well a model is identifying actual negatives
        Ideally should be 1 (No false alarams)
        """
        con_mat=self.get_confusion_matrix(y_true,y_pred)
        TP, TN, FP, FN=self.find_metrics_from_con_mat(y_true,y_pred,con_mat)
        return TN/(TN+FP)
    
    def calculate_roc_auc_score(self,y_true,y_pred,pos_probs):
        """
        This calculates the area under the curve, probablity that model ranks a random positive sample higher than a random negative sample
        Or how the model is performing in separating the two classes
        The higher this value, the better the classification
        """
        yt=np.array(y_true)
        ytprobs=np.array(pos_probs)
        test_pos_probabilities=pos_probs
        sorted_indices = np.argsort(test_pos_probabilities)[::-1]
        y_true_sorted = yt[sorted_indices]
        y_probs_sorted = ytprobs[sorted_indices]
        n_pos=(y_true==1).sum()
        n_neg=(y_true==0).sum()
        fpr_list = [0.0]
        tpr_list = [0.0]
        for i in range(len(y_probs_sorted)):
            TP=(y_true_sorted[:i+1]==1).sum()
            FP=(y_true_sorted[:i+1]==0).sum()
            tpr=TP/n_pos if n_pos>0 else 0
            fpr=FP/n_neg if n_neg>0 else 0
            tpr_list.append(tpr)
            fpr_list.append(fpr)
        tpr_list.append(1.0)
        fpr_list.append(1.0)
        #Area=(1/2)(sum of parallel sides)×width
        roc_auc = np.trapezoid(tpr_list, fpr_list)
        return roc_auc
    