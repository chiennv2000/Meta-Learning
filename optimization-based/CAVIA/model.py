import numpy as np
from utils import sigmoid, cross_entropy
from sklearn.model_selection import train_test_split

np.random.seed(42)

class DataGenerator(object):
    def __init__(self,
                 num_task,
                 num_samples,
                 input_dim=50):
        """
        To generate data contains k samples to train and k samples to test.
        """
        self.task = []
        for i in range(num_task):
            X, y = self.sample_points(num_samples + 5)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=5, random_state=42)
            self.task.append((X_train, X_test, y_train, y_test))
            
    
    def sample_points(self, k):
        X = np.random.rand(k, 50)
        y = np.random.choice([0, 1], size=k, p=[0.5, 0.5])
        return X, y
    
    def __len__(self):
        return len(self.task)
    
    def __getitem__(self, i):
        return self.task[i]

class CAVIA(object):
    def __init__(self, 
                 num_tasks,
                 num_samples,
                 input_dim=50,
                 context_dim=50):
        
        self.num_tasks = num_tasks
        self.num_samples = num_samples
        self.context_dim = context_dim

        # meta-learner - shared parameters
        self.theta = np.random.randn(input_dim + context_dim) * 0.01
        
        self.context_phi = np.zeros((num_tasks, context_dim))
        
        # data generator
        self.tasks = DataGenerator(num_tasks, num_samples, input_dim)
    
    def _set_context_phi_zero(self):
        self.context_phi = np.zeros((self.num_tasks, self.context_dim))
        
    def inner_loop(self, alpha):
        self._set_context_phi_zero()
        
        for i in range(len(self.tasks)):
            X_train, _, y_train, _ = self.tasks[i]
            
            # Concat x with context parameters
            X_train = np.concatenate((X_train, np.stack([self.context_phi[i]] * X_train.shape[0])), axis=1)
            y_hat = sigmoid(X_train.dot(self.theta))
            
            loss = cross_entropy(y_hat, y_train)
            
            d_context = (y_hat - y_train).reshape(-1, 1).dot(self.theta.reshape(-1, 1).T)[:, - self.context_dim:]
            d_context = np.mean(d_context, axis=0)
            
            # Update context parameters
            self.context_phi[i] -= alpha * d_context
            
    
    def outer_loop(self, beta):
        meta_gradient = np.zeros(self.theta.shape)
        loss = 0.0 
        
        for i in range(len(self.tasks)):
            _, X_test, _, y_test = self.tasks[i]
            
            X_test = np.concatenate((X_test, np.stack([self.context_phi[i]] * X_test.shape[0])), axis=1)
            y_hat = sigmoid(X_test.dot(self.theta))
            
            loss += cross_entropy(y_hat, y_test)
            
            meta_gradient += X_test.T.dot(y_hat - y_test) * 1.0 / X_test.shape[0]
            
        meta_gradient /= len(self.tasks)
        # Update shared parameters
        self.theta -= beta*meta_gradient
        
        loss /= len(self.tasks)
        return loss
    
    def train(self,
              num_epochs=10000,
              alpha=1e-3,
              beta=1e-3):
        
        print("Training....")
        for epoch in range(num_epochs):
            self.inner_loop(alpha)
            loss = self.outer_loop(beta)
            
            if epoch % int((num_epochs / 10)) == 0:
                print("Epoch {}: Loss {}\n".format(epoch, loss))
        
            
            
        
        