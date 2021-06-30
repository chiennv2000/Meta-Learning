import numpy as np
from utils import *
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

class MAML(object):
    def __init__(self, 
                 num_tasks,
                 num_samples,
                 input_dim=50):
        
        self.num_taks = num_tasks
        self.num_samples = num_samples

        # meta-learner
        self.theta = np.random.randn(input_dim) * 0.1
        self.alpha = np.random.randn(input_dim) * 0.1
        
        # data generator
        self.tasks = DataGenerator(num_tasks, num_samples, input_dim)
    
    def inner_loop(self):
        theta_ = []
        for i in range(len(self.tasks)):
            X_train, _, y_train, _ = self.tasks[i]
            y_hat = sigmoid(X_train.dot(self.theta))
            
            loss = cross_entropy(y_train, y_hat)
            
            d_theta = 1.0/y_hat.shape[0] * X_train.T.dot(y_hat - y_train)
            theta_.append(self.theta - self.alpha*d_theta)
        
        return theta_
    
    def outer_loop(self, task_theta_):
        meta_gradient = np.zeros(self.theta.shape)
        loss = 0.0
        for i in range(len(self.tasks)):
            _, X_test, _, y_test = self.tasks[i]
            
            y_hat = sigmoid(X_test.dot(task_theta_[i]))
            loss += cross_entropy(y_test, y_hat)
            
            meta_gradient += 1.0/y_hat.shape[0] * X_test.T.dot(y_hat - y_test)
        
        meta_gradient /= len(self.tasks)
        loss /= len(self.tasks)
        
        return meta_gradient, loss
    
    def train(self,
              num_epochs=10000,
              beta=1e-3):
        
        print("Training....")
        for epoch in range(num_epochs):
            theta_ = self.inner_loop()
            meta_gradient, loss = self.outer_loop(theta_)
            
            # update meta-learner
            self.theta -= beta*meta_gradient
            self.alpha -= beta*meta_gradient
            
            if epoch % int((num_epochs / 10)) == 0 or epoch == num_epochs - 1:
                print("Epoch {}: Loss {}\n".format(epoch, loss))
        
            
            
        
        