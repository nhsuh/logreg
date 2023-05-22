#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy.stats
from time import perf_counter


# In[2]:

#error function for both binary and cross entropy
def find_error(w, X, y, error_type):
    X = np.hstack((X, np.ones((len(X), 1))))  #add bias
    n, d = X.shape
    y = y.reshape(n,1)
    y[y == 0] = -1
    if error_type == "binary":
        y_pred = np.round(1/(1+np.exp(-X@w.T)))
        y_pred[y_pred == 0] = -1
        test_error = np.zeros(len(X))
        for i in range(len(test_error)):
            if y_pred[i] != y[i]: test_error[i] = 1
        test_error = sum(test_error)/len(test_error)
    if error_type == "cross_entropy":
        test_error = np.sum((np.log(1+np.exp(-y*X@w.T))), axis=0)/n
    return test_error


# In[3]:


def logistic_reg(X, y, w_init, max_its, eta = 10 ** -5): #logistic reg with a max_iteration
    w = w_init 
    t = 0
    X = np.hstack((X, np.ones((len(X), 1)))) 
    n, d = X.shape
    y = y.reshape(n,1)
    y[y == 0] = -1
    g_mag = 10 #arbitrarily large gradient magnitude
    while g_mag > 10 ** -3 and t < max_its: #check exit conditions
        g = (-np.sum((y*X/(1+np.exp(y*X@w.T))), axis=0)/n).reshape(1, 14)   #calculate gradient
        w = w - (eta * g) #gradient descent
        g_mag = np.linalg.norm(g, ord=np.inf) #infinity-norm of gradient
        t += 1 #increment t
    e_in = np.sum((np.log(1+np.exp(-y*X@w.T))), axis=0)/n #calculate cross-entropy error
    return t, w, e_in


# In[4]:


train_data = np.genfromtxt("cleveland_train.csv", dtype=float, delimiter=",")
train_data = train_data[1:]
train_X = train_data[:,:-1]
train_y = train_data[:,-1]
w_init = np.zeros((1, len(train_X[0])+1)) 


# In[5]:


test_data = np.genfromtxt("cleveland_test.csv", dtype=float, delimiter=",")
test_data = test_data[1:]
test_X = test_data[:,:-1]
test_y = test_data[:,-1]


# In[6]:


start_1 = perf_counter()
t, w, e_in = logistic_reg(train_X, train_y, w_init, np.power(10, 4))
stop_1 = perf_counter()

start_2 = perf_counter()
binary_train_error = find_error(w, train_X, train_y, "binary")
binary_test_error = find_error(w, test_X, test_y, "binary")
cross_entropy_test = find_error(w, test_X, test_y, "cross_entropy")
stop_2 = perf_counter()

print('max_its = 10 ** 4')
print('Iterations:', t)
print('Weights:', w[0])
print('-------------------------------------------------------------------------------------')
print('Cross Entropy Train Error:', e_in[0])
print('Binary Train Error:', binary_train_error)
print('Training Time (s):', stop_1-start_1)
print('-------------------------------------------------------------------------------------')
print('Binary Test Error:', binary_test_error)
print('Cross Entropy Test Error:', cross_entropy_test[0])


# In[7]:


start_1 = perf_counter()
t, w, e_in = logistic_reg(train_X, train_y, w_init, np.power(10, 5))
stop_1 = perf_counter()

start_2 = perf_counter()
binary_train_error = find_error(w, train_X, train_y, "binary")
binary_test_error = find_error(w, test_X, test_y, "binary")
cross_entropy_test = find_error(w, test_X, test_y, "cross_entropy")
stop_2 = perf_counter()

print('max_its = 10 ** 5')
print('Iterations:', t)
print('Weights:', w[0])
print('-------------------------------------------------------------------------------------')
print('Cross Entropy Train Error:', e_in[0])
print('Binary Train Error:', binary_train_error)
print('Training Time (s):', stop_1-start_1)
print('-------------------------------------------------------------------------------------')
print('Binary Test Error:', binary_test_error)
print('Cross Entropy Test Error:', cross_entropy_test[0])


# In[ ]:


start_1 = perf_counter()
t, w, e_in = logistic_reg(train_X, train_y, w_init, np.power(10, 6))
stop_1 = perf_counter()

start_2 = perf_counter()
binary_train_error = find_error(w, train_X, train_y, "binary")
binary_test_error = find_error(w, test_X, test_y, "binary")
cross_entropy_test = find_error(w, test_X, test_y, "cross_entropy")
stop_2 = perf_counter()

print('max_its = 10 ** 6')
print('Iterations:', t)
print('Weights:', w[0])
print('-------------------------------------------------------------------------------------')
print('Cross Entropy Train Error:', e_in[0])
print('Binary Train Error:', binary_train_error)
print('Training Time (s):', stop_1-start_1)
print('-------------------------------------------------------------------------------------')
print('Binary Test Error:', binary_test_error)
print('Cross Entropy Test Error:', cross_entropy_test[0])


# In[ ]:


start_1 = perf_counter()
w_init = np.zeros((1, len(train_X[0])+1)) 
start_1 = perf_counter()
train_X_norm = scipy.stats.zscore(train_X, axis=0, ddof=1)
test_X_norm = scipy.stats.zscore(test_X, axis=0,ddof=1)
t, w, e_in = logistic_reg(train_X_norm, train_y, w_init, np.power(10, 4))
stop_1 = perf_counter()

start_2 = perf_counter()
binary_train_error = find_error(w, train_X_norm, train_y, "binary")
binary_test_error = find_error(w, test_X_norm, test_y, "binary")
cross_entropy_test = find_error(w, test_X_norm, test_y, "cross_entropy")
stop_2 = perf_counter()

print('max_its = 10**4, normalized')
print('Iterations:', t)
print('Weights:', w[0])
print('-------------------------------------------------------------------------------------')
print('Cross Entropy Train Error:', e_in[0])
print('Binary Train Error:', binary_train_error)
print('Training Time (s):', stop_1-start_1)
print('-------------------------------------------------------------------------------------')
print('Binary Test Error:', binary_test_error)
print('Cross Entropy Test Error:', cross_entropy_test[0])


# In[ ]:


def logistic_reg_new(X, y, w_init, eta): #logistic reg with no max_iterations
    w = w_init 
    t = 0
    X = np.hstack((X, np.ones((len(X), 1)))) 
    n, d = X.shape
    y = y.reshape(n,1)
    y[y == 0] = -1
    g_mag = 10 #arbitrarily large gradient magnitude
    while g_mag > 10 ** -6: #check new exit condition
        #following is same as previous function
        g = (-np.sum((y*X/(1+np.exp(y*X@w.T))), axis=0)/n).reshape(1, 14)  
        w = w - (eta * g)
        g_mag = np.linalg.norm(g, np.inf)
        t += 1
        #print(g_mag)
    e_in = np.sum((np.log(1+np.exp(-y*X@w.T))), axis=0)/n
    return t, w, e_in


# In[ ]:


eta = 0.01
w_init = np.zeros((1, len(train_X[0])+1)) 
start_1 = perf_counter()
train_X_norm = scipy.stats.zscore(train_X, axis=0, ddof=1)
test_X_norm = scipy.stats.zscore(test_X, axis=0,ddof=1)
t, w, e_in = logistic_reg_new(train_X_norm, train_y, w_init, eta)
stop_1 = perf_counter()

start_2 = perf_counter()
binary_train_error = find_error(w, train_X_norm, train_y, "binary")
binary_test_error = find_error(w, test_X_norm, test_y, "binary")
cross_entropy_test = find_error(w, test_X_norm, test_y, "cross_entropy")
stop_2 = perf_counter()

print('eta =', eta)
print('Iterations:', t)
print('Weights:', w[0])
print('-------------------------------------------------------------------------------------')
print('Cross Entropy Train Error:', e_in[0])
print('Binary Train Error:', binary_train_error)
print('Training Time (s):', stop_1-start_1)
print('-------------------------------------------------------------------------------------')
print('Binary Test Error:', binary_test_error)
print('Cross Entropy Test Error:', cross_entropy_test[0])


# In[ ]:


eta = 0.1
w_init = np.zeros((1, len(train_X[0])+1)) 
start_1 = perf_counter()
train_X_norm = scipy.stats.zscore(train_X, axis=0, ddof=1)
test_X_norm = scipy.stats.zscore(test_X, axis=0,ddof=1)
t, w, e_in = logistic_reg_new(train_X_norm, train_y, w_init, eta)
stop_1 = perf_counter()

start_2 = perf_counter()
binary_train_error = find_error(w, train_X_norm, train_y, "binary")
binary_test_error = find_error(w, test_X_norm, test_y, "binary")
cross_entropy_test = find_error(w, test_X_norm, test_y, "cross_entropy")
stop_2 = perf_counter()

print('eta =', eta)
print('Iterations:', t)
print('Weights:', w[0])
print('-------------------------------------------------------------------------------------')
print('Cross Entropy Train Error:', e_in[0])
print('Binary Train Error:', binary_train_error)
print('Training Time (s):', stop_1-start_1)
print('-------------------------------------------------------------------------------------')
print('Binary Test Error:', binary_test_error)
print('Cross Entropy Test Error:', cross_entropy_test[0])


# In[ ]:


eta = 1
w_init = np.zeros((1, len(train_X[0])+1)) 
start_1 = perf_counter()
train_X_norm = scipy.stats.zscore(train_X, axis=0, ddof=1)
test_X_norm = scipy.stats.zscore(test_X, axis=0,ddof=1)
t, w, e_in = logistic_reg_new(train_X_norm, train_y, w_init, eta)
stop_1 = perf_counter()

start_2 = perf_counter()
binary_train_error = find_error(w, train_X_norm, train_y, "binary")
binary_test_error = find_error(w, test_X_norm, test_y, "binary")
cross_entropy_test = find_error(w, test_X_norm, test_y, "cross_entropy")
stop_2 = perf_counter()

print('eta =', eta)
print('Iterations:', t)
print('Weights:', w[0])
print('-------------------------------------------------------------------------------------')
print('Cross Entropy Train Error:', e_in[0])
print('Binary Train Error:', binary_train_error)
print('Training Time (s):', stop_1-start_1)
print('-------------------------------------------------------------------------------------')
print('Binary Test Error:', binary_test_error)
print('Cross Entropy Test Error:', cross_entropy_test[0])


# In[ ]:


eta = 4
w_init = np.zeros((1, len(train_X[0])+1)) 
start_1 = perf_counter()
train_X_norm = scipy.stats.zscore(train_X, axis=0, ddof=1)
test_X_norm = scipy.stats.zscore(test_X, axis=0,ddof=1)
t, w, e_in = logistic_reg_new(train_X_norm, train_y, w_init, eta)
stop_1 = perf_counter()

start_2 = perf_counter()
binary_train_error = find_error(w, train_X_norm, train_y, "binary")
binary_test_error = find_error(w, test_X_norm, test_y, "binary")
cross_entropy_test = find_error(w, test_X_norm, test_y, "cross_entropy")
stop_2 = perf_counter()

print('eta =', eta)
print('Iterations:', t)
print('Weights:', w[0])
print('-------------------------------------------------------------------------------------')
print('Cross Entropy Train Error:', e_in[0])
print('Binary Train Error:', binary_train_error)
print('Training Time (s):', stop_1-start_1)
print('-------------------------------------------------------------------------------------')
print('Binary Test Error:', binary_test_error)
print('Cross Entropy Test Error:', cross_entropy_test[0])


# In[ ]:


eta = 5
w_init = np.zeros((1, len(train_X[0])+1)) 
start_1 = perf_counter()
train_X_norm = scipy.stats.zscore(train_X, axis=0, ddof=1)
test_X_norm = scipy.stats.zscore(test_X, axis=0,ddof=1)
t, w, e_in = logistic_reg_new(train_X_norm, train_y, w_init, eta)
stop_1 = perf_counter()

start_2 = perf_counter()
binary_train_error = find_error(w, train_X_norm, train_y, "binary")
binary_test_error = find_error(w, test_X_norm, test_y, "binary")
cross_entropy_test = find_error(w, test_X_norm, test_y, "cross_entropy")
stop_2 = perf_counter()

print('eta =', eta)
print('Iterations:', t)
print('Weights:', w[0])
print('-------------------------------------------------------------------------------------')
print('Cross Entropy Train Error:', e_in[0])
print('Binary Train Error:', binary_train_error)
print('Training Time (s):', stop_1-start_1)
print('-------------------------------------------------------------------------------------')
print('Binary Test Error:', binary_test_error)
print('Cross Entropy Test Error:', cross_entropy_test[0])


# In[ ]:


eta = 6
w_init = np.zeros((1, len(train_X[0])+1)) 
start_1 = perf_counter()
train_X_norm = scipy.stats.zscore(train_X, axis=0, ddof=1)
test_X_norm = scipy.stats.zscore(test_X, axis=0,ddof=1)
t, w, e_in = logistic_reg_new(train_X_norm, train_y, w_init, eta)
stop_1 = perf_counter()

start_2 = perf_counter()
binary_train_error = find_error(w, train_X_norm, train_y, "binary")
binary_test_error = find_error(w, test_X_norm, test_y, "binary")
cross_entropy_test = find_error(w, test_X_norm, test_y, "cross_entropy")
stop_2 = perf_counter()

print('eta =', eta)
print('Iterations:', t)
print('Weights:', w[0])
print('-------------------------------------------------------------------------------------')
print('Cross Entropy Train Error:', e_in[0])
print('Binary Train Error:', binary_train_error)
print('Training Time (s):', stop_1-start_1)
print('-------------------------------------------------------------------------------------')
print('Binary Test Error:', binary_test_error)
print('Cross Entropy Test Error:', cross_entropy_test[0])


# In[ ]:


eta = 7
w_init = np.zeros((1, len(train_X[0])+1)) 
start_1 = perf_counter()
train_X_norm = scipy.stats.zscore(train_X, axis=0, ddof=1)
test_X_norm = scipy.stats.zscore(test_X, axis=0,ddof=1)
t, w, e_in = logistic_reg_new(train_X_norm, train_y, w_init, eta)
stop_1 = perf_counter()

start_2 = perf_counter()
binary_train_error = find_error(w, train_X_norm, train_y, "binary")
binary_test_error = find_error(w, test_X_norm, test_y, "binary")
cross_entropy_test = find_error(w, test_X_norm, test_y, "cross_entropy")
stop_2 = perf_counter()

print('eta =', eta)
print('Iterations:', t)
print('Weights:', w[0])
print('-------------------------------------------------------------------------------------')
print('Cross Entropy Train Error:', e_in[0])
print('Binary Train Error:', binary_train_error)
print('Training Time (s):', stop_1-start_1)
print('-------------------------------------------------------------------------------------')
print('Binary Test Error:', binary_test_error)
print('Cross Entropy Test Error:', cross_entropy_test[0])


# In[ ]:


eta = 7.5
w_init = np.zeros((1, len(train_X[0])+1)) 
start_1 = perf_counter()
train_X_norm = scipy.stats.zscore(train_X, axis=0, ddof=1)
test_X_norm = scipy.stats.zscore(test_X, axis=0,ddof=1)
t, w, e_in = logistic_reg_new(train_X_norm, train_y, w_init, eta)
stop_1 = perf_counter()

start_2 = perf_counter()
binary_train_error = find_error(w, train_X_norm, train_y, "binary")
binary_test_error = find_error(w, test_X_norm, test_y, "binary")
cross_entropy_test = find_error(w, test_X_norm, test_y, "cross_entropy")
stop_2 = perf_counter()

print('eta =', eta)
print('Iterations:', t)
print('Weights:', w[0])
print('-------------------------------------------------------------------------------------')
print('Cross Entropy Train Error:', e_in[0])
print('Binary Train Error:', binary_train_error)
print('Training Time (s):', stop_1-start_1)
print('-------------------------------------------------------------------------------------')
print('Binary Test Error:', binary_test_error)
print('Cross Entropy Test Error:', cross_entropy_test[0])


# In[ ]:


eta = 7.6
w_init = np.zeros((1, len(train_X[0])+1)) 
start_1 = perf_counter()
train_X_norm = scipy.stats.zscore(train_X, axis=0, ddof=1)
test_X_norm = scipy.stats.zscore(test_X, axis=0,ddof=1)
t, w, e_in = logistic_reg_new(train_X_norm, train_y, w_init, eta)
stop_1 = perf_counter()

start_2 = perf_counter()
binary_train_error = find_error(w, train_X_norm, train_y, "binary")
binary_test_error = find_error(w, test_X_norm, test_y, "binary")
cross_entropy_test = find_error(w, test_X_norm, test_y, "cross_entropy")
stop_2 = perf_counter()

print('eta =', eta)
print('Iterations:', t)
print('Weights:', w[0])
print('-------------------------------------------------------------------------------------')
print('Cross Entropy Train Error:', e_in[0])
print('Binary Train Error:', binary_train_error)
print('Training Time (s):', stop_1-start_1)
print('-------------------------------------------------------------------------------------')
print('Binary Test Error:', binary_test_error)
print('Cross Entropy Test Error:', cross_entropy_test[0])


# In[ ]:


eta = 7.65
w_init = np.zeros((1, len(train_X[0])+1)) 
start_1 = perf_counter()
train_X_norm = scipy.stats.zscore(train_X, axis=0, ddof=1)
test_X_norm = scipy.stats.zscore(test_X, axis=0,ddof=1)
t, w, e_in = logistic_reg_new(train_X_norm, train_y, w_init, eta)
stop_1 = perf_counter()

start_2 = perf_counter()
binary_train_error = find_error(w, train_X_norm, train_y, "binary")
binary_test_error = find_error(w, test_X_norm, test_y, "binary")
cross_entropy_test = find_error(w, test_X_norm, test_y, "cross_entropy")
stop_2 = perf_counter()

print('eta =', eta)
print('Iterations:', t)
print('Weights:', w[0])
print('-------------------------------------------------------------------------------------')
print('Cross Entropy Train Error:', e_in[0])
print('Binary Train Error:', binary_train_error)
print('Training Time (s):', stop_1-start_1)
print('-------------------------------------------------------------------------------------')
print('Binary Test Error:', binary_test_error)
print('Cross Entropy Test Error:', cross_entropy_test[0])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




