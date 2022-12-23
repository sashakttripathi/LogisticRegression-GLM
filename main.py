import pandas as pd
import numpy as np
import time

data = pd.read_csv('data/Telco-Customer-Churn.csv')
df = data.copy()

df = df[df.TotalCharges != ' ']
df['TotalCharges'] = df['TotalCharges'].astype('float64')

df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
X = df[['tenure', 'TotalCharges']].copy()
y = df['Churn'].copy()

affine_constants = np.ones((X.shape[0], 1))
X = np.concatenate((affine_constants, X), axis=1)

W = np.zeros(X.shape[1])
learning_rate = 0.01
iterations = 1000000
start = time.time()


def logit_function(X, W):
    return 1 / (1 + np.exp(-np.dot(X, W)))


def gradient_of_log_likelihood(X, Y, W):
    return np.dot(X.T, Y - logit_function(X, W))


def update_rule_w(W, X, Y, learning_rate):
    return W + learning_rate * gradient_of_log_likelihood(X, Y, W)


def test():
    result = logit_function(X, W)
    rf = pd.DataFrame(result).join(y)
    rf.to_csv('data/result.csv', index=False)
    print("accuracy is : ", rf.loc[rf[0] == rf['Churn']].shape[0] / rf.shape[0] * 100, "%")


for i in range(iterations):
    W = update_rule_w(W, X, y, learning_rate)

print("Time taken: ", time.time() - start)
test()
