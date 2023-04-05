import numpy as np
import matplotlib.pyplot as plt
#Set a random seed
np.random.seed(23)
#Sampled 150 x values from Normal(0, 10)
x_values = np.random.normal(0, 10, 150)

#print(sample.shape)
#creating the design matrix
d_matrix = np.zeros((150, 3))
d_matrix[:, 0] = 1
d_matrix[:, 1] = x_values
d_matrix[:, 2] = x_values**2

#print(d_matrix)
theta0 = np.random.uniform()
theta1 = np.random.uniform()
theta2 = np.random.uniform()
#print(theta0, theta1, theta2)

#create y-values for the regression data
y_values = d_matrix.dot(np.array([theta0, theta1, theta2]))

#adding noise to the data
noise = np.random.normal(0,8, 150)
y_values+= noise

plt.scatter(x_values, y_values, label="Data")
plt.xlabel("x values")
plt.ylabel("y values")
#plt.show()


#splitting into training, testing data
Xtrain, Xtest = np.split(d_matrix, [int(0.6*len(x_values))])
Ytrain, Ytest = np.split(y_values, [int(0.6 * len(x_values))])

#splitting into testing and validation data

Xtest, Xval = np.split(Xtest, [int(0.5 * len(Xtest))])
Ytest, Yval = np.split(Ytest, [int(0.5 * len(Ytest))])

#print(len(Xtrain), len(Xtest), len(Xval))

#print(Xtrain)


