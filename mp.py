import data
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(23)
#calculating closed form solution of paramters based on training data
X_pinv = np.linalg.pinv(data.Xtrain)
theta = X_pinv.dot(data.Ytrain)
actual_theta = np.array([data.theta0, data.theta1, data.theta2])
print("Actual theta: ", actual_theta)
print("calculated parameter values: ", theta)


#computing predicted values of y using the theta obtained based off the training data
y_pred = data.Xtrain.dot(theta)

#Checking the training errorr through MSE
MSE  = np.mean((y_pred - data.Ytrain)**2)

print("Training error : " , MSE) # lower the SSE the better the prediction

#Validation error
y_val_pred = data.Xval.dot(theta)
MSE_val = np.mean((y_val_pred - data.Yval)**2)

print("Validation error : ", MSE_val)


#generating new xvalues
x_new = np.linspace(data.x_values.min(), data.x_values.max(), num = 100)
d_new = np.zeros((100, 3))
d_new[:,0] = 1
d_new[:,1] = x_new
d_new[:,2] = x_new**2

y_new = d_new.dot(theta)
#Plotting the regression line based off of the traning data, using same scatter plot as before
plt.plot(x_new, y_new, 'r', label ="regression Line")
plt.legend()
plt.show()
