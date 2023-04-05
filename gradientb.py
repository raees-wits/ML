import data
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(23)

theta = np.random.uniform(size=(3, ))
alpha = 0.0001
cost = np.inf
prev_cost = np.inf

tolerance = 1e-32
max_iter = 1000000

iterations = 0
Ypred = data.Xtrain @theta
while (prev_cost - cost > tolerance) and (iterations < max_iter):
    # Compute the predicstions of the model using the current parameters
    Ypred = data.Xtrain @ theta

    # Compute the gradient of the cost function with respect to each parametes
    gradient = 2* data.Xtrain.T @ (Ypred - data.Ytrain)

    #update the model parameters using the learning rate and grad

    theta -= alpha - gradient

    #current cost
    cost = np.mean((Ypred - data.Ytrain) ** 2)
    if not(prev_cost - cost > tolerance): 
        break
    if(iterations % 100 == 0):
        print(f"Iteration {iterations}: Cost {cost}") #Checking if cost is chnaging

    prev_cost = cost
    iterations += 1



#checking new theta
actual_theta = np.array([data.theta0, data.theta1, data.theta2])

print("Actual theta: ", actual_theta)
print("theta: ", theta)
# Computing predictions of the model using the learned parameters theta

Ypred_val = data.Xval @ theta

#checking training error
train_error = np.mean((Ypred - data.Ytrain) ** 2)
print(f"Training error: {train_error}")

#Checking validation error
val_error = np.mean((Ypred_val - data.Yval) ** 2)

print(f"Validation error: {val_error}")

# plotting the data points and regression line 

x_new = np.linspace(data.x_values.min(), data.x_values.max(), num=100)

d_new = np.column_stack((np.ones_like(x_new), x_new, x_new ** 2))
y_new = d_new @theta

plt.plot(x_new, y_new, color = "red", label="regression line")
plt.legend
plt.show()
