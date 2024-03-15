import numpy as np
import matplotlib.pyplot as plt
 
np.random.seed(42)
X=2*np.random.rand(100,1)
y=4+3*X+np.random.randn(100,1)
x_b=np.c_[np.ones((100,1)),X]
 
learning_rate=0.01
n_iterations=1000
 
theta=np.random.randn(2,1)
for iteration in range(n_iterations):
    gradients=2/100*x_b.T.dot(x_b.dot(theta)-y)
    theta=theta-learning_rate*gradients
print(theta)
plt.scatter(X,y,label='Data')
plt.plot(X,x_b.dot(theta),color='pink',label='Linear Regression')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

