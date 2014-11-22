import numpy as np

#ex 8-10

def f(d):
  return np.sign(d[0,1] ** 2 + d[0,2] ** 2 - 0.6)

data_size = 1000
e_count = 0.
trials = 100

for i in range(trials):
  x = np.random.rand(data_size, 2) * 2 - 1
  x0 = np.ones(data_size)
  x = np.mat(np.column_stack((x0, x)))
  y = np.mat( [ f(d) for d in x] ).T

  #create noise
  for i in np.random.randint(0, data_size, data_size * .1):
    y[i] = y[i] * -1

  #w = (x' * x) ^ -1 * x' * y
  w = np.linalg.inv(x.T * x) * x.T * y
  diff = np.sign(w.T * x.T) - y.T 
  for i in np.nditer(diff):
    if i == 0:
      e_count += 1
  
print(e_count/(data_size * trials))


