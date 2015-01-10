from math import log
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as r

def prob1():
  n = 1000
  trials = 10
  dvc = 10
  sigma = .05
  e = .05

  for i in range(trials):
    n = (8 / e ** 2) * log( (4 * (1 * n) ** dvc) / sigma)
    print("n: {}\n".format(n))

def prob2():
  dvc = 50.
  sigma = .05
  n = np.arange(1000., 15000., 1000.)
  e = .5
  print(n)
  vc_bound = ((8/n) * np.log(4 * (2 * n) ** dvc / sigma )) ** 1/2
  rademacher_bound = (( 2 * np.log( 2 * n * n ** dvc) / n) ** 1/2 +
      ( 2/n * np.log(1/sigma)) ** 1/2 + 1/n)
  p_bound = e
  for i in range(10):
    p_bound = ( 1/n * (2 * p_bound + np.log(6 * ( 2 * n) ** dvc / sigma))) ** 1/2

  d_bound = e
  for i in range(2):
    d_bound =  ( 1/( 2 * n) * (4 * d_bound * (1 + d_bound) + np.log( 4/ sigma) +
      2 * dvc * np.log(n))) ** 1/2
  plt.plot(n, vc_bound, n, rademacher_bound, n, p_bound, n, d_bound)
  plt.xlabel('N')
  plt.ylabel('generalization error')
  plt.show()

def prob3():
  dvc = 50.
  sigma = .05
  n = np.arange(1., 10., 1.)
  e = .5
  print(n)
  vc_bound = ((8/n) * np.log(4 * (2 * n) ** dvc / sigma )) ** 1/2
  rademacher_bound = (( 2 * np.log( 2 * n * n ** dvc) / n) ** 1/2 +
      ( 2/n * np.log(1/sigma)) ** 1/2 + 1/n)
  p_bound = e
  for i in range(10):
    p_bound = ( 1/n * (2 * p_bound + np.log(6 * ( 2 * n) ** dvc / sigma))) ** 1/2

  d_bound = e
  for i in range(2):
    d_bound =  ( 1/( 2 * n) * (4 * d_bound * (1 + d_bound) + np.log( 4/ sigma) +
      2 * dvc * np.log(n))) ** 1/2
  plt.plot(n, vc_bound, 'b', n, rademacher_bound, 'g', n, p_bound, 'r', n,
      d_bound, 'y')
  plt.xlabel('N')
  plt.ylabel('generalization error')
  plt.show()


def prob4():
  result = np.array([])
  full_data = np.array([])
  n = 5000
  data = = r.random((n, 2)) * 2 - 1
  y = np.sin(data * np.pi)

  for i in range(5000):
    d = np.mat(r.random(2) * 2 - 1)
    full_data = np.append(full_data, d)
    d = d.T
    y = np.sin(d * np.pi)
    #print(y)
    a = np.linalg.inv(d.T * d) * d.T * y
    result = np.append(result, a)
    #print(a)
  #print(result)
  gbar = np.mean(result)

  var_array = np.array([])
  for i, d in enumerate(np.array_split(full_data, 5000)):
    y = result[i]
    dvar = np.mean((d * y - d * gbar) ** 2)
    var_array = np.append(var_array, dvar)

  expected_bias = np.mean((gbar * full_data - np.sin(full_data * np.pi)) ** 2)
  expected_var = np.mean(var_array)

  print("bias {} var {} gbar {}".format(expected_bias,expected_var, gbar))

def test_hypoth(n):
  g



def prob7():
  N = 200
  x = r.random(N) * 10 - 5
  x.sort()

  xm = np.mat(x).T

  z = x ** 2
  zm = np.mat(z).T
  zm = np.concatenate((np.ones((N,1)), np.power(xm,2)), axis=1)
  print(zm)

  y = x ** 2 + 10.
  #print(y)
  #plt.plot(x,y)
  #plt.show()
  #xm = np.concatenate((np.ones((N,1)), np.zeros((N,1)), np.mat(x).T), axis=1)
  #print(xm)
  h = np.linalg.inv(xm.T * xm) * xm.T * np.mat(y).T
  hp = np.linalg.inv(zm.T * zm) * zm.T * np.mat(y).T
  print(hp)
  #hx = h[0,0] * x ** 2
  hx = zm * hp 
  print(np.array(hx.T))
  print(x.shape)
  print(hx.shape)
  
  plt.plot(x,y, 'r--', x, np.array(hx), 'go')
  plt.show()

# b
# ax + b
# ax^2
# ax^2 + b
prob7()
#prob2()
#prob3()

