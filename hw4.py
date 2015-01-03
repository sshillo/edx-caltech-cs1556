from math import log
import matplotlib.pyplot as plt
import numpy as np

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


#prob2()
prob3()
