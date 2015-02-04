import math 
import numpy as np
import numpy.linalg as la
import random as r

def f(p):
  u,v = p
  return (u * math.e ** v - 2 * v * math.e ** -u) ** 2

def gradient(p):
  u,v = p
  common = 2 * (u * math.e ** v - 2 * v * math.e ** -u) 
  gu = common * (math.e ** v + 2 * v * math.e ** -u)
  gv = common * (u * math.e ** v - 2 * math.e ** -u)
  return np.array([gu, gv])

def prob_5():
  learning_rate = 0.1
  precision = 10 ** -14
  p = (1,1)
  i = 0
  while (f(p) > precision and i < 100):
    g = gradient(p)
    v = -1 * g
    p = p + learning_rate * v
    i += 1
  print("Local minimum occurs at {} took {} iteration, error {}".format(p,i,f(p)))

def gu(p):
  u,v = p
  common = 2 * (u * math.e ** v - 2 * v * math.e ** -u) 
  return common * (math.e ** v + 2 * v * math.e ** -u)

def gv(p):
  u,v = p
  common = 2 * (u * math.e ** v - 2 * v * math.e ** -u) 
  return common * (u * math.e ** v - 2 * math.e ** -u)

def prob_7():
  learning_rate = 0.1
  p = np.array([1.0,1.0])
  for i in range(15):
    step = gu(p) * learning_rate
    p[0] = p[0] - gu(p) * learning_rate
    p[1] = p[1] - gv(p) * learning_rate
  print("Local minimum occurs at {} error {}".format(p,f(p)))

def angle_dir(line_pts, pt):
  h = line_pts[0,:] - line_pts[1,:]
  p = line_pts[0,:] - pt
  return np.sign(np.cross(h,p))

def angle_dir_pts(line_pts, pts):
  h = line_pts[0,:] - line_pts[1,:]
  p = line_pts[0,:] * np.ones(pts.shape) - pts
  return np.sign(np.cross(h,p))


def sgd_trial(x,n,extra):
  learning_rate = 0.01
  l = np.random.rand(2,2) * 2 - 1
  h = l[0,:] - l[1,:]
  y = angle_dir_pts(l,x)
  x = np.hstack((np.ones((n+extra, 1)), x))
  w = np.array([0.0, 0.0, 0.0])

  wp = w
  epoc = 0
  while True:
    epoc += 1
    wp = w
    for i in range(n):
      ni = i
      xn = x[ni,:]
      yn = y[ni]
      g = (-yn * xn) / (1 + math.e ** (yn * w.dot(xn)))
      w = w - g * learning_rate
    if (la.norm(wp-w) < .01):
      break

  e_sum = 0.0
  for i in range(n, n+extra -1):
    xn = x[i,:]
    yn = y[i]
    e_sum += np.log(1 + math.e ** (-yn * w.dot(xn)))
  e_out = e_sum / (extra -1)
  return (epoc, e_out)

def prob_8():
  n = 100 
  extra = 100
  x = np.random.rand(n + extra,2) * 2 - 1

  e_avg = 0.0
  epoc_avg = 0.0
  trials = 15.0
  for i in range(int(trials)):
    epoc, e_out = sgd_trial(x,n,extra)
    e_avg += e_out
    epoc_avg += epoc

  e_avg = e_avg / trials
  epoc_avg = epoc_avg / trials
  print("Eout {} Epoc {}".format(e_avg, epoc_avg))
  #print(y)
  #a = np.array([[-1,-1],[1,1]])
  #b = np.array([[2,1],[2,3]])
  #print(angle_dir_2(a,b))

    


#prob_5()
#prob_7()
prob_8()
