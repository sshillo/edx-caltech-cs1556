import math 
import numpy as np


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

prob_5()
