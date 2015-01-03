import numpy as np
import math as m

limit  = 1000

for n in range(limit):
  target = 2 ** n
  c_fn = 0.
  print(int(m.floor(n ** .5)))
  for k in range(int(m.floor(n ** .5))):
    c_fn += n ** k
    #c_fn += m.factorial(n) / (m.factorial(k) * m.factorial(n-k) * 1.)
  print("diff {}".format(target - c_fn))
  if c_fn > target:
    print("c is not possible growth fn")
    break


