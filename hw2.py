import numpy as np

#ex 8-10

def f(d):
  return np.sign(d[0,1] ** 2 + d[0,2] ** 2 - 0.6)

def build_data(data_size):
  x = np.random.rand(data_size, 2) * 2 - 1
  x0 = np.ones(data_size)
  x = np.mat(np.column_stack((x0, x)))
  y = np.mat( [ f(d) for d in x] ).T
  return x,y

def add_noisy_data(y):
  data_size = len(y)
  for i in np.random.randint(0, data_size, data_size * .1):
    y[i] = y[i] * -1

def lin_reg(x,y):
  return np.linalg.inv(x.T * x) * x.T * y

def error_count(w, x, y):
  e_count = 0
  diff = np.sign(w.T * x.T) - y.T 
  for i in np.nditer(diff):
    if i != 0:
      e_count += 1
  return e_count

def ex_8():
  data_size = 1000
  e_count = 0.
  trials = 100

  for i in range(trials):
    x,y = build_data(data_size)

    #create noise
    for i in np.random.randint(0, data_size, data_size * .1):
      y[i] = y[i] * -1

    #w = (x' * x) ^ -1 * x' * y
    w = np.linalg.inv(x.T * x) * x.T * y
    diff = np.sign(w.T * x.T) - y.T 
    for i in np.nditer(diff):
      if i == 0:
        e_count += 1
    
  print("mean error {}".format(e_count/(data_size * trials)))

def ex_9_and_10():
  data_size = 1000
  trials = 100
  w_tot = None
  e_out = 0.0
  g_errors = [0.0 for i in range(5)]

  for i in range(trials):
    x,y = build_data(data_size)
    add_noisy_data(y)
    #transform
    transform = lambda x1,x2: [1, x1, x2, x1 * x2, x1 ** 2, x2 ** 2]
    x_tilde = np.mat([ transform(d[0,1], d[0,2])  for d in x]) #(1, x1, x2, x1x2, x1 ^2, x2 ^2)
    w = lin_reg(x_tilde,y)

    h_vals = np.sign(w.T * x_tilde.T).T
    g = np.mat([[-1,-.05,.08,.13,1.5,1.5],
         [-1,-.05,.08,.13,1.5,15],
         [-1,-.05,.08,.13,15,1.5],
         [-1,-1.5,.08,.13,.05,.05],
         [-1,-.05,.08,1.5,.15,.15]])

    g_errors += np.mat([error_count(gw.T,x_tilde,h_vals) for gw in g])

    #ex_10
    x2,y2 = build_data(data_size)
    add_noisy_data(y2)
    x_tilde2 = np.mat([ transform(d[0,1], d[0,2])  for d in x2]) #(1, x1, x2, x1x2, x1 ^2, x2 ^2)
    e_out += error_count(w, x_tilde2, y2)

  print("g errors {}".format(g_errors))
  print("e_out {}".format(e_out/(float(data_size) * trials)))
    


#ex_8()
ex_9_and_10()
