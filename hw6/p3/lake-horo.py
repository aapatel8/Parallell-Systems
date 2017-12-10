#!/bin/python

"""
/*   Group info:
 *   kmishra Kushagra Mishra
 *   pranjan Pritesh Ranjan
 *   aapatel8 Akshit Patel
 */
"""

#Import libraries for simulation
import tensorflow as tf
import numpy as np
import sys
import time
import horovod.tensorflow as hvd

#Imports for visualization
import PIL.Image

# Get the command line args
if len(sys.argv) != 4:
	print "Usage:", sys.argv[0], "N npebs num_iter"
	sys.exit()
	
N = int(sys.argv[1])
npebs = int(sys.argv[2])
num_iter = int(sys.argv[3])

hvd.init()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(hvd.local_rank())

sess = tf.InteractiveSession() #CPU version
#sess = tf.InteractiveSession(config=config) #Use only for capability 3.0 GPU


def DisplayArray(a, fmt='jpeg', rng=[0,1]):
  """Display an array as a picture."""
  a = (a - rng[0])/float(rng[1] - rng[0])*255
  a = np.uint8(np.clip(a, 0, 255))
  with open("lake_py_{0}.jpg".format(hvd.rank()), "w") as f:    # TODO: Chage name
      PIL.Image.fromarray(a).save(f, "jpeg")


# Computational Convenience Functions
def make_kernel(a):
  """Transform a 2D array into a convolution kernel"""
  a = np.asarray(a)
  a = a.reshape(list(a.shape) + [1,1])
  return tf.constant(a, dtype=1)

def simple_conv(x, k):
  """A simplified 2D convolution operation"""
  x = tf.expand_dims(tf.expand_dims(x, 0), -1)
  y = tf.nn.depthwise_conv2d(x, k, [1, 1, 1, 1], padding='SAME')
  return y[0, :, :, 0]

def laplace(x):
  """Compute the 2D laplacian of an array"""
#  5 point stencil #
  five_point = [[0.0, 1.0, 0.0],
                [1.0, -4., 1.0],
                [0.0, 1.0, 0.0]]

#  9 point stencil #
  nine_point = [[0.25, 1.0, 0.25],
                [1.00, -5., 1.00],
                [0.25, 1.0, 0.25]]

  thirteen_point = [[0.0,   0.0,   0.125, 0.0,   0.0],
                    [0.0,   0.250, 1.0,   0.250, 0.0],
                    [0.125, 1.0,   -5.50, 1.0,   0.125],
                    [0.0,   0.250, 1.0,   0.250, 0.0],
                    [0.0,   0.0,   0.125, 0.0,   0.0]]
                
  #laplace_k = make_kernel(nine_point)

  laplace_k = make_kernel(thirteen_point)
  return simple_conv(x, laplace_k)


# Initial Conditions -- some rain drops hit a pond

# Set everything to zero
u_init  = np.zeros([N, N], dtype=np.float32)
ut_init = np.zeros([N+2, N], dtype=np.float32)

np_send_buf = np.zeros([2,N], dtype=np.float32)
np_recv_buf_0 = np.zeros([2,N], dtype=np.float32)
np_recv_buf_1 = np.zeros([2,N], dtype=np.float32)

# Some rain drops hit a pond at random points
for n in range(npebs):
  a,b = np.random.randint(0, N, 2)
  u_init[a,b] = np.random.uniform()

# Add the extra rows at bottom for rank 0 
zero_pad = np.zeros([2, N], dtype=np.float32)
if hvd.rank() == 0:
    u_init = np.concatenate((u_init, zero_pad), axis=0)
else:  # Add the extra rows at top for rank 1 
    u_init = np.concatenate((zero_pad, u_init), axis=0)

# Parameters:
# eps -- time resolution
# damping -- wave damping
eps = tf.placeholder(tf.float32, shape=())
damping = tf.placeholder(tf.float32, shape=())

# Create variables for simulation state
U  = tf.Variable(u_init)
Ut = tf.Variable(ut_init)

# Create variables for broadcast operation.
tf_send_buf = tf.Variable(np_send_buf, "send_buf")
tf_recv_buf_0 = tf.Variable(np_recv_buf_0, "recv_buf_0")
tf_recv_buf_1 = tf.Variable(np_recv_buf_1, "recv_buf_1")


# Discretized PDE update rules
U_ = U + eps * Ut
Ut_ = Ut + eps * (laplace(U) - damping * Ut)

if hvd.rank() == 0:
    send_buf = U[-4:-2, :]
else:
    send_buf = U[2:4, :]

# Operation to update the state
r0_step = tf.group(
  U.assign(tf.concat(values=[tf.slice(U, [0,0],[N,N]), tf_recv_buf_0], axis=0)),
  U.assign(U_),
  Ut.assign(Ut_),
  tf_send_buf.assign(send_buf))
  

r1_step = tf.group(
  U.assign(tf.concat(values=[tf_recv_buf_1, tf.slice(U, [2,0], [N,N])], axis=0),
  U.assign(U_),
  Ut.assign(Ut_),
  tf_send_buf.assign(send_buf)
  )

broadcast = tf.group(
  tf.assign(tf_recv_buf_1, hvd.broadcast(tf_send_buf, 0)),
  tf.assign(tf_recv_buf_0, hvd.broadcast(tf_send_buf, 1)))

# Initialize state to initial conditions
tf.global_variables_initializer().run()

# Run num_iter steps of PDE
start = time.time()
for i in range(num_iter):
  broadcast.run()
  # Step simulation
  if hvd.rank() == 0:
    r0_step.run({eps: 0.06, damping: 0.03})
  else:
    r1_step.run({eps: 0.06, damping: 0.03})

end = time.time()
print('Elapsed time: {} seconds'.format(end - start))  
if hvd.rank() == 0:
  DisplayArray(U.eval()[:-2,:], rng=[-0.1, 0.1])
else:
  DisplayArray(U.eval()[2:, :], rng=[-0.1, 0.1])