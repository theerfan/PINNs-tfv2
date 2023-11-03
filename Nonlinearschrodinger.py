"""
@author: Maziar Raissi
"""

import sys

# sys.path.insert(0, "../../Utilities/")

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from pyDOE import lhs
from Utilities.plotting import newfig, savefig
from mpl_toolkits.mplot3d import Axes3D
import time
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import tensorflow_probability as tfp


np.random.seed(1234)
tf.random.set_seed(1234)

tf.compat.v1.disable_eager_execution()


class ScipyOptimizerInterfaceReplacement:
    def __init__(self, loss, method="L-BFGS-B", options=None):
        self.loss = loss
        self.method = method
        self.options = options or {}
        self.results = None

    def minimize(self, session=None, feed_dict=None, **kwargs):
        # Ensure the loss is a zero-argument callable
        value_and_gradients_fn = tfp.math.value_and_gradients_function(self.loss)

        # Initial position (you might need to adjust this depending on how you handle variables)
        initial_position = tf.Variable(tf.zeros_like(self.loss), dtype=tf.float32)

        # Extract options
        max_iterations = self.options.get("maxiter", 1000)
        f_relative_tolerance = self.options.get("ftol", 1.0 * np.finfo(float).eps)
        max_line_search_iterations = self.options.get("maxls", 50)

        # Run the optimizer
        self.results = tfp.optimizer.lbfgs_minimize(
            value_and_gradients_fn,
            initial_position=initial_position,
            max_iterations=max_iterations,
            f_relative_tolerance=f_relative_tolerance,
            max_line_search_iterations=max_line_search_iterations,
        )


class PhysicsInformedNN:
    # Initialize the class
    # x0 is the initial condition
    # ub is the upper bound of the domain
    # lb is the lower bound of the domain
    # X_f is the collocation points
    # layers is the number of neurons in each layer
    # v0 is the initial condition
    # u0 is the boundary condition
    def __init__(self, x0, u0, v0, tb, X_f, layers, lb: np.array, ub: np.array):
        X0 = np.concatenate((x0, 0 * x0), 1)  # (x0, 0)
        X_lb = np.concatenate((0 * tb + lb[0], tb), 1)  # (lb[0], tb)
        X_ub = np.concatenate((0 * tb + ub[0], tb), 1)  # (ub[0], tb)

        self.lb = lb
        self.ub = ub

        self.x0 = X0[:, 0:1]
        self.t0 = X0[:, 1:2]

        self.x_lb = X_lb[:, 0:1]
        self.t_lb = X_lb[:, 1:2]

        self.x_ub = X_ub[:, 0:1]
        self.t_ub = X_ub[:, 1:2]

        self.x_f = X_f[:, 0:1]
        self.t_f = X_f[:, 1:2]

        self.u0 = u0
        self.v0 = v0

        # Initialize NNs
        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers)

        # tf Placeholders
        self.x0_tf = tf.compat.v1.placeholder(
            tf.float32, shape=[None, self.x0.shape[1]]
        )
        self.t0_tf = tf.compat.v1.placeholder(
            tf.float32, shape=[None, self.t0.shape[1]]
        )

        self.u0_tf = tf.compat.v1.placeholder(
            tf.float32, shape=[None, self.u0.shape[1]]
        )
        self.v0_tf = tf.compat.v1.placeholder(
            tf.float32, shape=[None, self.v0.shape[1]]
        )

        self.x_lb_tf = tf.compat.v1.placeholder(
            tf.float32, shape=[None, self.x_lb.shape[1]]
        )
        self.t_lb_tf = tf.compat.v1.placeholder(
            tf.float32, shape=[None, self.t_lb.shape[1]]
        )

        self.x_ub_tf = tf.compat.v1.placeholder(
            tf.float32, shape=[None, self.x_ub.shape[1]]
        )
        self.t_ub_tf = tf.compat.v1.placeholder(
            tf.float32, shape=[None, self.t_ub.shape[1]]
        )

        self.x_f_tf = tf.compat.v1.placeholder(
            tf.float32, shape=[None, self.x_f.shape[1]]
        )
        self.t_f_tf = tf.compat.v1.placeholder(
            tf.float32, shape=[None, self.t_f.shape[1]]
        )

        # tf Graphs
        # Size of t = (batch_size, 1)
        # Size of x = (batch_size, 1) 
        self.u0_pred, self.v0_pred, _, _ = self.net_uv(self.x0_tf, self.t0_tf)

        (
            self.u_lb_pred,
            self.v_lb_pred,
            self.u_x_lb_pred,
            self.v_x_lb_pred,
        ) = self.net_uv(self.x_lb_tf, self.t_lb_tf)

        (
            self.u_ub_pred,
            self.v_ub_pred,
            self.u_x_ub_pred,
            self.v_x_ub_pred,
        ) = self.net_uv(self.x_ub_tf, self.t_ub_tf)

        self.f_u_pred, self.f_v_pred = self.net_f_uv(self.x_f_tf, self.t_f_tf)

        # Loss
        self.loss = (
            tf.reduce_mean(tf.square(self.u0_tf - self.u0_pred))
            + tf.reduce_mean(tf.square(self.v0_tf - self.v0_pred))
            + tf.reduce_mean(tf.square(self.u_lb_pred - self.u_ub_pred))
            + tf.reduce_mean(tf.square(self.v_lb_pred - self.v_ub_pred))
            + tf.reduce_mean(tf.square(self.u_x_lb_pred - self.u_x_ub_pred))
            + tf.reduce_mean(tf.square(self.v_x_lb_pred - self.v_x_ub_pred))
            + tf.reduce_mean(tf.square(self.f_u_pred))
            + tf.reduce_mean(tf.square(self.f_v_pred))
        )

        # Optimizers
        self.optimizer = ScipyOptimizerInterfaceReplacement(
            self.loss,
            method="L-BFGS-B",
            options={
                "maxiter": 50000,
                "maxfun": 50000,
                "maxcor": 50,
                "maxls": 50,
                "ftol": 1.0 * np.finfo(float).eps,
            },
        )

        self.optimizer_Adam = tf.compat.v1.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

        # tf session
        self.sess = tf.compat.v1.Session(
            config=tf.compat.v1.ConfigProto(
                allow_soft_placement=True, log_device_placement=True
            )
        )

        init = tf.compat.v1.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for i in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[i], layers[i + 1]])
            b = tf.Variable(
                tf.zeros([1, layers[i + 1]], dtype=tf.float32), dtype=tf.float32
            )
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(
            tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev),
            dtype=tf.float32,
        )

    def neural_net(self, X):
        num_layers = len(self.weights) + 1

        # The function first normalizes the input X to the range [-1, 1] 
        # using the attributes self.lb (lower bound) and self.ub (upper bound) of the class to which this function belongs.
        # H = normalized X
        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for i in range(0, num_layers - 2):
            W = self.weights[i]
            b = self.biases[i]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))

        # Last layer does not have an activation function
        W = self.weights[-1]
        b = self.biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    # Extracts two fields u and v from the output of the neural network
    # Basically, the input of this neural network is the concatenation of x and t
    def net_uv(self, x, t):
        X = tf.concat([x, t], 1)

        uv = self.neural_net(X)
        u = uv[:, 0:1]
        v = uv[:, 1:2]

        # Calculates the first derivative of u and v with respect to x
        u_x = tf.gradients(u, x)[0]
        v_x = tf.gradients(v, x)[0]

        return u, v, u_x, v_x

    # Calculating the second derivative of u and v with respect to x
    # and the first derivative of u and v with respect to t
    def net_f_uv(self, x, t):
        u, v, u_x, v_x = self.net_uv(x, t)

        u_t = tf.gradients(u, t)[0]
        u_xx = tf.gradients(u_x, x)[0]

        v_t = tf.gradients(v, t)[0]
        v_xx = tf.gradients(v_x, x)[0]

        # Then check the PDE conditions
        f_u = u_t + 0.5 * v_xx + (u**2 + v**2) * v
        f_v = v_t - 0.5 * u_xx - (u**2 + v**2) * u

        return f_u, f_v

    def callback(self, loss):
        print("Loss:", loss)

    def train(self, nIter):
        tf_dict = {
            self.x0_tf: self.x0,
            self.t0_tf: self.t0,
            self.u0_tf: self.u0,
            self.v0_tf: self.v0,
            self.x_lb_tf: self.x_lb,
            self.t_lb_tf: self.t_lb,
            self.x_ub_tf: self.x_ub,
            self.t_ub_tf: self.t_ub,
            self.x_f_tf: self.x_f,
            self.t_f_tf: self.t_f,
        }

        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)

            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                print("It: %d, Loss: %.3e, Time: %.2f" % (it, loss_value, elapsed))
                start_time = time.time()

        self.optimizer.minimize(
            self.sess,
            feed_dict=tf_dict,
            fetches=[self.loss],
            loss_callback=self.callback,
        )

    def predict(self, X_star):
        tf_dict = {self.x0_tf: X_star[:, 0:1], self.t0_tf: X_star[:, 1:2]}

        u_star = self.sess.run(self.u0_pred, tf_dict)
        v_star = self.sess.run(self.v0_pred, tf_dict)

        tf_dict = {self.x_f_tf: X_star[:, 0:1], self.t_f_tf: X_star[:, 1:2]}

        f_u_star = self.sess.run(self.f_u_pred, tf_dict)
        f_v_star = self.sess.run(self.f_v_pred, tf_dict)

        return u_star, v_star, f_u_star, f_v_star


if __name__ == "__main__":
    noise = 0.0

    # Domain bounds
    lb = np.array([-5.0, 0.0])
    ub = np.array([5.0, np.pi / 2])

    # N0: Number of points for the initial condition
    N0 = 50
    # N_b: Number of points for the boundary condition
    N_b = 50
    # N_f: Number of points for the collocation
    # NOTE: Coallocation: The idea is to choose a finite-dimensional space of candidate solutions
    # (usually polynomials up to a certain degree) and a number of points in the domain (called collocation points),
    # and to select that solution which satisfies the given equation at the collocation points. 
    N_f = 20000

    # This is why the first layer of the neural network has 2 neurons
    # Because our initial data is literally two dimensional!
    layers = [2, 100, 100, 100, 100, 2]

    data = scipy.io.loadmat("main/Data/NLS.mat")

    # t : time
    t = data["tt"].flatten()[:, None]
    # x : space
    x = data["x"].flatten()[:, None]
    # uu : complex wave
    Exact = data["uu"]
    # u : real part of complex wave
    Exact_u = np.real(Exact)
    # v: imaginary part of complex wave
    Exact_v = np.imag(Exact)
    # h: magnitude of complex wave
    Exact_h = np.sqrt(Exact_u**2 + Exact_v**2)

    # Putting the data in a 2D grid
    X, T = np.meshgrid(x, t)

    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    u_star = Exact_u.T.flatten()[:, None]
    v_star = Exact_v.T.flatten()[:, None]
    h_star = Exact_h.T.flatten()[:, None]

    ###########################

    idx_x = np.random.choice(x.shape[0], N0, replace=False)
    x0 = x[idx_x, :]
    u0 = Exact_u[idx_x, 0:1]
    v0 = Exact_v[idx_x, 0:1]

    idx_t = np.random.choice(t.shape[0], N_b, replace=False)
    # tb: times at the boundary
    tb = t[idx_t, :]

    # lhs = latin-hypercube design
    X_f = lb + (ub - lb) * lhs(2, N_f)

    model = PhysicsInformedNN(x0, u0, v0, tb, X_f, layers, lb, ub)

    start_time = time.time()
    model.train(50000)
    elapsed = time.time() - start_time
    print("Training time: %.4f" % (elapsed))

    u_pred, v_pred, f_u_pred, f_v_pred = model.predict(X_star)
    h_pred = np.sqrt(u_pred**2 + v_pred**2)

    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
    error_v = np.linalg.norm(v_star - v_pred, 2) / np.linalg.norm(v_star, 2)
    error_h = np.linalg.norm(h_star - h_pred, 2) / np.linalg.norm(h_star, 2)
    print("Error u: %e" % (error_u))
    print("Error v: %e" % (error_v))
    print("Error h: %e" % (error_h))

    U_pred = griddata(X_star, u_pred.flatten(), (X, T), method="cubic")
    V_pred = griddata(X_star, v_pred.flatten(), (X, T), method="cubic")
    H_pred = griddata(X_star, h_pred.flatten(), (X, T), method="cubic")

    FU_pred = griddata(X_star, f_u_pred.flatten(), (X, T), method="cubic")
    FV_pred = griddata(X_star, f_v_pred.flatten(), (X, T), method="cubic")

    ######################################################################
    ############################# Plotting ###############################
    ######################################################################

    X0 = np.concatenate((x0, 0 * x0), 1)  # (x0, 0)
    X_lb = np.concatenate((0 * tb + lb[0], tb), 1)  # (lb[0], tb)
    X_ub = np.concatenate((0 * tb + ub[0], tb), 1)  # (ub[0], tb)
    X_u_train = np.vstack([X0, X_lb, X_ub])

    fig, ax = newfig(1.0, 0.9)
    ax.axis("off")

    ####### Row 0: h(t,x) ##################
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1 - 0.06, bottom=1 - 1 / 3, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs0[:, :])

    h = ax.imshow(
        H_pred.T,
        interpolation="nearest",
        cmap="YlGnBu",
        extent=[lb[1], ub[1], lb[0], ub[0]],
        origin="lower",
        aspect="auto",
    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)

    ax.plot(
        X_u_train[:, 1],
        X_u_train[:, 0],
        "kx",
        label="Data (%d points)" % (X_u_train.shape[0]),
        markersize=4,
        clip_on=False,
    )

    line = np.linspace(x.min(), x.max(), 2)[:, None]
    ax.plot(t[75] * np.ones((2, 1)), line, "k--", linewidth=1)
    ax.plot(t[100] * np.ones((2, 1)), line, "k--", linewidth=1)
    ax.plot(t[125] * np.ones((2, 1)), line, "k--", linewidth=1)

    ax.set_xlabel("$t$")
    ax.set_ylabel("$x$")
    leg = ax.legend(frameon=False, loc="best")
    #    plt.setp(leg.get_texts(), color='w')
    ax.set_title("$|h(t,x)|$", fontsize=10)

    ####### Row 1: h(t,x) slices ##################
    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(top=1 - 1 / 3, bottom=0, left=0.1, right=0.9, wspace=0.5)

    ax = plt.subplot(gs1[0, 0])
    ax.plot(x, Exact_h[:, 75], "b-", linewidth=2, label="Exact")
    ax.plot(x, H_pred[75, :], "r--", linewidth=2, label="Prediction")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$|h(t,x)|$")
    ax.set_title("$t = %.2f$" % (t[75]), fontsize=10)
    ax.axis("square")
    ax.set_xlim([-5.1, 5.1])
    ax.set_ylim([-0.1, 5.1])

    ax = plt.subplot(gs1[0, 1])
    ax.plot(x, Exact_h[:, 100], "b-", linewidth=2, label="Exact")
    ax.plot(x, H_pred[100, :], "r--", linewidth=2, label="Prediction")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$|h(t,x)|$")
    ax.axis("square")
    ax.set_xlim([-5.1, 5.1])
    ax.set_ylim([-0.1, 5.1])
    ax.set_title("$t = %.2f$" % (t[100]), fontsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.8), ncol=5, frameon=False)

    ax = plt.subplot(gs1[0, 2])
    ax.plot(x, Exact_h[:, 125], "b-", linewidth=2, label="Exact")
    ax.plot(x, H_pred[125, :], "r--", linewidth=2, label="Prediction")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$|h(t,x)|$")
    ax.axis("square")
    ax.set_xlim([-5.1, 5.1])
    ax.set_ylim([-0.1, 5.1])
    ax.set_title("$t = %.2f$" % (t[125]), fontsize=10)

    # savefig('./figures/NLS')
