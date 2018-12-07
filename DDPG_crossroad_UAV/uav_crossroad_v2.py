import os
import time

import cv2
import numpy as np
import tensorflow as tf

"""
2018-12-02
https://github.com/MorvanZhou/
Reinforcement-learning-with-tensorflow/tree/master/contents/
9_Deep_Deterministic_Policy_Gradient_DDPG/DDPG.py

2018-12-03 
Deep Reinforcement Learning for UAV-Enabled vehicular Networks in Smart Cities. 
Ming Zhu

YonvZen1943
2018-12-03 Yonv Reproduce. No elegant enough.
2018-12-04 debug state
2018-12-05 ddpg_v2
"""

a1, a2 = 9.6, 0.28  # los_link()
b1, b2 = 3, 1e2  # h_local()  # 20 dB == 10 * lg(x), x == 1e2

error2 = 1e-10  # sinr(),  # -130 dBm/Hz == 10 * lg(x/1e-3), x == 1e-10
b_max = 1 * 1000
xi = 0.3
tao = 15
M = 4  # def state_transition()

d_grid = 30
p_max = 6
z_min, z_max = 6, 50  # def state_transition()
traffic_gap = 4
lamb1357 = np.array((0.3, 0.6, 0.4, 1.0)) / traffic_gap  # vehicles poisson
max_pass = 196 // traffic_gap  # cross road pass
g_lsr = np.array((0.3, 0.4, 0.3)) * 1.5  # lsr, left, straight, right

cross_indicies = ((0, 6, 2), (1, 6, 4),
                  (2, 4, 6), (3, 2, 6),
                  (4, 0, 4), (5, 0, 2),
                  (6, 2, 0), (7, 4, 0),)
dict_draw_uav = {8: (3, 3),
                 0: (5, 2), 1: (5, 4),
                 2: (4, 5), 3: (2, 5),
                 4: (1, 4), 5: (1, 2),
                 6: (2, 1), 7: (4, 1), }
xy_distances = np.array([
    [0, 1, 5, 8, 10, 9, 5, 2, ],
    [1, 0, 2, 5, 9, 10, 8, 5, ],
    [5, 8, 10, 9, 5, 2, 0, 1, ],
    [2, 5, 9, 10, 8, 5, 1, 0, ],
    [10, 9, 5, 2, 0, 1, 5, 8, ],
    [9, 10, 8, 5, 1, 0, 2, 5, ],
    [5, 2, 0, 1, 5, 8, 10, 9, ],
    [8, 5, 1, 0, 2, 5, 9, 10, ],
    [2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, ]
], np.float32) ** 0.5 * d_grid

cv2_name = 'crossroad'
rd_uniform = np.random.uniform
rd_poisson = np.random.poisson
rd_normals = np.random.normal


def vehicles_pass(vehicles, out, in0, in1, in2):
    v_pass = min(vehicles[out], max_pass)
    vehicles[out] -= v_pass

    rate_ins = rd_uniform(0.5, 1.0, 3) * g_lsr
    rate_int = (rate_ins / np.sum(rate_ins) * v_pass).astype(np.int)
    rate_int[1] += int(v_pass - np.sum(rate_int))

    vehicles[np.array((in0, in1, in2))] = rate_int


def vehicle_arrival(vehicles):
    poisson = np.array(rd_poisson(100, 4) * lamb1357, np.int)
    vehicles[1::2] += poisson

    # return np.exp(lamb) * lamb ** k / np.math.factorial(k)


def draw_crossroad(n, pos, z, switch):  # n: vehicles_number
    cross = np.zeros((7, 7), np.int)
    vehicle_arrival(n)

    if switch:
        vehicles_pass(n, 1, 2, 4, 6)
        vehicles_pass(n, 5, 6, 0, 2)
    else:
        vehicles_pass(n, 3, 4, 6, 0)
        vehicles_pass(n, 7, 0, 2, 4)

    for (k, i, j) in cross_indicies:  # draw vehicles
        cross[i, j] = n[k]
    id_i, id_j = dict_draw_uav[pos]  # draw uav

    img_show = cross * (255 / np.max(n))
    # if switch:  # draw traffic pass
    #     img_show[:, 3] = 64
    # else:
    #     img_show[3, :] = 64
    img_show[id_i, id_j] = z * 3 + 100  # draw uav
    cv2.imshow(cv2_name, img_show.astype(np.uint8))
    cv2.waitKey(1)

    return n


def h_local(pos, z_high):  # b1, b2
    xy_distance = xy_distances[pos] + 1e-10  # keep the denominator not zero
    distance = (xy_distance ** 2 + z_high ** 2) ** 0.5

    f1 = np.ones(8, np.float32) if pos == 8 \
        else 1 / (1 + a1 * np.exp(-a2 * (np.tanh(z_high / (xy_distance + 1e-10)) - a1)))
    # f1 = 1 / (1 + a1 * np.exp(-a2 * (np.tanh(z_high / (xy_distance + 1e-10)) - a1)))

    res = (f1 + (1 - f1) * b2) * (distance ** -b1)
    return res


def bandwidth_allocation_state(switch):  # xi
    res = [3, 1, 3, 2, 3, 1, 3, 2] if switch else [3, 2, 3, 1, 3, 2, 3, 1]
    return np.array(res) * xi


def state_transition(x, z, m):  # M=4, z_min, z_max
    # m = 0 if np.isnan(m) else m
    m_M = int(m - M)

    if -M <= m <= M:
        res = (x, np.clip(z + m, z_min, z_max))
    elif m_M == 1:
        res = (x, z)
    elif m_M == 2:
        res = ((x + 1) % 8, z)
    elif m_M == 3:
        res = ((x - 1) % 8, z)
    else:
        res = ((m_M - 5) % 9, z)

    return res


def reward(b, p, n, pos, z, switch):
    # b = bandwidth_allocation_state(switch)
    h = h_local(pos, z)
    nb = b * n
    sinr = p * h / error2  # SINR
    return b_max * np.sum(nb * np.log(1 + sinr)) / np.sum(nb)


MAX_EPISODES = 256
MAX_EP_STEPS = 256
LR_A = 0.001  # learning rate for actor
LR_C = 0.002  # learning rate for critic
GAMMA = 0.9  # reward discount
TAU = 0.01  # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32
gpu_id = 0


class DDPG(object):
    def __init__(self, a_dim, s_dim, ):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)  # choose GPU:0
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # ignore the TensorFlow log messages
        self.sess = tf.Session()
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # recover the TensorFlow log messages

        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0

        self.a_dim, self.s_dim = a_dim, s_dim
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        self.a = self._build_a(self.S, )
        q = self._build_c(self.S, self.a, )
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic')
        ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)  # soft replacement

        def ema_getter(getter, name, *args, **kwargs):
            return ema.average(getter(name, *args, **kwargs))

        target_update = [ema.apply(a_params), ema.apply(c_params)]  # soft update operation
        a_ = self._build_a(self.S_, reuse=True, custom_getter=ema_getter)  # replaced target parameters
        q_ = self._build_c(self.S_, a_, reuse=True, custom_getter=ema_getter)

        a_loss = - tf.reduce_mean(q)  # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=a_params)

        with tf.control_dependencies(target_update):  # soft replacement happened at here
            q_target = self.R + GAMMA * q_
            td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
            self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=c_params)

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def learn(self):
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Actor', reuse=reuse, custom_getter=custom_getter):
            net = tf.layers.dense(s, 128, activation=tf.nn.relu6, trainable=trainable)

            ten = tf.layers.dense(net, 128, activation=tf.nn.relu6, trainable=trainable)
            net = tf.layers.dense(ten, 128, activation=tf.nn.relu6, trainable=trainable) + net
            ten = tf.layers.dense(net, 128, activation=tf.nn.relu6, trainable=trainable)
            net = tf.layers.dense(ten, 128, activation=tf.nn.relu6, trainable=trainable) + net
            net = tf.layers.dense(net, 128, activation=tf.nn.relu6, trainable=trainable)

            a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return a

    def _build_c(self, s, a, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Critic', reuse=reuse, custom_getter=custom_getter):
            n_l1 = 30
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net = tf.layers.dense(net, 128, trainable=trainable)

            ten = tf.layers.dense(net, 128, trainable=trainable)
            net = tf.layers.dense(ten, 128, trainable=trainable) + net
            net = tf.layers.dense(net, 128, trainable=trainable)

            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)


def run():
    s_dim = 11
    a_dim = 17
    # state_dim = 11  # 1 traffic_light + 2 (uav_pos, uav_high) + 8 crossroad_vehicle_number
    # action_dim = 17  # motion1, power8, bandwidth8

    ddpg = DDPG(a_dim, s_dim)
    var = 1.0
    timer = time.time()
    logger_gap = 16
    loggers = list()
    cv2.namedWindow(cv2_name)
    for i in range(MAX_EPISODES):
        if i == 128:
            var = 0.5

        uav_pos, z_high = i % 9, (i % 8) / 7 * (z_max - z_min) + z_min
        motion, powers, bandwd = 0, 0, 0
        vehicles = rd_poisson(100, 8)
        s = np.concatenate(((0, uav_pos, z_high,), vehicles), axis=0)
        # 1 traffic_light + 2 (uav_pos, uav_high) + 8 crossroad_vehicle_number

        uav_info = list()
        for j in range(MAX_EP_STEPS):
            # Add exploration noise
            a = ddpg.choose_action(s)
            a = np.clip(np.random.normal(a, var), -1, +1)  # add randomness to action selection for exploration

            traffic_switch = (j // traffic_gap) % 2
            motion = a[0] * 10 + 6.0
            powers = a[1:9] + 1.01  # avoid overflow
            powers = powers * (p_max / np.sum(vehicles * powers))
            bandwd = a[9:17] + 1.01  # avoid overflow
            bandwd = bandwd * (b_max / np.sum(vehicles * bandwd))

            uav_pos, z_high = state_transition(uav_pos, z_high, motion)
            r = reward(bandwd, powers, vehicles, uav_pos, z_high, traffic_switch)
            r = np.tanh((r - 11000) * 0.001)
            vehicles = draw_crossroad(vehicles, uav_pos, z_high, traffic_switch)
            s_ = np.concatenate(((traffic_switch, uav_pos, z_high), vehicles), axis=0)
            uav_info.append(r)

            # s_, r, done, info = env.step(a)
            # s_, r = env.step(a)[0:2]
            ddpg.store_transition(s, a, r, s_)

            if ddpg.pointer > MEMORY_CAPACITY:
                var *= .9999  # decay the action randomness
                ddpg.learn()
            if j < logger_gap:
                # reward1, uav_position1, uav_z_high1,
                # rho_powers8, bandwidth8, state_vehicle_number8
                logger = np.concatenate((s, powers, bandwd), axis=0)
                loggers.append(logger)
            s = s_
        ave_reward = np.average(uav_info)
        # ave_reward = 0 if np.isnan(ave_reward) else ave_reward
        print("I %i Reward %.3f  Var %.2e" % (i, ave_reward, var),
              "   max_veh %3i  uav_pos_z %i %2i   pow %.3f  bdw %.3f" %
              (np.max(vehicles), uav_pos, int(z_high), np.average(powers), np.average(bandwd)))
    print('Running time: ', time.time() - timer)

    # reward, traffic_switch, uav_position, uav_z_high, vehicle_number, rho_powers, bandwidth
    logger_path = 'V2_uav_info_r_t_pos_z_n_p_b.npy'
    np.save(logger_path, loggers)
    print("SAVE loggers to:", logger_path)


if __name__ == '__main__':
    # from uav_crossroad import run
    # from DDPG import run

    run()
