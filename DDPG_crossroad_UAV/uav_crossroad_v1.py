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
    # if i == 3 or i == 7:
    #     return xi if switch else 2 * xi
    # elif i == 1 or i == 5:
    #     return 2 * xi if switch else xi
    # else:
    #     return 3 * xi
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


"""DDPG"""


class Actor(object):
    def __init__(self, S, S_, sess, action_dim, learning_rate, replacement):
        self.S = S
        self.S_ = S_
        self.sess = sess
        self.a_dim = action_dim
        self.lr = learning_rate
        self.replacement = replacement
        self.t_replace_counter = 0

        with tf.variable_scope('Actor'):
            # input s, output a
            self.a = self._build_net(S, scope='eval_net', trainable=True)

            # input s_, output a, get a_ for critic
            self.a_ = self._build_net(S_, scope='target_net', trainable=False)

        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval_net')
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target_net')

        if self.replacement['name'] == 'hard':
            self.t_replace_counter = 0
            self.hard_replace = [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)]
        else:
            self.soft_replace = [tf.assign(t, (1 - self.replacement['tau']) * t + self.replacement['tau'] * e)
                                 for t, e in zip(self.t_params, self.e_params)]

    def _build_net(self, s, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.random_normal_initializer(0., 0.3)
            init_b = tf.constant_initializer(0.1)
            with tf.name_scope('l1'):
                net = tf.layers.dense(s, 64, activation=tf.nn.relu6, trainable=trainable,
                                      kernel_initializer=init_w, bias_initializer=init_b, )
                net = tf.layers.dense(net, 64, activation=tf.nn.relu6, trainable=trainable,
                                      kernel_initializer=init_w, bias_initializer=init_b, )
                net = tf.layers.dense(net, 64, activation=tf.nn.relu6, trainable=trainable,
                                      kernel_initializer=init_w, bias_initializer=init_b, )
            with tf.variable_scope('a'):
                actions = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, kernel_initializer=init_w,
                                          bias_initializer=init_b, name='a', trainable=trainable)
                scaled_a = actions
            return scaled_a

    def learn(self, s):  # batch update
        self.sess.run(self.train_op, feed_dict={self.S: s})

        if self.replacement['name'] == 'soft':
            self.sess.run(self.soft_replace)
        else:
            if self.t_replace_counter % self.replacement['rep_iter_a'] == 0:
                self.sess.run(self.hard_replace)
            self.t_replace_counter += 1

    def choose_action(self, s):
        s = s[np.newaxis, :]  # single state
        return self.sess.run(self.a, feed_dict={self.S: s})[0]  # single action

    def add_grad_to_graph(self, a_grads):
        with tf.variable_scope('policy_grads'):
            # ys = policy;
            # xs = policy's parameters;
            # a_grads = the gradients of the policy to get more Q
            # tf.gradients will calculate dys/dxs with a initial gradients for ys, so this is dq/da * da/dparams
            self.policy_grads = tf.gradients(ys=self.a, xs=self.e_params, grad_ys=a_grads)

        with tf.variable_scope('A_train'):
            opt = tf.train.AdamOptimizer(-self.lr)  # (- learning rate) for ascent policy
            self.train_op = opt.apply_gradients(zip(self.policy_grads, self.e_params))


class Critic(object):
    def __init__(self, S, R, S_, sess, state_dim, action_dim, learning_rate, gamma, replacement, a, a_):
        self.S = S
        self.R = R
        self.S_ = S_
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.replacement = replacement

        with tf.variable_scope('Critic'):
            # Input (s, a), output q
            self.a = a
            self.q = self._build_net(S, self.a, 'eval_net', trainable=True)

            # Input (s_, a_), output q_ for q_target
            self.q_ = self._build_net(S_, a_, 'target_net',
                                      trainable=False)  # target_q is based on a_ from Actor's target_net

            self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval_net')
            self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target_net')

        with tf.variable_scope('target_q'):
            self.target_q = R + self.gamma * self.q_

        with tf.variable_scope('TD_error'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.target_q, self.q))

        with tf.variable_scope('C_train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        with tf.variable_scope('a_grad'):
            self.a_grads = tf.gradients(self.q, a)[0]  # tensor of gradients of each sample (None, a_dim)

        if self.replacement['name'] == 'hard':
            self.t_replace_counter = 0
            self.hard_replacement = [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)]
        else:
            self.soft_replacement = [tf.assign(t, (1 - self.replacement['tau']) * t + self.replacement['tau'] * e)
                                     for t, e in zip(self.t_params, self.e_params)]

    def _build_net(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.random_normal_initializer(0., 0.1)
            init_b = tf.constant_initializer(0.1)

            with tf.variable_scope('l1'):
                n_l1 = 64
                w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], initializer=init_w, trainable=trainable)
                w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], initializer=init_w, trainable=trainable)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=init_b, trainable=trainable)
                net = tf.nn.relu6(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)

                net = tf.layers.dense(net, n_l1, activation=tf.nn.relu6)
                net = tf.layers.dense(net, n_l1, activation=tf.nn.relu6)

            with tf.variable_scope('q'):
                q = tf.layers.dense(net, 1, kernel_initializer=init_w, bias_initializer=init_b,
                                    trainable=trainable)  # Q(s,a)
        return q

    def learn(self, s, a, r, s_):
        self.sess.run(self.train_op, feed_dict={self.S: s, self.a: a, self.R: r, self.S_: s_})
        if self.replacement['name'] == 'soft':
            self.sess.run(self.soft_replacement)
        else:
            if self.t_replace_counter % self.replacement['rep_iter_c'] == 0:
                self.sess.run(self.hard_replacement)
            self.t_replace_counter += 1


class Memory(object):
    def __init__(self, capacity, dims):
        self.capacity = capacity
        self.data = np.zeros((capacity, dims))
        self.pointer = 0

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.capacity  # replace the old memory with new memory
        self.data[index, :] = transition
        self.pointer += 1

    def sample(self, n):
        assert self.pointer >= self.capacity, 'Memory has not been fulfilled'
        indices = np.random.choice(self.capacity, size=n)
        return self.data[indices, :]


def run():
    MAX_EPISODES = 256
    MAX_EP_STEPS = 256
    logger_gap = 16
    LR_A = 0.001  # learning rate for actor
    LR_C = 0.001  # learning rate for critic
    GAMMA = 0.9  # reward discount
    REPLACEMENT = [
        dict(name='soft', tau=0.01),
        dict(name='hard', rep_iter_a=600, rep_iter_c=500)
    ][0]  # you can try different target replacement strategies
    MEMORY_CAPACITY = 8192
    BATCH_SIZE = 32

    # ENV_NAME = 'Pendulum-v0'
    # env = gym.make(ENV_NAME)
    # env = env.unwrapped
    # env.seed(1)
    #
    # state_dim = env.observation_space.shape[0]
    # action_dim = env.action_space.shape[0]
    # action_bound = env.action_space.high
    # action_bias = 0

    state_dim = 11  # 1 traffic_light + 2 (uav_pos, uav_high) + 8 crossroad_vehicle_number
    action_dim = 17  # motion1, power8, bandwidth8

    with tf.name_scope('S'):
        S = tf.placeholder(tf.float32, shape=[None, state_dim], name='s')
    with tf.name_scope('R'):
        R = tf.placeholder(tf.float32, [None, 1], name='r')
    with tf.name_scope('S_'):
        S_ = tf.placeholder(tf.float32, shape=[None, state_dim], name='s_')

    gpu_limit = 0.45
    gpu_id = 1
    tf_config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=gpu_limit))
    tf_config.gpu_options.allow_growth = True

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)  # choose GPU:0
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # ignore the TensorFlow log messages
    sess = tf.Session(config=tf_config)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # recover the TensorFlow log messages

    actor = Actor(S, S_, sess, action_dim, LR_A, REPLACEMENT)
    critic = Critic(S, R, S_, sess, state_dim, action_dim, LR_C, GAMMA, REPLACEMENT, actor.a, actor.a_)
    actor.add_grad_to_graph(critic.a_grads)

    sess.run(tf.global_variables_initializer())
    M = Memory(MEMORY_CAPACITY, dims=2 * state_dim + action_dim + 1)

    var = 1.0
    timer = time.time()

    cv2.namedWindow(cv2_name)
    loggers = list()
    for i in range(MAX_EPISODES):
        # s = env.reset()
        uav_pos, z_high = i % 9, (i % 8) / 7 * (z_max - z_min) + z_min
        motion, powers, bandwd = 0, 0, 0
        vehicles = rd_poisson(100, 8)
        s = np.concatenate(((0, uav_pos, z_high, ), vehicles), axis=0)
        # 1 traffic_light + 2 (uav_pos, uav_high) + 8 crossroad_vehicle_number

        uav_info = list()

        # if i % 32 == 0:
        #     var *= 4

        # if i == 128:
        #     var = 0.5

            # if lamb1357 change, the memory should be clear
            # lamb1357[:] = rd_uniform(0.3, 1.0, 4)
            # lamb1357[:] = np.array((1.0, 0.6, 0.4, 0.3)) / traffic_gap
            # print("| lamb1357", lamb1357)

        for j in range(MAX_EP_STEPS):
            traffic_switch = (j // traffic_gap) % 2
            a = actor.choose_action(s)
            a = np.clip(rd_normals(a, var), -1, +1)

            motion = a[0] * 10 + 6.0
            powers = a[1:9] + 1.01  # avoid overflow
            powers = powers * (p_max / np.sum(vehicles * powers))
            bandwd = a[9:17] + 1.01  # avoid overflow
            bandwd = bandwd * (b_max / np.sum(vehicles * bandwd))
            # motion = rd_uniform(-4, 16),   # DDPG placeholder
            # powers = rd_uniform(0, p_max, 8)
            # bandwd = rd_uniform(0, b_max, 8)  # bandwidth8

            uav_pos, z_high = state_transition(uav_pos, z_high, motion)
            r = reward(bandwd, powers, vehicles, uav_pos, z_high, traffic_switch)
            r = np.tanh((r - 11000) * 0.001)
            vehicles = draw_crossroad(vehicles, uav_pos, z_high, traffic_switch)
            s_ = np.concatenate(((traffic_switch, uav_pos, z_high), vehicles), axis=0)
            uav_info.append(r)

            # s_, r, done, info = env.step(a)
            # s_, r = env.step(a)[0:2]

            M.store_transition(s, a, r, s_)
            # M.store_transition(s, a, r / 10, s_)

            if M.pointer > MEMORY_CAPACITY:
                var *= 0.9999  # decay the action randomness
                b_M = M.sample(BATCH_SIZE)
                b_s = b_M[:, :state_dim]
                b_a = b_M[:, state_dim: state_dim + action_dim]
                b_r = b_M[:, -state_dim - 1: -state_dim]
                b_s_ = b_M[:, -state_dim:]

                critic.learn(b_s, b_a, b_r, b_s_)
                actor.learn(b_s)
            if j < logger_gap:
                # reward1, uav_position1, uav_z_high1,
                # rho_powers8, bandwidth8, state_vehicle_number8
                logger = np.concatenate((s, powers, bandwd), axis=0)
                loggers.append(logger)

            s = s_
        ave_reward = np.average(uav_info)
        # ave_reward = 0 if np.isnan(ave_reward) else ave_reward
        print("I %i Reward %.3f  Var %.2e" % (i, ave_reward, var),
              "   max_veh %3i  uav_pos_z %i %2i   pow %.2e  bdw %.2e" %
              (np.max(s), uav_pos, int(z_high), np.average(powers), np.average(bandwd)))
        # print("   n_i * p_i", s * powers)
        # print("   n_i * b_i", s * bandwd)
    print('Running time: ', time.time() - timer)

    # reward, traffic_switch, uav_position, uav_z_high, vehicle_number, rho_powers, bandwidth
    logger_path = 'V1_uav_info_r_t_pos_z_n_p_b.npy'
    np.save(logger_path, loggers)
    print("SAVE loggers to:", logger_path)


if __name__ == '__main__':
    # from uav_crossroad import run
    # from DDPG import run

    run()
