'''
    This Deep-Q-Learning module might not be suitable for 
    other general problems
'''
import tensorflow as tf
import numpy as np
from tensorflow.contrib import layers
from pprint import pprint


class Environment(object):
    # override me
    state_shape = (None, 1, 2)
    action_count = 5

    def sample_action(self):
        return np.random.randint(self.action_count)

    def Qnetwork(self, state):
        x = state
        x = layers.linear(state, 2048)
        x = layers.linear(x, self.action_count, scope='qvs')
    
    @property
    def state(self):
        pass # return current state in shape `state_shape`

    def done(self):
        pass # True of False
    
    def step(self, action): 
        pass # return reward

    def reset(self):
        pass

class Qnet(object):
    def __init__(self,
            state, action, action_count, expected_qv,
            network_structure,
            learning_rate,
            scope='dqn'):
        with tf.variable_scope(scope):
            action_one_hot = tf.one_hot(action, depth=action_count)

            qvalues = network_structure(state)
            qvalue = tf.reduce_sum(qvalues*action_one_hot, 1)

            loss = expected_qv-qvalue
            loss = tf.where(tf.abs(loss) < 1.0,
                            0.5 * tf.square(loss),
                            tf.abs(loss) - 0.5)
            loss = tf.reduce_mean(loss)

            optimizer = tf.train.AdamOptimizer(
                learning_rate=learning_rate)
            train_op = tf.contrib.slim.learning.create_train_op(
                    loss, optimizer)

        self.state = state
        self.action = action
        self.expected_qv = expected_qv

        self.qvalues = qvalues
        self.qvalue = qvalue
        self.train_op = train_op
        self.loss = loss
        self.max_qv = tf.reduce_max(qvalues, 1)
        self.max_qv_ind = tf.argmax(qvalues, 1)
        self.sample_qv_ind = tf.reshape(tf.multinomial(qvalues, 1), (-1,))

    def predict(self, state):
        sess = tf.get_default_session()
        qvalues = sess.run(
                self.qvalues,
                feed_dict={self.state: state})
        return qvalues 

    def predict_action(self, state):
        sess = tf.get_default_session()
        max_qv, max_qv_ind = sess.run(
                [self.max_qv, self.max_qv_ind],
                feed_dict={self.state: state})
        return max_qv, max_qv_ind

    def sample_action(self, state):
        sess = tf.get_default_session()
        sampled_ind = sess.run(
                self.sample_qv_ind,
                feed_dict={self.state: state})
        return sampled_ind 

    def train(self, state, action, expected_qv):
        sess = tf.get_default_session()

        feed_dict = {self.state: state,
                     self.action: action,
                     self.expected_qv: expected_qv}
        sess.run(self.train_op, feed_dict=feed_dict)


class Replay(object):
    def __init__(self):
        self._state = []
        self._action = []
        self._reward = []
        self._next_state = []
        self._done = []

    def record(self, state, action, reward, next_state, done):
        self._state.append(state)
        self._action.append(action)
        self._reward.append(reward)
        self._next_state.append(next_state)
        self._done.append(done)

    def __len__(self):
        return len(self.state)

    def resize(self, length):
        if length >= len(self):
            return 
        else:
            picked = np.random.choice(len(self), size=(length,), replace=False)
            self._state = self.state[picked].tolist()
            self._action = self.action[picked].tolist()
            self._reward = self.reward[picked].tolist()
            self._next_state = self.next_state[picked].tolist()
            self._done = self.done[picked].tolist()

    @property
    def state(self):
        return np.array(self._state)

    @property
    def action(self):
        return np.array(self._action)

    @property
    def reward(self):
        return np.array(self._reward)

    @property
    def next_state(self):
        return np.array(self._next_state)

    @property
    def done(self):
        return np.array(self._done)

    def sample(self, batch_size):
        sample_len = min(len(self), batch_size)
        batch_ind = np.random.choice(
                len(self), size=(sample_len,))
        return self.state[batch_ind],\
                self.action[batch_ind],\
                self.reward[batch_ind],\
                self.next_state[batch_ind],\
                self.done[batch_ind]


class DDQL(object):
    def __init__(self, env):
        self.env = env 
        self.built = False

    def build(self,
              batch_size=32,
              learning_rate=1e-4,
              clip_norm=1.,
              gamma=1.0):

        state = tf.placeholder(
                'float32', 
                self.env.state_shape,
                name='state')
        action = tf.placeholder(
                'int32',
                [None],
                name='action')

        # expected_qv
        expected_qv = tf.placeholder(
                'float32', [None],
                name='expected_qv')
        dqn1 = Qnet(state, action, self.env.action_count, expected_qv, self.env.Qnetwork, learning_rate, 'dqn1')
        dqn2 = Qnet(state, action, self.env.action_count, expected_qv, self.env.Qnetwork, learning_rate, 'dqn2')

        # inputs
        self.state = state
        self.action = action
        self.expected_qv = expected_qv
        self.batch_size = batch_size
        self.gamma = gamma
        self.dqns = [dqn1, dqn2]
        self.built = True

    def decide_double_qnet(self):
        ob_net_ind = np.random.randint(2)
        act_net_ind = 1-ob_net_ind
        return self.dqns[ob_net_ind], self.dqns[act_net_ind] 

    def learn_from_replays(self, replays):
        sess = tf.get_default_session()

        ob_net, act_net = self.decide_double_qnet()
        # sample and update
        batch_s, batch_a, batch_q, batch_ns, batch_done = replays.sample(self.batch_size)
        max_next_qv, _ = ob_net.predict_action(batch_ns)
        expected_qv = batch_done * batch_q + (1-batch_done) * (batch_q+self.gamma*max_next_qv)
        act_net.train(batch_s, batch_a, batch_q)

    def train(self,
              continued=False,
              epsilon=0.9, epsilon_decay=0.99, epsilon_min=0.1,
              iterations=1000,
              rng=np.random,
              max_replays =200):
        assert (self.built)

        env = self.env
        sess = tf.get_default_session()

        sess.run(tf.global_variables_initializer())
        train_writer = tf.summary.FileWriter('outputs/tensorboard', sess.graph)
        saver = tf.train.Saver()
        if continued:
            saver.restore(sess, 'outputs/session/sess.ckpt')

        replays = Replay()
        global_step = tf.get_collection(tf.GraphKeys.GLOBAL_STEP)[0]
        i = begin = global_step.eval()
        while i < begin+iterations:
#       for i in range(begin, begin+iterations):
            i += 1
            env.reset()
            episode_reward = 0
            ob_net, act_net = self.decide_double_qnet()

            print('=========[{}, {}]========'.format(i, epsilon))
            while True:
                cur_state = env.state

                if rng.rand() < epsilon:
                    action = env.sample_action() 
                else:
                    # _, action = act_net.predict_action([cur_state])
                    action = act_net.sample_action([cur_state])
                    action = action[0]
                reward = env.step(action)
                print(reward)
                next_state = env.state
                replays.record(cur_state, action, reward, next_state, env.done())

                episode_reward += reward

                # if hasattr(env, 'visualize'):
                #     env.visualize(cur_state)
                #     qv = act_net.predict([cur_state])[0]
                #     print(action)
                #     print(qv)
                #     print('episode_rewar for now:', episode_reward)
                pprint(dict(zip(env.action_names, act_net.predict([cur_state])[0])))
                print('action', env.action_names[action], '--> reward', reward)

                if env.done():
                    print(episode_reward)
                    break

                # learning
                self.learn_from_replays(replays)

            # if episode_reward > 100:
            #     import ipdb
            #     ipdb.set_trace()

            epsilon = max(epsilon*epsilon_decay, epsilon_min)

            if len(replays) > max_replays:
                replays.resize(int(max_replays*0.2))
                print('shuffle')

            summary = tf.Summary(value=[
                tf.Summary.Value(tag="R", simple_value=episode_reward),
            ])
            train_writer.add_summary(summary, i)

            if i % 10 == 0:
                saver.save(sess, 'outputs/session/sess.ckpt')

if __name__ == '__main__':
    pass
