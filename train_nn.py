import numpy as np
from game_logic import *
import pygame
import random
import tensorflow as tf
import time
import os

tf.random.set_seed(80085)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Constants
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
CARD_WIDTH = 40
CARD_HEIGHT = 60
CARD_SPACING = 20
STACK_X_OFFSET = 10
STACK_Y_OFFSET = 30

pygame.init()
window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption('Solitaire Spider')

game = Game()


def display_table(game):
    txtsurf = font.render('Possible to unclose: ' + str(len(game.closed)), True, (0, 0, 0))
    window.blit(txtsurf, (5, 5))

    for i, stack in enumerate(game.stacks):
        x = STACK_X_OFFSET + i * (CARD_WIDTH + CARD_SPACING)
        for j, card in enumerate(stack.stack):
            y = STACK_Y_OFFSET + j * CARD_SPACING
            dp = j * 3
            dt = 1.07**(len(stack.stack)-j-1)
            if not card.hidden:
                col = (230//dt, 40//dt, 40//dt)
                pygame.draw.rect(window, col, (x+dp, y, CARD_WIDTH, CARD_HEIGHT))
                txtsurf = font.render(card.name, True, (0, 0, 0))
                window.blit(txtsurf, (x+dp+2, y))
            else:
                col = (30//dt, 150//dt, 20//dt)
                pygame.draw.rect(window, col, (x+dp, y, CARD_WIDTH, CARD_HEIGHT))


def desk_repr(game):
    res = np.zeros((10, len(RANKS)+1, CARD_EMB_LEN))

    for i, stack in enumerate(game.stacks):
        for j, card in enumerate(reversed(stack.stack)):
            res[i, j, :] = card.repr()
            if j+1 >= 14: break

    return res


"""while True:
    game.print()

    unc = input('Unclose: ')
    if unc.lower() == 'yes':
        status = game.unclose_one()
    else:
        fr = int(input('from: '))
        to = int(input('to: '))
        n = int(input('n: '))
        status = game.make_move(fr, to, n)

    print('take rows ', game.check_rows())

    print(status)"""


class MyModel(tf.keras.Model):
    def __init__(self, inp_shape, dense_units, rnn_units1, outp_len):
        super().__init__(self)
        self.conv = tf.keras.layers.Conv2D(8, 3, strides=1, padding='same', activation='linear')
        """self.gru1 = tf.keras.layers.GRU(rnn_units1,
                                        return_sequences=True,
                                        return_state=True,
                                        recurrent_initializer='glorot_uniform',
                                        stateful=True,
                                        name='gru_1')"""
        self.flatten = tf.keras.layers.Flatten(name='flatt')
        self.dense_inp = tf.keras.layers.Dense(dense_units, input_dim=inp_shape, name='input_dense', activation='linear')
        self.dense = tf.keras.layers.Dense(outp_len, name='output_dense', activation='linear')

    def call(self, inputs, training=False):
        x = inputs
        x = self.conv(x, training=training)
        x = self.flatten(x)
        #x, state1 = self.gru1(x, training=training)

        x = self.dense_inp(x, training=training)

        x = self.dense(x, training=training)

        return x

    #@tf.function
    def train(self, states, rewards, actions):
        disc_rewards = discount_rewards(rewards)

        for state, reward, action in zip(states, disc_rewards, actions):
            with tf.GradientTape() as tape:
                p = self(np.array([state]), training=True)[0]
                loss = self.loss(p, action) * reward

            grads = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

    """@tf.function
    def train_step(self, inputs):
        inputs, labels = inputs

        with tf.GradientTape() as tape:
            predictions = self(inputs, training=True)
            loss = self.loss(labels, predictions)

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.reset_states()

        return {'loss': loss}"""

def rand_ind(lgts, temperature=1.0):
    lgts = lgts / temperature
    ids = tf.random.categorical(tf.math.log([lgts]), num_samples=1)
    return ids[0, 0].numpy()

def discount_rewards(rewards):
    res = []
    rewards.reverse()
    sum_reward = 0
    for r in rewards:
        sum_reward = r + 0.98 * sum_reward
        res.append(sum_reward)
    res.reverse()
    return res


EPISODE_SIZE = 10
MAX_STEPS = 100
SAVE_PER = 5

desk_emb_len = CARD_EMB_LEN * 10 * (len(RANKS) + 1)

model = MyModel(
    inp_shape=(10, len(RANKS)+1, CARD_EMB_LEN),
    dense_units=256,
    rnn_units1=256,
    outp_len=10+10+len(RANKS)+1)

#model.build((1))

model.compile(
    optimizer='rmsprop',
    loss=tf.keras.losses.MeanSquaredError()
)

#model.summary()

#

i = 1
game_step = 0
gamed = 0
stacks_collected = 0
wined = 0
start_epoch_time = time.time()

rewards, states, actions, losses = [], [], [], []

tape = tf.GradientTape(persistent=True)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        font = pygame.font.SysFont("Arial", 16)

    # Move

    desk_state = desk_repr(game)
    states.append(desk_state)

    move_pred = model(inputs=np.array([desk_state]), training=True)[0]

    action = np.zeros_like(move_pred)

    frm_out = move_pred[:10]
    to_out = move_pred[10:20]
    n_take_out = move_pred[20:-1]
    is_open_closed_out = move_pred[-1]

    if np.random.rand() < is_open_closed_out:
        action[-1] = 1.0
        status = game.unclose_one()
    else:
        ti = 0
        while True:
            frm = rand_ind(frm_out)
            to = rand_ind(to_out)
            n_take = rand_ind(n_take_out)

            if game.is_can_move(frm, to, n_take) or ti > 10: break
            ti += 1

        status = game.make_move(frm, to, n_take)

        action[frm] = 1
        action[10 + to] = 1
        action[20 + n_take] = 1

    actions.append(action)

    # Calculating reward

    st_collected = game.check_rows()
    stacks_collected += st_collected

    rew = 0.0001
    #if status == 'ok move': rew += 0.5
    if status == 'same suit move': rew += 0.1
    rew += st_collected * 1
    rewards.append(rew)

    # Training

    if game_step >= MAX_STEPS or stacks_collected == 8:
        if stacks_collected == 8:
            print(f'WIN!!! vvv with {game_step} steps')
            wined += 1

        #model.train(states, rewards, actions)
        rewards = discount_rewards(rewards)
        for loss, reward in zip(losses, rewards):
            l = loss * reward
            grads = tape.gradient(l, self.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        print(f'i: {i}, rew: {sum(rewards)}, step: {game_step}, gamed: {gamed}, stks: {stacks_collected}, duration: {round(time.time()-start_epoch_time, 2)} s')

        rewards, states, actions, losses = [], [], [], []
        game = Game()
        stacks_collected = 0
        gamed += 1
        game_step = 0
        start_epoch_time = time.time()
        tape = tf.GradientTape(persistent=True)

        if gamed % SAVE_PER == 0:
            model.save('SSP01')
            print('model saved')

    #

    game_step += 1
    i += 1

    window.fill((230, 230, 230))
    display_table(game)

    pygame.display.flip()

pygame.quit()


