# Imports:
import os
import pickle
import numpy as np
import game_mechanics as gm
import data_functions as df
from tensorflow.keras import Sequential
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Softmax
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Input
from tensorflow_addons.optimizers import AdamW
import tensorflow as tf
import keras.backend.tensorflow_backend as tfback
from keras import backend as K
from tensorflow.keras.models import load_model
from _collections import deque
import random as rnd


# Set seeds:
tf.random.set_seed(0)
np.random.seed(0)
rnd.seed(0)


# import time
# from tensorflow.keras.callbacks import TensorBoard
# NAME = f"TTT-{int(time.time())}"
# NAME = "TTT-test1"
# tensorboard = TensorBoard(log_dir=f'C:/Users/gig13/OneDrive/Pulpit/Programy/TTT_AI/logs/{NAME}')


# Class:
class AlphaTTT:
    def __init__(self, player, policy_name=None, value_name=None, learning_rate=0.001, weight_decay=0.001):

        self.symbol = player
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        if policy_name is None:
            self.policy_network, self.policy_optimizer, self.policy_loss = self.create_policy_net()
        else:
            self.policy_network, self.policy_optimizer, self.policy_loss = self.load_arch_model(policy_name, model_type='policy')
        if value_name is None:
            self.value_network, self.value_optimizer, self.value_loss = self.create_value_net()
        else:
            self.value_network, self.value_optimizer, self.value_loss = self.load_arch_model(value_name, model_type='value')

        self.short_term_memory = deque([gm.create_board(), gm.create_board(), gm.create_board(), gm.create_board(), gm.create_board()], maxlen=5)
        self.gradient_memory = []
        # Taught memory is (step, gradient) pairs of insight from better model
        self.taught_memory = []

    def load_arch_model(self, model_name, model_type):
        model = load_model('./models/' + model_name)
        optimizer = AdamW(learning_rate=self.learning_rate, weight_decay=self.weight_decay)
        if model_type == 'policy':
            loss = tf.keras.losses.categorical_crossentropy
        else:
            loss = tf.keras.losses.MSE
        return model, optimizer, loss

    # Trying the bottleneck architecture for now
    @staticmethod
    def identity_block(X, f, filters):

        F1, F2, F3 = filters

        X_shortcut = X

        # First part of main path
        X = Conv2D(filters=F1, kernel_size=(f, f), strides=(1, 1), padding='same', use_bias=True)(X)
        # X = BatchNormalization()(X)
        # X = Activation('relu')(X)
        X = LeakyReLU(alpha=0.2)(X)

        # Second part of main path
        X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', use_bias=True)(X)
        # X = BatchNormalization()(X)
        # X = Activation('relu')(X)
        X = LeakyReLU(alpha=0.2)(X)

        # Third part of main path
        X = Conv2D(filters=F3, kernel_size=(f, f), strides=(1, 1), padding='same', use_bias=True)(X)
        # X = BatchNormalization()(X)

        # Add shortcut to main path and apply RELU
        X = Add()([X, X_shortcut])
        # X = Activation('relu')(X)
        X = LeakyReLU(alpha=0.2)(X)

        return X

    @staticmethod
    def convolutional_block(X, f, filters):

        F1, F2, F3 = filters

        X_shortcut = X

        # First part of main path
        X = Conv2D(filters=F1, kernel_size=(f, f), strides=(1, 1), padding='same', use_bias=True)(X)
        # X = BatchNormalization()(X)
        # X = Activation('relu')(X)
        X = LeakyReLU(alpha=0.2)(X)

        # Second part of main path
        X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', use_bias=True)(X)
        # X = BatchNormalization()(X)
        # X = Activation('relu')(X)
        X = LeakyReLU(alpha=0.2)(X)

        # Third part of main path
        X = Conv2D(filters=F3, kernel_size=(f, f), strides=(1, 1), padding='same', use_bias=True)(X)
        # X = BatchNormalization()(X)

        # Makes shortcut the same dimensionality as last layer so you can add them
        X_shortcut = Conv2D(F3, (1, 1), strides=(1, 1), padding='same', use_bias=True)(X_shortcut)
        # X_shortcut = BatchNormalization()(X_shortcut)

        # Add shortcut to main path and apply RELU
        X = Add()([X, X_shortcut])
        # X = Activation('relu')(X)
        X = LeakyReLU(alpha=0.2)(X)

        return X

    def create_policy_net(self):

        # The policy network is supposed to learn to classify expert player moves (sets a good foundation for self-play)

        # X_input = Input((gm.board_len, gm.board_len, 11))
        X_input = Input((gm.board_len, gm.board_len, 3))

        # Small model:
        X = Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), padding='same', use_bias=True)(X_input)
        # X = BatchNormalization()(X)
        X = LeakyReLU(alpha=0.2)(X)

        X = self.convolutional_block(X, 5, [24, 48, 24])
        X = self.identity_block(X, 5, [24, 48, 24])

        # Flattening
        X = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1))(X)
        X = Flatten()(X)
        X = Softmax()(X)

        # Model:
        model = Model(inputs=X_input, outputs=X)
        optimizer = AdamW(learning_rate=self.learning_rate, weight_decay=self.weight_decay)
        loss = tf.keras.losses.categorical_crossentropy
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

        return model, optimizer, loss

    def create_value_net(self):

        X_input = Input((gm.board_len, gm.board_len, 3))

        # Small model:
        X = Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), padding='same', use_bias=True)(X_input)
        # X = BatchNormalization()(X)
        X = LeakyReLU(alpha=0.2)(X)

        X = self.convolutional_block(X, 5, [16, 32, 16])
        # X = self.identity_block(X, 5, [24, 48, 24])

        # Flattening
        X = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1))(X)
        X = LeakyReLU(alpha=0.2)(X)
        X = Flatten()(X)
        X = Dense(1, activation="tanh")(X)

        # Model:
        model = Model(inputs=X_input, outputs=X)

        optimizer = AdamW(learning_rate=self.learning_rate, weight_decay=self.weight_decay)
        loss = tf.keras.losses.MSE
        model.compile(loss=loss, optimizer=optimizer)

        return model, optimizer, loss

    def probe_memory(self, external_memory=None, swap=False):

        if external_memory is None:
            memory = self.short_term_memory
        else:
            memory = external_memory

        empty_x = deque([], maxlen=5)
        empty_o = deque([], maxlen=5)

        for i in range(5):
            if swap is False:
                state = memory[i].copy()
            else:
                state = memory[i].copy()
                gm.swap_symbols(state)

            split_state = df.split_state(state)

            # empty_x.appendleft(split_state[1])
            # empty_o.appendleft(split_state[0])

            empty_x.append(split_state[1])
            empty_o.append(split_state[0])

        np_empty_x = np.array(empty_x)
        np_empty_o = np.array(empty_o)

        if self.symbol == 'X':
            turn = np.ones((1, gm.board_len, gm.board_len))
        else:
            turn = np.zeros((1, gm.board_len, gm.board_len))

        observation = np.r_[np_empty_x, np_empty_o, turn]

        # Model specific reshaping:
        observation = observation.reshape((gm.board_len, gm.board_len, 11))

        return observation

    def policy_predict(self, obs, external_memory=None, swap=False):

        # obs = np.array([self.probe_memory(external_memory=external_memory, swap=swap)])
        obs = df.split_state(obs)
        obs = np.r_[obs, df.get_symbol_token(self.symbol)]
        obs = np.moveaxis(obs, 0, -1)
        obs = np.array([obs])

        # test = obs.reshape((11, 30, 30))
        # check this

        prediction = self.policy_network.predict(obs)[0]
        prediction = prediction.reshape((gm.board_len, gm.board_len))

        return prediction

    def policy_predict_action(self, obs, epsilon, external_memory=None, swap=False, custom_mask=None, first_move=False, model_playing=True):
        # Add randomness 100%
        heat_map = self.policy_predict(obs, external_memory=external_memory, swap=swap)
        mask = gm.generate_availability_mask(obs)
        heat_map *= mask

        seed = rnd.random()

        # Opponent should be playing his best, model should be trying new things
        if model_playing is False:
            seed = 1

        # Using this so X model makes a random move his first move
        if first_move is True:
            # Completely random starting move
            av_moves = gm.available_moves(obs)
            moves_update = []
            for move in av_moves:
                if (6 < move[0] < 24) and (6 < move[1] < 24):
                    moves_update.append(move)

            av_moves = moves_update
            best_move = rnd.choice(av_moves)
            was_random = True
            return best_move, was_random

        if custom_mask is not None:
            heat_map *= custom_mask

        if seed < epsilon:
            # # Choosing according to confidence:
            flat_map = heat_map.flatten()

            # Protection from sum(pvals) > 1.0
            # if sum(flat_map[:-1]) > 1.0:
            #     flat_map /= sum(flat_map[:-1])
            #     flat_map = np.round(flat_map, 4)
            #
            # ind = np.where(np.random.multinomial(1, pvals=flat_map))[0][0]
            #
            # best_move = np.unravel_index(ind, (gm.board_len, gm.board_len))

            max_ignore = 100
            # Select how many moves we want to ignore, as epsilon decreases we begin to consider less moves
            ignore_n = rnd.randint(1, int(max_ignore*epsilon))

            for i in range(ignore_n):
                flat_map[np.argmax(flat_map)] = 0

            best_move = np.unravel_index(np.argmax(flat_map), (gm.board_len, gm.board_len))

            # Completely random choice:
            # av_moves = gm.available_moves(obs)
            # best_move = rnd.choice(av_moves)

            was_random = True
        else:
            best_move = np.unravel_index(np.argmax(heat_map), (gm.board_len, gm.board_len))
            was_random = False

        return best_move, was_random

    def value_predict(self, obs):

        obs = df.split_state(obs)
        obs = df.reshape_board(obs)
        obs = np.array([obs])

        prediction = self.value_network.predict(obs)[0][0]

        return prediction

    def train_policy(self, game_count, game_start_nr=0, epochs=1, batch_size=100, game_sample_count=100, permutation_sample_count=10, variation_sample_count=4,
                     validation_start_nr=170, validation_game_count=10, error_load=False, model_name='default_name'):

        data_path = "D:/machine_learning_data/TTT/data_flat"

        # Hyperparameters:
        steps_per_save = 5

        # Create the index structure to sample from
        if error_load is False:
            game_collection = {}

            for i in range(game_count):
                game_nr = i + game_start_nr

                game_collection[i] = {}

                perm_count = len(os.listdir(f"{data_path}/game_{game_nr}"))
                for j in range(perm_count):
                    game_collection[i][j] = []
                    for k in range(16):
                        game_collection[i][j].append(k)
        else:
            file_to_read = open("game_collection1.pkl", "rb")
            game_collection = pickle.load(file_to_read)
            file_to_read.close()

        # Create validation data:
        validation_data = []
        validation_labels = []

        for i in range(validation_game_count):
            val_game_nr = i + validation_start_nr
            for k in range(16):
                val_set = np.load(f"{data_path}/game_{val_game_nr}/perm_0/var_{k}.npz")
                val_dat = val_set['data']
                val_lab = val_set['label']

                # val_lab = df.add_noise_to_labels(val_lab, label_noise_mean, label_noise_std)

                for dat, lab in zip(val_dat, val_lab):
                    validation_data.append(dat)
                    validation_labels.append(lab)

        validation_data = np.array(validation_data)
        validation_labels = np.array(validation_labels)

        training_data = None
        training_labels = None
        data_iterator = 0
        move_count = 0

        error_data = None

        if error_load is False:
            steps = 0
        else:
            error_data = np.load('optimizer_state1.npy')
            steps = int(error_data[1]) + 1

        while len(game_collection) > 0:
        # for i in range(epochs):

            # training_data = []
            # training_labels = []

            if data_iterator == 0:
                arbitrary_above_average_game_length = 50
                move_count = game_sample_count*permutation_sample_count*variation_sample_count*arbitrary_above_average_game_length
                training_data_shape = (move_count, gm.board_len, gm.board_len, 3)
                training_labels_shape = (move_count, gm.board_len*gm.board_len)

                training_data = np.zeros(training_data_shape)
                training_labels = np.zeros(training_labels_shape)
            else:
                data_iterator = 0

            game_samples = rnd.sample(game_collection.items(), min(game_sample_count, len(game_collection)))

            for gm_nr, permutations in game_samples:

                permutation_samples = rnd.sample(permutations.items(), min(permutation_sample_count, len(permutations)))

                for perm_nr, variations in permutation_samples:

                    variation_samples = rnd.sample(variations, min(variation_sample_count, len(variations)))

                    # Add the data instances here for fit at end of while loop
                    for var in variation_samples:
                        # Hard set:
                        # perm_nr = 0
                        # var = 0

                        load_set = np.load(f"{data_path}/game_{gm_nr}/perm_{perm_nr}/var_{var}.npz")
                        data = load_set['data']
                        label = load_set['label']

                        for data_inst, label_inst in zip(data, label):
                            if data_iterator < move_count:
                                training_data[data_iterator] = data_inst
                                training_labels[data_iterator] = label_inst
                            data_iterator += 1

                    for var in variation_samples:
                        # pass
                        game_collection[gm_nr][perm_nr].remove(var)

                    if len(game_collection[gm_nr][perm_nr]) == 0:
                        # pass
                        del game_collection[gm_nr][perm_nr]

                if len(game_collection[gm_nr]) == 0:
                    # pass
                    del game_collection[gm_nr]

            if data_iterator + 1 < move_count:
                functional_data_slice = training_data[:data_iterator]
                functional_labels_slice = training_labels[:data_iterator]
            else:
                functional_data_slice = training_data
                functional_labels_slice = training_labels

            # Train model:
            # training_data = np.array(training_data)
            # training_labels = np.array(training_labels)

            # Shuffle it up:
            p = np.random.permutation(len(functional_data_slice))
            functional_data_slice = functional_data_slice[p]
            functional_labels_slice = functional_labels_slice[p]

            if error_load is False:
                pass
            else:
                pass

            self.policy_network.fit(functional_data_slice, functional_labels_slice, initial_epoch=steps, epochs=steps+epochs, batch_size=batch_size,
                                    validation_data=(validation_data, validation_labels))# , callbacks=[tensorboard])

            if steps % steps_per_save == 0 and steps != 0:
                print("Saving...")
                save_ttt_model(self.policy_network, model_name)
                dict_file = open("game_collection.pkl", "wb")
                pickle.dump(game_collection, dict_file)
                dict_file.close()
            # Always save optimizer_state because steps always needs to be updated
            optimizer_state = np.array([self.policy_network.optimizer.learning_rate, steps])
            np.save("optimizer_state.npy", optimizer_state)

            # Divide learning_rate by decay every decay_steps
            steps += 1

    def evaluate_policy(self, game_nr_start, game_count, move_cluster=False):

        data_path = "D:/machine_learning_data/TTT/data_flat"

        evaluation_data = []
        evaluation_labels = []

        for i in range(game_count):
            # game_hists = df.pull_game_data(game_nr=game_nr_start + i, move_cluster=move_cluster)
            # data, labels = df.augment_game(game_hists)

            load_set = np.load(os.path.join(data_path, f"game_{game_nr_start + i}", "perm_0", "var_0.npz"))

            data = load_set["data"]
            labels = load_set["label"]

            for game, label in zip(data, labels):
                evaluation_data.append(game)
                evaluation_labels.append(label)

        evaluation_data = np.array(evaluation_data)
        evaluation_labels = np.array(evaluation_labels)

        self.policy_network.evaluate(evaluation_data, evaluation_labels)

        # predictions = self.policy_network(evaluation_data)
        # correct_list = []
        #
        # for i, pred in enumerate(predictions):
        #     if np.argmax(pred) == np.argmax(evaluation_labels[i]):
        #         correct_list.append(1)
        #     else:
        #         correct_list.append(0)
        #
        # print(correct_list)

    def compute_policy_gradients(self, obs, action, memorize=False):
        y_true = df.generate_y_data([action])
        y_true = convert_to_tensor_func(y_true)
        # obs = self.probe_memory()

        obs = df.get_prediction_format(obs, self.symbol)
        obs = convert_to_tensor_func(obs)

        grads, variables = get_gradients(self, obs, y_true)

        if memorize is False:
            return grads, variables
        else:
            self.gradient_memory.append(grads)

    def apply_win_to_gradients(self, discounted_win, teacher_discounts=None):
        for step, gradients in enumerate(self.gradient_memory):
            for var_index, layer in enumerate(gradients):
                self.gradient_memory[step][var_index] *= discounted_win[step]
                if teacher_discounts is not None:
                    self.taught_memory[step][var_index] *= teacher_discounts[step]
        return

    def clear_memory(self):
        self.gradient_memory = []
        self.taught_memory = []
        self.short_term_memory = deque([gm.create_board(), gm.create_board(), gm.create_board(), gm.create_board(), gm.create_board()], maxlen=5)
        return

    def apply_policy_gradients(self, step_batch, iterator):

        grad_count = iterator
        gradient_path = "D:/machine_learning_data/TTT/gradients/"

        step_iterator = 0

        while step_iterator < grad_count:

            first = True
            batch_gradient = None

            batch_size = min(step_batch, grad_count - step_iterator)

            for step_ind in range(batch_size):

                gradient_step = np.load(gradient_path + f'grad{step_iterator + step_ind}.npz', allow_pickle=True)['step_gradient']

                # Remove after pulling from memory
                os.remove(gradient_path + f'grad{step_iterator + step_ind}.npz')

                if first is True:
                    batch_gradient = gradient_step
                    first = False
                else:
                    batch_gradient += gradient_step

            batch_gradient /= step_batch
            step_iterator += batch_size

            # Return to tensorflow to apply this
            current_update = []
            for layer in batch_gradient:
                current_update.append(convert_to_tensor_func(layer))

            self.policy_optimizer.apply_gradients(zip(current_update, self.policy_network.trainable_variables))

    def train_value(self, game_count, game_start_nr=0, epochs=1, batch_size=100, game_sample_count=100,
                    permutation_sample_count=10, variation_sample_count=4,
                    validation_start_nr=170, validation_game_count=10, error_load=False, model_name='default_name'):

        data_path = "D:/machine_learning_data/TTT/data_flat"
        value_path = "D:/machine_learning_data/TTT/value_labels"
        value_labels = np.load(os.path.join(value_path, "v_labels.npy"))

        # Hyperparameters:
        steps_per_save = 5

        # Create the index structure to sample from
        if error_load is False:
            game_collection = {}

            for i in range(game_count):
                game_nr = i + game_start_nr

                game_collection[i] = {}

                perm_count = len(os.listdir(f"{data_path}/game_{game_nr}"))
                for j in range(perm_count):
                    game_collection[i][j] = []
                    for k in range(16):
                        game_collection[i][j].append(k)
        else:
            file_to_read = open("game_collection1.pkl", "rb")
            game_collection = pickle.load(file_to_read)
            file_to_read.close()

        # Create validation data:
        validation_data = []
        validation_labels = []

        for i in range(validation_game_count):
            val_game_nr = i + validation_start_nr
            for k in range(16):
                val_set = np.load(f"{data_path}/game_{val_game_nr}/perm_0/var_{k}.npz")
                val_dat = val_set['data']

                if k > 7:
                    swap_param = -1
                else:
                    swap_param = 1

                # val_lab = val_set['label']
                val_lab = value_labels[val_game_nr] * np.ones((len(val_dat), 1)) * swap_param

                # val_lab = df.add_noise_to_labels(val_lab, label_noise_mean, label_noise_std)

                for dat, lab in zip(val_dat, val_lab):
                    validation_data.append(dat)
                    validation_labels.append(lab)

        validation_data = np.array(validation_data)
        validation_labels = np.array(validation_labels)

        training_data = None
        training_labels = None
        data_iterator = 0
        move_count = 0

        error_data = None

        if error_load is False:
            steps = 0
        else:
            error_data = np.load('optimizer_state1.npy')
            steps = int(error_data[1]) + 1

        while len(game_collection) > 0:
            # for i in range(epochs):

            # training_data = []
            # training_labels = []

            if data_iterator == 0:
                arbitrary_above_average_game_length = 50
                move_count = game_sample_count * permutation_sample_count * variation_sample_count * arbitrary_above_average_game_length
                training_data_shape = (move_count, gm.board_len, gm.board_len, 3)
                training_labels_shape = (move_count, 1)

                training_data = np.zeros(training_data_shape)
                training_labels = np.zeros(training_labels_shape)
            else:
                data_iterator = 0

            game_samples = rnd.sample(game_collection.items(), min(game_sample_count, len(game_collection)))

            for gm_nr, permutations in game_samples:

                permutation_samples = rnd.sample(permutations.items(), min(permutation_sample_count, len(permutations)))

                for perm_nr, variations in permutation_samples:

                    variation_samples = rnd.sample(variations, min(variation_sample_count, len(variations)))

                    # Add the data instances here for fit at end of while loop
                    for var in variation_samples:

                        load_set = np.load(f"{data_path}/game_{gm_nr}/perm_{perm_nr}/var_{var}.npz")
                        data = load_set['data']

                        if var > 7:
                            swap_param = -1
                        else:
                            swap_param = 1

                        label = value_labels[gm_nr] * np.ones((len(data), 1)) * swap_param

                        for data_inst, label_inst in zip(data, label):
                            if data_iterator < move_count:
                                training_data[data_iterator] = data_inst
                                training_labels[data_iterator] = label_inst
                            data_iterator += 1

                    for var in variation_samples:
                        # pass
                        game_collection[gm_nr][perm_nr].remove(var)

                    if len(game_collection[gm_nr][perm_nr]) == 0:
                        # pass
                        del game_collection[gm_nr][perm_nr]

                if len(game_collection[gm_nr]) == 0:
                    # pass
                    del game_collection[gm_nr]

            if data_iterator + 1 < move_count:
                functional_data_slice = training_data[:data_iterator]
                functional_labels_slice = training_labels[:data_iterator]
            else:
                functional_data_slice = training_data
                functional_labels_slice = training_labels

            # Shuffle it up:
            p = np.random.permutation(len(functional_data_slice))
            functional_data_slice = functional_data_slice[p]
            functional_labels_slice = functional_labels_slice[p]

            if error_load is False:
                pass
            else:
                pass

            self.value_network.fit(functional_data_slice, functional_labels_slice, initial_epoch=steps,
                                   epochs=steps + epochs, batch_size=batch_size,
                                   validation_data=(validation_data, validation_labels))  # , callbacks=[tensorboard])

            if steps % steps_per_save == 0 and steps != 0:
                print("Saving...")
                save_ttt_model(self.value_network, model_name)
                # dict_file = open("game_collection.pkl", "wb")
                # pickle.dump(game_collection, dict_file)
                # dict_file.close()
            # Always save optimizer_state because steps always needs to be updated
            # optimizer_state = np.array([self.policy_network.optimizer.learning_rate, steps])
            # np.save("optimizer_state.npy", optimizer_state)

            # Divide learning_rate by decay every decay_steps
            steps += 1



# Functions:
def _get_available_gpus():
    """Get a list of available gpu devices (formatted as strings).

    # Returns
        A list of available GPU devices.
    """
    #global _LOCAL_DEVICES
    if tfback._LOCAL_DEVICES is None:
        devices = tf.config.list_logical_devices()
        tfback._LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]


tfback._get_available_gpus = _get_available_gpus


# Functions:
@tf.function(experimental_relax_shapes=True)
def gradient_mean(layer_var):
    return tf.reduce_mean(layer_var, axis=0)


@tf.function(experimental_relax_shapes=True)
def gradient_sum(layer_var):
    return tf.reduce_sum(layer_var, axis=0)


@tf.function(experimental_relax_shapes=True)
def convert_to_tensor_func(array):
    return tf.convert_to_tensor(array)


@tf.function(experimental_relax_shapes=True)
def comp_grad_reshape(tensor, shape):
    return tf.reshape(tensor, shape)


@tf.function(experimental_relax_shapes=True)
def get_gradients(agent, obs, y_true):
    with tf.GradientTape() as tape:
        # y_pred = comp_grad_reshape(agent.model(obs), (900,))
        y_pred = agent.policy_network(obs)
        # y_true = comp_grad_reshape(y_true, (900,))
        loss = agent.policy_loss(y_true, y_pred)

    variables = agent.policy_network.trainable_variables
    grads = tape.gradient(loss, variables)

    return grads, variables


def save_ttt_model(model, model_name):
    path = './models/' + model_name
    tf.keras.models.save_model(model, filepath=path, save_format='h5')
    return


def get_parameter_count(model):
    return model.count_params()


def lr_schedule(initial_learning_rate, decay, step, decay_step):
    return initial_learning_rate * decay ** (step/decay_step)
