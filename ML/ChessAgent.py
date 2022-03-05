from datetime import datetime
import tensorflow as tf
import zarr
from keras.callbacks import ReduceLROnPlateau
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types

from ML.ChessAgentPolicy import ChessAgentPolicy
from ML.ChessEnv import ChessEnv

from ML.ModifiedPyDriver import ModifiedPyDriver
from ML.dqn_agent import DqnAgent
from zarr_gen import ZarrGen


class ChessAgent(DqnAgent):
    def __init__(self, time_step_spec: ts.TimeStep, action_spec: types.NestedArraySpec, replay_buffer_size=50_000):
        super().__init__(time_step_spec, action_spec, replay_buffer_size)

        # models
        self.supervised_model = self.build_model()

        #  filepaths
        self.log_dir = "/home/ricardo/PycharmProjects/pythonProject/logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        self.cp_file = "/home/ricardo/PycharmProjects/pythonProject/models/cp3.ckpt"

        # #  filepaths
        # self.log_dir = "/home/arjudoso/evo/pythonProject/logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        # self.cp_file = "/home/arjudoso/evo/pythonProject/models/cp2.ckpt"

    def build_model(self) -> tf.keras.Model:
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', input_shape=(8, 8, 13)),
            # tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
            tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding="same"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dense(4272)
        ])

        opt = tf.keras.optimizers.Adam(self.hyperparams["learning_rate"])
        loss = tf.keras.losses.MeanSquaredError()
        model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])
        return model

    def build_model_functional(self) -> tf.keras.Model:
        inputs = tf.keras.Input(shape=(8, 8, 91))
        x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(inputs)  # (8, 8, 128)
        x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)  # (8, 8, 256)
        block1_out = tf.keras.layers.MaxPooling2D((2, 2))(x)  # (4, 4, 256)

        x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(block1_out)  # (4, 4, 256)
        x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)  # (4, 4, 256)
        block2_out = tf.keras.layers.add([x, block1_out])

        x = tf.keras.layers.ReLU()(block2_out)
        x = tf.keras.layers.BatchNormalization()(x)  # (4, 4, 256)
        x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(block2_out)  # (4, 4, 256)
        x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)  # (4, 4, 256)
        block3_out = tf.keras.layers.add([x, block2_out])

        x = tf.keras.layers.ReLU()(block3_out)
        x = tf.keras.layers.BatchNormalization()(x)  # (4, 4, 256)
        x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu')(block3_out)  # (4, 4, 256)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)  # (512)

        x = tf.keras.layers.Flatten()(x)
        # x = tf.keras.layers.Dense(512, activation="relu")(x)
        # x = tf.keras.layers.Dense(1024, activation="relu")(x)
        outputs = tf.keras.layers.Dense(4272, activation="relu")(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        opt = tf.keras.optimizers.Adam(self.hyperparams["learning_rate"])
        loss = tf.keras.losses.MeanSquaredError()
        model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])
        # tf.keras.utils.plot_model(model, "mini_resnet.png", show_shapes=True)

        return model

    def test_play(self, env: ChessEnv, mode="eval"):
        transitions = []
        env.print = True
        t_step = env.reset()

        if mode == "supr":
            supr_policy = ChessAgentPolicy(self.time_step_spec, self.action_spec,
                                           self.supervised_model, env=env)
            driver = ModifiedPyDriver(env, supr_policy, observers=None,
                                      transition_observers=[transitions.append],
                                      max_steps=None, max_episodes=1)
        else:
            driver = ModifiedPyDriver.ModifiedPyDriver(env, self.eval_policy, observers=None,
                                                       transition_observers=[transitions.append],
                                                       max_steps=None, max_episodes=1)
        driver.run(t_step)
        print(transitions[-1][2].reward)
        self.current_time_step = env.reset()
        env.print = False
        return transitions

    def load_checkpoint(self):
        self.supervised_model.load_weights(self.cp_file)
        print("Load model from checkpoint")

    def supervised_train_zarr(self, x_train: zarr.Array, y_train: zarr.array,
                              x_test: zarr.array, y_test: zarr.Array,
                              eps=1_000_000, num_to_load=500_000, mini_batch_size=64,
                              load_checkpoint=False):
        if load_checkpoint:
            self.load_checkpoint()

        train_gen = ZarrGen(x_train, y_train, mini_batch_size,
                            num_to_load=num_to_load, repeat=True)
        test_gen = ZarrGen(x_test, y_test, mini_batch_size,
                           num_to_load=num_to_load, repeat=True)

        if self.supervised_model.loss.name == 'mean_squared_error':
            opt = tf.keras.optimizers.Adam(self.hyperparams["learning_rate"])
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            print("Recompile with categorical loss")
            self.supervised_model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.log_dir)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.cp_file,
                                                         save_weights_only=True,
                                                         verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5,
                                      patience=10, min_lr=0.00001)

        self.supervised_model.fit(x=train_gen, validation_data=test_gen, shuffle=False,
                                  epochs=eps, callbacks=[tensorboard_callback, cp_callback, reduce_lr],
                                  steps_per_epoch=100, validation_steps=1000, validation_freq=25)
