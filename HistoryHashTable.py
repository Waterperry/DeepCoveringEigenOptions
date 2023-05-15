from typing import Callable

from tensorflow import CriticalSection
from tensorflow.python.ops.lookup_ops import MutableHashTable
import tensorflow as tf


class HistoryHashTable:
    def __init__(self):
        # hashmap default value & empty/deleted key placeholders
        self.hashmap_def_val = tf.convert_to_tensor(tf.Variable(0, dtype=tf.int32))  # mutable and dense
        self.hashmap_emp_val = tf.constant('<-e->', dtype=tf.string)  # dense only
        self.hashmap_del_val = tf.constant('<-d->', dtype=tf.string)  # dense only
        self.hashtable: MutableHashTable = MutableHashTable(key_dtype=tf.string,
                                                            value_dtype=tf.int32,
                                                            default_value=self.hashmap_def_val,
                                                            checkpoint=False,
                                                            experimental_is_anonymous=True)
        self._rho_normalizer = tf.Variable(0)       # if we compute rho distribution on the fly it saves us a lot
                                                    # of time later on.
        # self.hashtable = tf.lookup.experimental.DenseHashTable(key_dtype=tf.string,
        #                                                        value_dtype=tf.int32,
        #                                                        default_value=self.hashmap_def_val,
        #                                                        empty_key=self.hashmap_emp_val,
        #                                                        deleted_key=self.hashmap_del_val)

    def __call__(self, trajectory):
        if trajectory is None:
            return tf.constant(0)

        self.insert(trajectory)
        self._rho_normalizer.assign_add(1)

    def lookup_wrapper(self, query):
        obs = tf.strings.reduce_join(tf.strings.as_string(query), separator=" ")
        return self.hashtable.lookup(obs)

    def get_rho(self):
        return self._rho_normalizer

    def rho_distribution_of(self, x) -> tf.Tensor:
        # tf.print("state = ", x)
        value = self.lookup_wrapper(x)
        normalizer = self._rho_normalizer
        # tf.print(f"value = {value}\t normalizer = {normalizer}")
        return tf.convert_to_tensor(value/normalizer)

    def insert(self, trajectory, is_full_trajectory=True):
        # take first element of the observation as it is stored in a 2d tensor.
        if is_full_trajectory:
            obs = tf.strings.reduce_join(tf.strings.as_string(trajectory.observation), separator=" ")
        else:
            obs = tf.strings.reduce_join(tf.strings.as_string(trajectory), separator=" ")

        # hashtable lookup result is an EagerTensor, so we take 0'th element to get back to tf.Variable form.
        if tf.executing_eagerly():
            if self.hashtable.lookup(obs).numpy() == self.hashmap_def_val:
                val = tf.Variable(1, dtype=tf.int32)
                # val = tf.reshape(tf.convert_to_tensor(val), (1,))
                # in the continuous environment, this is complaining about being passed a reshaped tensor
                # but in the discrete environment, this is complaining about being passed a regular variable!
                self.hashtable.insert(obs, val)
            else:
                new_val = self.hashtable.lookup(obs) + 1
                self.hashtable.remove(obs)
                self.hashtable.insert(obs, new_val)
        else:
            if self.hashtable.lookup(obs) == self.hashmap_def_val:
                val = tf.Variable(1, dtype=tf.int32)
                # val = tf.reshape(tf.convert_to_tensor(val), (1,))
                # in the continuous environment, this is complaining about being passed a reshaped tensor
                # but in the discrete environment, this is complaining about being passed a regular variable!
                self.hashtable.insert(obs, val)
            else:
                new_val = self.hashtable.lookup(obs).assign_add(1)
                self.hashtable.remove(obs)
                self.hashtable.insert(obs, new_val)

    @staticmethod
    def static_insert(table, default_value, trajectory):
        # this is a terrible way of doing this, but it's also probably the simplest...
        # at least we don't have to do it in eager mode...
        obs = tf.strings.reduce_join(tf.strings.as_string(trajectory.observation), separator=" ")
        if table.lookup(obs) == default_value:
            val = tf.Variable(1, dtype=tf.int32)
            # val = tf.reshape(tf.convert_to_tensor(val), (1,))
            # in the continuous environment, this is complaining about being passed a reshaped tensor
            # but in the discrete environment, this is complaining about being passed a regular variable!
            table.insert(obs, val)
        else:
            new_val = table.lookup(obs) + 1
            table.insert(obs, new_val)

        return tf.constant(0)

    def get_hashtable(self):
        return self.hashtable

    def export_state_hashmap(self, verbose=False):
        """
        Export the state hash-map into a list of states and counts.
        :return:
        """
        if verbose:
            print(f"Exporting hashtable with {self.hashtable.size()} values")

        keys, values = self.hashtable.export()

        keys = list(map(lambda x: x.numpy().decode('utf-8').split(' '), keys))
        values = values.numpy()

        kv_pairs = zip(keys, values)
        if not verbose:
            return kv_pairs

        for tup in kv_pairs:
            obs = tup[0]
            try:
                obs = list(map(float, obs))
            except ValueError:
                obs = list(map(float, obs[:-1]))
            if verbose:
                print(f"{tup[1]} counts of {obs}")

        return kv_pairs


class TranslatedHashTable(HistoryHashTable):
    def __init__(self, translation_function: Callable):
        super().__init__()
        self._tf = translation_function

    def __call__(self, trajectory):
        if trajectory is None:
            return
        self.insert(trajectory)
        self._rho_normalizer.assign_add(1)

    def insert(self, trajectory, is_full_trajectory=True):
        # take first element of the observation as it is stored in a 2d tensor.
        key = self._tf(trajectory.observation)
        super().insert(key, is_full_trajectory=False)
        del key

    def rho_distribution_of(self, x) -> tf.Tensor:
        # tf.print("state = ", x)
        value = self.lookup_wrapper(x)
        normalizer = self._rho_normalizer
        # tf.print(f"value = {value}\t normalizer = {normalizer}")
        return tf.convert_to_tensor(value/normalizer)

    def lookup_wrapper(self, query):
        transformed_query = self._tf(query)
        obs = tf.strings.reduce_join(tf.strings.as_string(transformed_query), separator=" ")
        return self.hashtable.lookup(obs)
