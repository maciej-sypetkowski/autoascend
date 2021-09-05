import numpy as np


class RLModel:
    def __init__(self, observation_def, action_space, train=False, training_comm=(None, None)):
        # observation_def -- dict {name -> tuple (shape, dtype)}
        self.observation_def = observation_def
        self.action_space = action_space
        self.train = train
        if self.train:
            self.input_queue, self.output_queue = training_comm
            self.input_queue.put(self)

    def encode_observation(self, observation):
        assert sorted(observation.keys()) == sorted(self.observation_def.keys())
        ret = []
        for key in sorted(self.observation_def.keys()):
            val = observation[key]
            shape, dtype = self.observation_def[key]
            assert val.shape == shape, (val.shape, shape)
            ret.append(np.array(list(val.reshape(-1).astype(dtype).tobytes()), dtype=np.uint8))
        ret = np.concatenate(ret)
        return ret

    def zero_observation(self):
        ret = {}
        for key in sorted(self.observation_def.keys()):
            shape, dtype = self.observation_def[key]
            ret[key] = np.zeros(shape=shape, dtype=dtype)
        return ret

    def observation_size(self):
        return len(self.encode_observation(self.zero_observation()))

    def decode_observation(self, data):
        ret = {}
        for key in sorted(self.observation_def.keys()):
            shape, dtype = self.observation_def
            arr = np.zeros(shape=shape, dtype=dtype)
            s = len(arr.tobytes())
            ret[key] = np.frombuffer(bytes(data[:s]), dtype=np.dtype).reshape(shape)
            data = data[s:]
        assert len(data) == 0
        return ret

    def choose_action(self, observation, legal_actions):
        assert all(map(lambda action: action in self.action_space, legal_actions))
        assert len(legal_actions) > 0
        if self.train:
            self.input_queue.put((observation, [self.action_space.index(action) for action in legal_actions]))
            action_id = self.output_queue.get()
            if action_id is None:
                raise KeyboardInterrupt()
            assert self.action_space[action_id] in legal_actions
            return self.action_space[action_id]

        i = np.random.randint(0, len(legal_actions))
        return legal_actions[i]
