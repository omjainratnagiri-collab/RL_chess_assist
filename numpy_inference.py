

import numpy as np

def _clip_relu(x):
                                                                                                            
    return np.minimum(np.maximum(x, 0.0), 1.0)


def _relu(x):
    return np.maximum(x, 0.0)


class NumpyValueHead:
    

    def __init__(self, model=None):
        self.ready = False
        if model is not None:
            self.sync(model)

    def sync(self, model):
        def w(layer):
            return layer.weight.detach().cpu().numpy().astype(np.float32)

        def b(layer):
            return layer.bias.detach().cpu().numpy().astype(np.float32)

        self.input_proj_w = w(model.input_proj)
        self.input_proj_b = b(model.input_proj)

        self.res_fc1_w = w(model.shared.fc1)
        self.res_fc1_b = b(model.shared.fc1)
        self.res_fc2_w = w(model.shared.fc2)
        self.res_fc2_b = b(model.shared.fc2)

                                                                             
        self.val_fc1_w = w(model.value_head[0])
        self.val_fc1_b = b(model.value_head[0])
        self.val_fc2_w = w(model.value_head[2])
        self.val_fc2_b = b(model.value_head[2])

        self.ready = True

    def evaluate(self, us_vec, them_vec):
        """us_vec/them_vec: raw accumulator sums (embedding_dim each,
        BEFORE activation -- same inputs forward_from_accumulators takes).
        Returns a python float, side-to-move-relative, same convention as
        the torch path."""
        if not self.ready:
            raise RuntimeError("NumpyValueHead.sync(model) must be called before evaluate()")

        us = _clip_relu(us_vec)
        them = _clip_relu(them_vec)
                                                                    
                                  
        x = np.empty(us.size + them.size, dtype=np.float32)
        x[:us.size] = us
        x[us.size:] = them

                                  
        x = _clip_relu(self.input_proj_w @ x + self.input_proj_b)

                                                   
        residual = x
        h = _clip_relu(self.res_fc1_w @ x + self.res_fc1_b)
        h = self.res_fc2_w @ h + self.res_fc2_b
        x = _clip_relu(h + residual)

                    
        h = _relu(self.val_fc1_w @ x + self.val_fc1_b)
        v = self.val_fc2_w @ h + self.val_fc2_b
        return float(np.tanh(v[0]))
