# #!/usr/bin/env python
# # coding: utf-8
# import copy
# import time

# import matplotlib.pyplot as plt
# import torch.multiprocessing as mp
# from sklearn.manifold import TSNE

# from utils import *
# from utils.metrics import evaluate
# from models import build_encoder
# from typing import Callable, Dict, Tuple, Union, List


# from servers.build import SERVER_REGISTRY

# @SERVER_REGISTRY.register()
# class Server():

#     def __init__(self, args):
#         self.args = args
#         return
    
#     def aggregate(self, local_weights, local_deltas, client_ids, model_dict, current_lr):
#         C = len(client_ids)
#         for param_key in local_weights:
#             local_weights[param_key] = sum(local_weights[param_key])/C
#         return local_weights
    

# @SERVER_REGISTRY.register()
# class ServerM(Server):    
    
#     def set_momentum(self, model):

#         global_delta = copy.deepcopy(model.state_dict())
#         for key in global_delta.keys():
#             global_delta[key] = torch.zeros_like(global_delta[key])

#         global_momentum = copy.deepcopy(model.state_dict())
#         for key in global_momentum.keys():
#             global_momentum[key] = torch.zeros_like(global_momentum[key])

#         self.global_delta = global_delta
#         self.global_momentum = global_momentum


#     @torch.no_grad()
#     def FedACG_lookahead(self, model):
#         sending_model_dict = copy.deepcopy(model.state_dict())
#         for key in self.global_momentum.keys():
#             sending_model_dict[key] += self.args.server.momentum * self.global_momentum[key]

#         model.load_state_dict(sending_model_dict)
#         return copy.deepcopy(model)
    

#     def aggregate(self, local_weights, local_deltas, client_ids, model_dict, current_lr):
#         C = len(client_ids)
#         for param_key in local_weights:
#             local_weights[param_key] = sum(local_weights[param_key])/C
#         if self.args.server.momentum>0:
#             # print("self.args.server.get('FedACG'): ",self.args.server.get('FedACG'))

#             if not self.args.server.get('FedACG'): 
#                 for param_key in local_weights:               
#                     local_weights[param_key] += self.args.server.momentum * self.global_momentum[param_key]
                    
#             for param_key in local_deltas:
#                 self.global_delta[param_key] = sum(local_deltas[param_key])/C
#                 self.global_momentum[param_key] = self.args.server.momentum * self.global_momentum[param_key] + self.global_delta[param_key]
            

#         return local_weights


# @SERVER_REGISTRY.register()
# class ServerAdam(Server):    
    
#     def set_momentum(self, model):

#         global_delta = copy.deepcopy(model.state_dict())
#         for key in global_delta.keys():
#             global_delta[key] = torch.zeros_like(global_delta[key])

#         global_momentum = copy.deepcopy(model.state_dict())
#         for key in global_momentum.keys():
#             global_momentum[key] = torch.zeros_like(global_momentum[key])

#         global_v = copy.deepcopy(model.state_dict())
#         for key in global_v.keys():
#             global_v[key] = torch.zeros_like(global_v[key]) + (self.args.server.tau * self.args.server.tau)

#         self.global_delta = global_delta
#         self.global_momentum = global_momentum
#         self.global_v = global_v

    
#     def aggregate(self, local_weights, local_deltas, client_ids, model_dict, current_lr):
#         C = len(client_ids)
#         server_lr = self.args.trainer.global_lr
        
#         for param_key in local_deltas:
#             self.global_delta[param_key] = sum(local_deltas[param_key])/C
#             self.global_momentum[param_key] = self.args.server.momentum * self.global_momentum[param_key] + (1-self.args.server.momentum) * self.global_delta[param_key]
#             self.global_v[param_key] = self.args.server.beta * self.global_v[param_key] + (1-self.args.server.beta) * (self.global_delta[param_key] * self.global_delta[param_key])

#         for param_key in model_dict.keys():
#             model_dict[param_key] += server_lr *  self.global_momentum[param_key] / ( (self.global_v[param_key]**0.5) + self.args.server.tau)
            
#         return model_dict

# !/usr/bin/env python
# coding: utf-8
import copy
import time
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
from sklearn.manifold import TSNE

from utils import *
from utils.metrics import evaluate
from models import build_encoder
from typing import Callable, Dict, Tuple, Union, List

from servers.build import SERVER_REGISTRY

@SERVER_REGISTRY.register()
class Server():
    def __init__(self, args):
        self.args = args
        return
    
    def aggregate(self, local_weights, local_deltas, client_ids, model_dict, current_lr):
        # Basic FedAvg aggregation using PyTorch averaging.
        avg_weights = {}
        # local_weights is a dict: keys are parameter names, values are lists of client tensors.
        for param_key in local_weights:
            # Stack client weights along a new dimension and take the mean.
            avg_weights[param_key] = torch.mean(torch.stack(local_weights[param_key], dim=0), dim=0)
        return avg_weights
    

@SERVER_REGISTRY.register()
class ServerM(Server):    
    def set_momentum(self, model):
        global_delta = copy.deepcopy(model.state_dict())
        for key in global_delta.keys():
            global_delta[key] = torch.zeros_like(global_delta[key])
        global_momentum = copy.deepcopy(model.state_dict())
        for key in global_momentum.keys():
            global_momentum[key] = torch.zeros_like(global_momentum[key])
        self.global_delta = global_delta
        self.global_momentum = global_momentum

    @torch.no_grad()
    def FedACG_lookahead(self, model):
        sending_model_dict = copy.deepcopy(model.state_dict())
        for key in self.global_momentum.keys():
            sending_model_dict[key] += self.args.server.momentum * self.global_momentum[key]
        model.load_state_dict(sending_model_dict)
        return copy.deepcopy(model)
    
    def aggregate(self, local_weights, local_deltas, client_ids, model_dict, current_lr):
        C = len(client_ids)
        # Basic averaging of client weights:
        avg_weights = {}
        for param_key in local_weights:
            avg_weights[param_key] = torch.mean(torch.stack(local_weights[param_key], dim=0), dim=0)
        
        if self.args.server.momentum > 0:
            if not self.args.server.get('FedACG'): 
                # Add momentum term to each parameter:
                for param_key in avg_weights:
                    avg_weights[param_key] += self.args.server.momentum * self.global_momentum[param_key]
            # Update global momentum using client deltas.
            for param_key in local_deltas:
                self.global_delta[param_key] = torch.mean(torch.stack(local_deltas[param_key], dim=0), dim=0)
                self.global_momentum[param_key] = self.args.server.momentum * self.global_momentum[param_key] + self.global_delta[param_key]
        
        return avg_weights


@SERVER_REGISTRY.register()
class ServerAdam(Server):    
    def set_momentum(self, model):
        global_delta = copy.deepcopy(model.state_dict())
        for key in global_delta.keys():
            global_delta[key] = torch.zeros_like(global_delta[key])
        global_momentum = copy.deepcopy(model.state_dict())
        for key in global_momentum.keys():
            global_momentum[key] = torch.zeros_like(global_momentum[key])
        global_v = copy.deepcopy(model.state_dict())
        for key in global_v.keys():
            global_v[key] = torch.zeros_like(global_v[key]) + (self.args.server.tau * self.args.server.tau)
        self.global_delta = global_delta
        self.global_momentum = global_momentum
        self.global_v = global_v

    def aggregate(self, local_weights, local_deltas, client_ids, model_dict, current_lr):
        C = len(client_ids)
        server_lr = self.args.trainer.global_lr
        
        for param_key in local_deltas:
            self.global_delta[param_key] = torch.mean(torch.stack(local_deltas[param_key], dim=0), dim=0)
            self.global_momentum[param_key] = (self.args.server.momentum * self.global_momentum[param_key] +
                                               (1 - self.args.server.momentum) * self.global_delta[param_key])
            self.global_v[param_key] = (self.args.server.beta * self.global_v[param_key] +
                                        (1 - self.args.server.beta) * (self.global_delta[param_key] ** 2))
        
        updated_model_dict = {}
        for param_key in model_dict.keys():
            updated_model_dict[param_key] = model_dict[param_key] + server_lr * self.global_momentum[param_key] / (torch.sqrt(self.global_v[param_key]) + self.args.server.tau)
            
        return updated_model_dict


