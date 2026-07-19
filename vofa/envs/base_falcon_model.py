import torch 
import torch.nn as nn
import inspect
import logging

log = logging.getLogger(__name__)


class BaseModule(nn.Module):
    def __init__(self, actor_obs_num, module_config_dict):
        super(BaseModule, self).__init__()
        self.actor_obs_num = actor_obs_num
        self.module_config_dict = module_config_dict
        self.history_length = module_config_dict.get('history_length')

        self._calculate_input_dim()
        self._calculate_output_dim()
        self._build_network_layer(self.module_config_dict.layer_config)

    def _calculate_input_dim(self):
        # calculate input dimension based on the input specifications
        self.input_dim = self.actor_obs_num * self.history_length

    def _calculate_output_dim(self):
        output_dim = 0
        for each_output in self.module_config_dict['output_dim']:
            if isinstance(each_output, (int, float)):
                output_dim += each_output
            else:
                current_function_name = inspect.currentframe().f_code.co_name
                raise ValueError(f"{current_function_name} - Unknown output type: {each_output}")
        self.output_dim = output_dim

    def _build_network_layer(self, layer_config):
        if layer_config['type'] == 'MLP':
            self._build_mlp_layer(layer_config)
        else:
            raise NotImplementedError(f"Unsupported layer type: {layer_config['type']}")
        
    def _build_mlp_layer(self, layer_config):
        layers = []
        hidden_dims = layer_config['hidden_dims']
        output_dim = self.output_dim
        activation = getattr(nn, layer_config['activation'])()

        layers.append(nn.Linear(self.input_dim, hidden_dims[0]))
        layers.append(activation)

        dropout = layer_config.get("dropout_prob", 0)
        
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))

        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                layers.append(nn.Linear(hidden_dims[l], output_dim))
            else:
                layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                layers.append(activation)
                if dropout > 0:
                    layers.append(nn.Dropout(p=dropout))

        self.module = nn.Sequential(*layers)

    def forward(self, input):
        return self.module(input)
    
class FalconModel(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.device = device
        self.actors = {}
        actor_obs_num = config.actor_obs_num
        for key, module_config_dict in config.module_dict.items():
            key = key.replace("actor_", "")
            self.actors[key] = BaseModule(actor_obs_num, module_config_dict).to(device)
        self.load(config.checkpoint)
        print("Pretrained Falcon Model Loaded")
        
    def load(self, ckpt_path):
        if ckpt_path is not None:
            log.info(f"Loading checkpoint from {ckpt_path}")
            loaded_dict = torch.load(ckpt_path, map_location=torch.device(self.device))
            
            for key in self.actors.keys():
                # only load the actor_module weights
                actor_weights = {
                    k.replace('actor_module.', ''): v
                    for k, v in loaded_dict["actor_model_state_dict"][key].items()
                    if 'actor_module' in k
                }
                self.actors[key].load_state_dict(actor_weights)
            
            # ---- freeze params ----
            for actor in self.actors.values():
                for param in actor.parameters():
                    param.requires_grad = False
            
            return loaded_dict["infos"]
        else:
            raise Exception("Falcon model not loaded")