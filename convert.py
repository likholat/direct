import torch
import numpy as np
import torch.onnx

from direct.data.tests.test_transforms import *
from omegaconf import OmegaConf
from direct.environment import *

cfg_from_file = OmegaConf.load('/home/alikholat/projects/direct/projects/calgary_campinas/baseline_model/config.yaml')
models, models_config = load_models_into_environment_config(cfg_from_file)
forward_operator, backward_operator = build_operators(cfg_from_file['physics'])

model, _ = initialize_models_from_config(cfg_from_file, models, forward_operator, backward_operator, torch.device('cpu'))

inp_dict = torch.load('projects/calgary_campinas/baseline_model/model_80500.pt',
                       map_location=torch.device('cpu'))
state_dict = {key.replace('model.', ''): val for key, val in inp_dict['model'].items()}
model.load_state_dict(state_dict=state_dict)

batch, coil, height, width, complex = 3, 15, 100, 80, 2
input_image = create_input([batch, height, width, complex])
sensitivity_map = create_input([batch, coil, height, width, complex])
masked_kspace = create_input([batch, coil, height, width, complex])
sampling_mask = torch.from_numpy(np.random.binomial(size=(batch, 1, height, width, 1), n=1, p=0.5))
previous_state = create_input([3, 128, 100, 80, 256])
loglikelihood_scaling = torch.tensor([0.2])

np.save('ref/input_image.npy', input_image)
np.save('ref/sensitivity_map.npy', sensitivity_map)
np.save('ref/masked_kspace.npy', masked_kspace)
np.save('ref/sampling_mask.npy', sampling_mask)
np.save('ref/previous_state.npy', previous_state)
np.save('ref/loglikelihood_scaling.npy', loglikelihood_scaling)


origin_forward = model.forward
input_map = {'input_image':input_image, 'masked_kspace':masked_kspace, 'sampling_mask':sampling_mask, 
                          'sensitivity_map':sensitivity_map, 'loglikelihood_scaling':loglikelihood_scaling}
model.forward = lambda x: origin_forward(input_image, 
               masked_kspace=masked_kspace, 
               sampling_mask=sampling_mask, 
               sensitivity_map=sensitivity_map,
               loglikelihood_scaling=loglikelihood_scaling
              )
torch.onnx.export(model, input_map,
                         'model.onnx',
                          opset_version=11,
                          enable_onnx_checker=False
                )
