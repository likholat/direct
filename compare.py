from openvino.inference_engine import IECore
import numpy as np
import torch
import torch.onnx
import openvino.inference_engine as ie
print(ie.__version__)

from direct.data.tests.test_transforms import *
from omegaconf import OmegaConf
from direct.environment import *


input_image = np.load('ref/input_image.npy')
sensitivity_map = np.load('ref/sensitivity_map.npy')
masked_kspace = np.load('ref/masked_kspace.npy')
sampling_mask = np.load('ref/sampling_mask.npy')
loglikelihood_scaling = np.load('ref/loglikelihood_scaling.npy')
previous_state = np.load('ref/previous_state.npy')

inputs = {}
inputs['0'] = input_image
inputs['1'] = masked_kspace
inputs['sampling_mask'] = sampling_mask
inputs['data.1'] = sensitivity_map 
inputs['4'] = loglikelihood_scaling


ie = IECore()
ie.add_extension('/home/alikholat/projects/openvino_pytorch_layers/user_ie_extensions/build/libuser_cpu_extension.so', 'CPU')

net = ie.read_network('model.xml', 'model.bin')
exec_net = ie.load_network(net, 'CPU')

out = exec_net.infer(inputs)
out = list(out.values())
print(len(out))
print(np.shape(out[0]))
# print(np.shape(out[1]))

input_image = torch.tensor(input_image)
sensitivity_map = torch.tensor(sensitivity_map)
masked_kspace = torch.tensor(masked_kspace)
sampling_mask = torch.tensor(sampling_mask)
loglikelihood_scaling = torch.tensor(loglikelihood_scaling)

cfg_from_file = OmegaConf.load('/home/alikholat/projects/direct/projects/calgary_campinas/baseline_model/config.yaml')
models, models_config = load_models_into_environment_config(cfg_from_file)
forward_operator, backward_operator = build_operators(cfg_from_file['physics'])

model, _ = initialize_models_from_config(cfg_from_file, models, forward_operator, backward_operator, torch.device('cpu'))

inp_dict = torch.load('projects/calgary_campinas/baseline_model/model_80500.pt',
                       map_location=torch.device('cpu'))
state_dict = {key.replace('model.', ''): val for key, val in inp_dict['model'].items()}
model.load_state_dict(state_dict=state_dict)

ref = model(input_image, 
            masked_kspace=masked_kspace, 
            sampling_mask=sampling_mask, 
            sensitivity_map=sensitivity_map,
            loglikelihood_scaling=loglikelihood_scaling
           )
print(len(ref))
print(np.shape(ref[0]))
print(np.shape(ref[1]))

# ref = ref[1].detach().numpy()
# out = out[0]

ref = ref[0].detach().numpy()

print('Reference range: [{}, {}]'.format(np.min(ref), np.max(ref)))
print('Out range: [{}, {}]'.format(np.min(out), np.max(out)))

maxdiff = np.max(np.abs(ref - out))
print('Maximal difference:', maxdiff)

# print(torch_fft.shape)
# print(fft.shape)
# print(np.max(np.abs(torch_fft-fft)))
