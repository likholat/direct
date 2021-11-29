from openvino_extensions import get_extensions_path
from openvino.inference_engine import IECore

import torch
import torch.nn as nn

import subprocess
import sys
import os


class OpenVINOModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_image, masked_kspace, sampling_mask, sensitivity_map, loglikelihood_scaling=None):
        input_map = {
            "input_image": input_image,
            "masked_kspace": masked_kspace,
            "sampling_mask": sampling_mask,
            "sensitivity_map": sensitivity_map,
        }

        origin_forward = self.model.forward
        self.model.forward = lambda x: origin_forward(
            input_image, masked_kspace=masked_kspace, sampling_mask=sampling_mask, sensitivity_map=sensitivity_map
        )

        torch.onnx.export(
            self.model,
            [input_image, masked_kspace, sampling_mask, sensitivity_map],
            "model.onnx",
            opset_version=11,
            enable_onnx_checker=False,
            input_names=["input_image", "masked_kspace", "sampling_mask", "sensitivity_map"],
            output_names=["cell_outputs", "previous_state"],
        )

        dirname = os.path.dirname(__file__)
        mo_extension = os.path.join(dirname, "mo_extensions")

        subprocess.run(
            [
                sys.executable,
                "-m",
                "mo",
                "--input_model=model.onnx",
                "--extension",
                mo_extension,
            ]
        )

        ie = IECore()
        ie.add_extension(get_extensions_path(), "CPU")
        net = ie.read_network("model.xml", "model.bin")
        exec_net = ie.load_network(net, "CPU")

        return exec_net.infer(input_map)
