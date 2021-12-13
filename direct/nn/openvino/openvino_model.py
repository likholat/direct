# from openvino_extensions import get_extensions_path
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
        self.exec_net = None

    def create_net(self, input_image, masked_kspace, sampling_mask, sensitivity_map):
        torch.onnx.export(
            self.model,
            (input_image, masked_kspace, sampling_mask, sensitivity_map),
            "model.onnx",
            opset_version=11,
            enable_onnx_checker=False,
            input_names=["input_image", "masked_kspace", "sampling_mask", "sensitivity_map"],
            output_names=["cell_outputs", "previous_state"],
        )

        ie = IECore()
        # ie.add_extension(get_extensions_path(), "CPU")
        ie.add_extension(
            "/home/alikholat/projects/openvino_pytorch_layers/user_ie_extensions/build/libuser_cpu_extension.so", "CPU")

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

        net = ie.read_network("model.xml", "model.bin")
        self.exec_net = ie.load_network(net, "CPU")

    def forward(self, input_image, masked_kspace, sampling_mask, sensitivity_map):
        input_map = {
            "input_image": input_image,
            "masked_kspace": masked_kspace,
            "sampling_mask": sampling_mask,
            "sensitivity_map": sensitivity_map,
        }

        if self.exec_net is None:
            self.create_net(input_image, masked_kspace, sampling_mask, sensitivity_map)

        res = self.exec_net.infer(input_map)

        # types = {}
        # counts = self.exec_net.requests[0].get_perf_counts()
        # for count in counts.values():
        #     t = count['layer_type']
        #     types[t] = types.get(t, 0) + count['real_time']

        # for k, v in types.items():
        #     print(k, v)

        out = ([torch.Tensor(res["cell_outputs"])], torch.Tensor(res["previous_state"]))

        return out
