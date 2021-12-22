from openvino_extensions import get_extensions_path
from openvino.inference_engine import IECore

import torch
import torch.nn as nn
import io


class OpenVINOModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.exec_net = None

    def create_net(self, input_image, masked_kspace, sampling_mask, sensitivity_map):
        ie = IECore()
        ie.add_extension(get_extensions_path(), "CPU")

        buf = io.BytesIO()        
        torch.onnx.export(
            self.model,
            (input_image, masked_kspace, sampling_mask, sensitivity_map),
            buf,
            opset_version=11,
            enable_onnx_checker=False,
            input_names=["input_image", "masked_kspace", "sampling_mask", "sensitivity_map"],
            output_names=["cell_outputs", "previous_state"],
        )

        net = ie.read_network(buf.getvalue(), b"", init_from_buffer=True)
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
        out = ([torch.Tensor(res["cell_outputs"])], torch.Tensor(res["previous_state"]))

        return out
