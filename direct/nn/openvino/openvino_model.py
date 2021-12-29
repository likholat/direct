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
        self.input = None
        self.model_name = self.model.__class__.__name__

    def create_net(self, **kwargs):
        ie = IECore()
        ie.add_extension(get_extensions_path(), "CPU")

        if self.model_name == "RIM":
            args = ["input_image", "masked_kspace", "sampling_mask", "sensitivity_map"]
            output_names = ["cell_outputs", "previous_state"]
        elif self.model_name == "Unet2d":
            args = ["masked_kspace", "sensitivity_map"]
            output_names = ["output"]

        if args:
            self.input = [kwargs[k] for k in args]
        else:
            raise ValueError(f"The model is not supported by OpenVINO: {self.model.__class__}")

        buf = io.BytesIO()
        torch.onnx.export(
            self.model,
            tuple(self.input),
            buf,
            opset_version=11,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH,
            input_names=args,
            output_names=output_names,
        )

        net = ie.read_network(buf.getvalue(), b"", init_from_buffer=True)
        self.exec_net = ie.load_network(net, "CPU")

    def postprocess(self, res):
        if self.model_name == "RIM":
            out = ([torch.Tensor(res["cell_outputs"])], torch.Tensor(res["previous_state"]))
        elif self.model_name == "Unet2d":
            out = torch.Tensor(next(res.iter()))
        return out

    def forward(self, **kwargs):
        if self.exec_net is None:
            self.create_net(**kwargs)

        res = self.exec_net.infer(kwargs)
        return self.postprocess(res)
