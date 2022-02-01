from openvino_extensions import get_extensions_path
from openvino.inference_engine import IECore

import torch
import torch.nn as nn
import io


class InstanceNorm2dFunc(torch.autograd.Function):
    @staticmethod
    def symbolic(g, cls, input):
        c_scale = g.op("Constant", value_t=cls.scale_one)
        c_bias = g.op("Constant", value_t=cls.bias_zero)
        return g.op("InstanceNormalization", input, c_scale, c_bias)

    @staticmethod
    def forward(self, cls, input):
        y = cls.origin_forward(input)
        return y


class InstanceNorm2dONNX(nn.InstanceNorm2d):
    """
    This is a support class which helps export network with InstanceNorm2d in ONNX format.
    """

    def __init__(self, num_features):
        super().__init__(num_features)
        self.origin_forward = super().forward
        self.scale_one = torch.ones(num_features)
        self.bias_zero = torch.zeros(num_features)

    def forward(self, input):
        y = InstanceNorm2dFunc.apply(self, input).clone()
        return y


def convert_layer(model):
    for name, l in model.named_children():
        layer_type = l.__class__.__name__
        if layer_type == "InstanceNorm2d":
            new_layer = InstanceNorm2dONNX(l.num_features)
            setattr(model, name, new_layer)
        else:
            convert_layer(l)


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
            output_names = ["output"]
            args = ["masked_kspace"]
            if self.model.image_initialization == "sense":
                args.append("sensitivity_map")
        if args:
            self.input = [kwargs[k] for k in args]
        else:
            raise ValueError(f"The model is not supported by OpenVINO: {self.model.__class__}")

        convert_layer(self.model)

        with torch.no_grad():
            buf = io.BytesIO()
            torch.onnx.export(
                self.model,
                tuple(self.input),
                buf,
                opset_version=12,
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
            out = torch.Tensor(res["output"])
        return out

    def forward(self, **kwargs):
        if self.exec_net is None:
            self.create_net(**kwargs)

        res = self.exec_net.infer(kwargs)
        return self.postprocess(res)
