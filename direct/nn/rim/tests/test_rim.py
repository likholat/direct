# coding=utf-8
# Copyright (c) DIRECT Contributors
from openvino_extensions import get_extensions_path
from openvino.inference_engine import IECore

import subprocess
import sys
import numpy as np
import os

import pytest
import torch

from direct.data.transforms import fft2, ifft2
from direct.nn.rim.rim import RIM


def create_input(shape):

    data = torch.rand(shape).float()

    return data


@pytest.mark.parametrize(
    "shape",
    [
        [3, 3, 16, 16],
        # [2, 5, 16, 32],
    ],
)
@pytest.mark.parametrize(
    "hidden_channels",
    [4],  # , 8
)
@pytest.mark.parametrize(
    "length",
    [3],
)
@pytest.mark.parametrize(
    "depth",
    [1],  # , 2
)
@pytest.mark.parametrize(
    "no_parameter_sharing",
    [True],  # , False
)
@pytest.mark.parametrize(
    "instance_norm",
    [True],  # , False
)
@pytest.mark.parametrize(
    "dense_connect",
    [True],  # , False
)
@pytest.mark.parametrize(
    "skip_connections",
    [True],  # , False
)
@pytest.mark.parametrize(
    "image_init",
    [
        "zero-filled",
        # "sense",
        # "input-kspace",
    ],
)
def test_rim(
    shape,
    hidden_channels,
    length,
    depth,
    no_parameter_sharing,
    instance_norm,
    dense_connect,
    skip_connections,
    image_init,
):
    model = RIM(
        fft2,
        ifft2,
        hidden_channels=hidden_channels,
        length=length,
        depth=depth,
        no_parameter_sharing=no_parameter_sharing,
        instance_norm=instance_norm,
        dense_connect=dense_connect,
        skip_connections=skip_connections,
        image_initialization=image_init,
    ).cpu()

    img = create_input([shape[0]] + shape[2:] + [2]).cpu()
    kspace = create_input(shape + [2]).cpu()
    sens = create_input(shape + [2]).cpu()
    mask = create_input([shape[0]] + [1] + shape[2:] + [1]).round().int().cpu()

    out = model(img, kspace, mask, sens)[0][-1]

    input_map = {"input_image": img, "masked_kspace": kspace, "sampling_mask": mask, "sensitivity_map": sens}

    origin_forward = model.forward
    model.forward = lambda x: origin_forward(img, masked_kspace=kspace, sampling_mask=mask, sensitivity_map=sens)

    torch.onnx.export(
        model,
        input_map,
        "model.onnx",
        opset_version=11,
        enable_onnx_checker=False,
        input_names=["input_image", "masked_kspace", "sampling_mask", "sensitivity_map"],
    )

    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = dir_path.split("nn")[0]

    subprocess.run(
        [
            sys.executable,
            "-m",
            "mo",
            "--input_model=model.onnx",
            "--extension=" + dir_path + "openvino/mo_extensions",
        ]
    )

    ie = IECore()
    ie.add_extension(get_extensions_path(), "CPU")
    net = ie.read_network("model.xml", "model.bin")
    exec_net = ie.load_network(net, "CPU")
    ov = exec_net.infer(input_map)
    ov = list(ov.values())[1]

    print(np.max(np.abs(out.detach().numpy() - ov)))

    assert np.max(np.abs(out.detach().numpy() - ov)) < 3

    assert list(out.shape) == [shape[0]] + [2] + shape[2:]
