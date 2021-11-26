# coding=utf-8
# Copyright (c) DIRECT Contributors

import pytest
import torch
import numpy as np

from direct.data.transforms import fft2, ifft2
from direct.nn.rim.rim import RIM
from direct.nn.openvino.openvino_model import OpenVINOModel


def create_input(shape):

    data = torch.rand(shape).float()

    return data


@pytest.mark.parametrize(
    "shape",
    [
        # [1, 1, 3, 3],
        # [1, 1, 4, 4],
        # [3, 3, 16, 16],
        [2, 5, 16, 32],
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

    ov_model = OpenVINOModel(model)
    ov_out = ov_model(img, kspace, mask, sens)

    ov_out = list(ov_out.values())[1]

    # print(out.shape)
    # print(ov_out.shape)
    # print('Reference range: [{}, {}]'.format(np.min(out.detach().numpy()), np.max(out.detach().numpy())))
    # print('Out range: [{}, {}]'.format(np.min(ov_out), np.max(ov_out)))
    # print(np.max(np.abs(out.detach().numpy() - ov_out)))
    assert np.max(np.abs(out.detach().numpy() - ov_out)) < 1e-5
    assert list(out.shape) == [shape[0]] + [2] + shape[2:]
