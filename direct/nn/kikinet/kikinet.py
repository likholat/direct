# coding=utf-8
# Copyright (c) DIRECT Contributors

from typing import Callable, Optional

import torch
import torch.nn as nn

import direct.data.transforms as T
from direct.nn.conv.conv import Conv2d
from direct.nn.didn.didn import DIDN
from direct.nn.mwcnn.mwcnn import MWCNN
from direct.nn.crossdomain.multicoil import MultiCoil
from direct.nn.unet.unet_2d import UnetModel2d, NormUnetModel2d


class KIKINet(nn.Module):
    """
    Based on KIKINet implementation as in "KIKI-net: cross-domain convolutional neural networks for
    reconstructing undersampled magnetic resonance images" by Taejoon Eo et all. Modified to work with
    multicoil kspace data.
    """

    def __init__(
        self,
        forward_operator: Callable,
        backward_operator: Callable,
        image_model_architecture: str = "MWCNN",
        kspace_model_architecture: str = "DIDN",
        num_iter: int = 2,
        normalize: bool = False,
        **kwargs,
    ):
        """

        :param forward_operator: Callable,
                Forward Operator.
        :param backward_operator: Callable,
                Backward Operator.
        :param image_model_architecture: str,
                Image model architecture. Currently only implemented for MWCNN and (NORM)UNET. Default: 'MWCNN'.
        :param kspace_model_architecture: str,
                Kspace model architecture. Currently only implemented for CONV and DIDN and (NORM)UNET. Default: 'DIDN'.
        :param num_iter: int,
                Number of unrolled iterations.
        :param normalize: bool,
                If true, input is normalised based on input scaling_factor.
        :param kwargs: str,
                Keyword arguments for model architectures.
        """
        super(KIKINet, self).__init__()

        if image_model_architecture == "MWCNN":
            image_model = MWCNN(
                input_channels=2,
                first_conv_hidden_channels=kwargs.get("image_mwcnn_hidden_channels", 32),
                num_scales=kwargs.get("image_mwcnn_num_scales", 4),
                bias=kwargs.get("image_mwcnn_bias", False),
                batchnorm=kwargs.get("image_mwcnn_batchnorm", False),
            )
        elif image_model_architecture in ["UNET", "NORMUNET"]:
            unet = UnetModel2d if image_model_architecture == "UNET" else NormUnetModel2d
            image_model = unet(
                in_channels=2,
                out_channels=2,
                num_filters=kwargs.get("image_unet_num_filters", 8),
                num_pool_layers=kwargs.get("image_unet_num_pool_layers", 4),
                dropout_probability=kwargs.get("image_unet_dropout_probability", 0.0),
            )
        else:
            raise NotImplementedError(
                f"XPDNet is currently implemented only with image_model_architecture == 'MWCNN', 'UNET' or 'NORMUNET."
                f"Got {image_model_architecture}."
            )

        if kspace_model_architecture == "CONV":
            kspace_model = Conv2d(
                in_channels=2,
                out_channels=2,
                hidden_channels=kwargs.get("kspace_conv_hidden_channels", 16),
                n_convs=kwargs.get("kspace_conv_n_convs", 4),
                batchnorm=kwargs.get("kspace_conv_batchnorm", False),
            )
        elif kspace_model_architecture == "DIDN":
            kspace_model = DIDN(
                in_channels=2,
                out_channels=2,
                hidden_channels=kwargs.get("kspace_didn_hidden_channels", 16),
                num_dubs=kwargs.get("kspace_didn_num_dubs", 6),
                num_convs_recon=kwargs.get("kspace_didn_num_convs_recon", 9),
            )
        elif kspace_model_architecture in ["UNET", "NORMUNET"]:
            unet = UnetModel2d if kspace_model_architecture == "UNET" else NormUnetModel2d
            kspace_model = unet(
                in_channels=2,
                out_channels=2,
                num_filters=kwargs.get("kspace_unet_num_filters", 8),
                num_pool_layers=kwargs.get("kspace_unet_num_pool_layers", 4),
                dropout_probability=kwargs.get("kspace_unet_dropout_probability", 0.0),
            )
        else:
            raise NotImplementedError(
                f"XPDNet is currently implemented for kspace_model_architecture == 'CONV', 'DIDN',"
                f" 'UNET' or 'NORMUNET'. Got kspace_model_architecture == {kspace_model_architecture}."
            )

        self._coil_dim = 1
        self._complex_dim = -1
        self._spatial_dims = (2, 3)

        self.image_model_list = nn.ModuleList([image_model] * num_iter)
        self.kspace_model_list = nn.ModuleList([MultiCoil(kspace_model, self._coil_dim)] * num_iter)

        self.forward_operator = forward_operator
        self.backward_operator = backward_operator
        self.num_iter = num_iter
        self.normalize = normalize

    def forward(
        self,
        masked_kspace: torch.Tensor,
        sampling_mask: torch.Tensor,
        sensitivity_map: torch.Tensor,
        scaling_factor: Optional[torch.Tensor] = None,
    ):

        kspace = masked_kspace.clone()
        if self.normalize and scaling_factor is not None:
            kspace = kspace / (scaling_factor ** 2).view(-1, 1, 1, 1, 1)

        for i in range(self.num_iter):
            kspace = self.kspace_model_list[i](kspace.permute(0, 1, 4, 2, 3)).permute(0, 1, 3, 4, 2)

            image = T.reduce_operator(
                self.backward_operator(
                    torch.where(
                        sampling_mask == 0,
                        torch.tensor([0.0], dtype=kspace.dtype).to(kspace.device),
                        kspace,
                    ),
                    self._spatial_dims,
                ),
                sensitivity_map,
                self._coil_dim,
            )

            image = self.image_model_list[i](image.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

            if i < self.num_iter - 1:
                kspace = forward = torch.where(
                    sampling_mask == 0,
                    torch.tensor([0.0], dtype=image.dtype).to(image.device),
                    self.forward_operator(
                        T.expand_operator(image, sensitivity_map, self._coil_dim), dim=self._spatial_dims
                    ),
                )

        if self.normalize and scaling_factor is not None:
            image = image * (scaling_factor ** 2).view(-1, 1, 1, 1)

        return image