# coding=utf-8
# Copyright (c) DIRECT Contributors

# Code and comments can be shared with code of FastMRI under the same MIT license:
# https://github.com/facebookresearch/fastMRI/
# The code can have been adjusted to our needs.

from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
# import torch.fft
from packaging import version

from direct.data.bbox import crop_to_bbox
from direct.utils import ensure_list, is_power_of_two
from direct.utils.asserts import assert_complex, assert_same_shape

class FFT2(torch.autograd.Function):
        @staticmethod
        def symbolic(g, x, inverse):
            return g.op('FFT', x)

        @staticmethod
        def forward(self, x, inverse):
            return  torch.fft(x, 2)


def to_tensor(data: np.ndarray) -> torch.Tensor:
    """
    Convert numpy array to PyTorch tensor. Complex arrays will have real and imaginary parts on the last axis.

    Parameters
    ----------
    data : np.ndarray

    Returns
    -------
    torch.Tensor
    """
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)

    data = torch.from_numpy(data)

    return data


def verify_fft_dtype_possible(data: torch.Tensor, dims: Tuple[int, ...]) -> bool:
    """
    Fft and ifft can only be performed on GPU in float16 if the shapes are powers of 2.
    This function verifies if this is the case.

    Parameters
    ----------
    data : torch.Tensor
    dims : tuple

    Returns
    -------
    bool
    """
    is_complex64 = data.dtype == torch.complex64
    is_complex32_and_power_of_two = (data.dtype == torch.float32) and all(
        is_power_of_two(_) for _ in [data.size(idx) for idx in dims]
    )

    return is_complex64 or is_complex32_and_power_of_two


def view_as_complex(data):
    """
    Returns a view of input as a complex tensor.

    For an input tensor of size (N, ..., 2) where the last dimension of size 2 represents the real and imaginary
    components of complex numbers, this function returns a new complex tensor of size (N, ...).

    Parameters
    ----------
    data : torch.Tensor
        with torch.dtype torch.float64 and torch.float32

    """
    return torch.view_as_complex(data)


def view_as_real(data):
    """
    Returns a view of data as a real tensor.

    For an input complex tensor of size (N, ...) this function returns a new real tensor of size (N, ..., 2) where the
    last dimension of size 2 represents the real and imaginary components of complex numbers.

    Parameters
    ----------
    data : torch.Tensor
        with complex torch.dtype
    """

    return torch.view_as_real(data)

import os.path
def fft2(
    data: torch.Tensor,
    dim: Tuple[int, ...] = (1, 2),
    centered: bool = True,
    normalized: bool = True,
) -> torch.Tensor:
    """
    Apply centered two-dimensional Inverse Fast Fourier Transform. Can be performed in half precision when
    input shapes are powers of two.

    Version for PyTorch >= 1.7.0.

    Parameters
    ----------
    data : torch.Tensor
        Complex-valued input tensor. Should be of shape (*, 2) and dim is in *.
    dim : tuple, list or int
        Dimensions over which to compute. Should be positive. Negative indexing not supported
        Default is (1, 2), corresponding to ('height', 'width').
    centered : bool
        Whether to apply a centered fft (center of kspace is in the center versus in the corners).
        For FastMRI dataset this has to be true and for the Calgary-Campinas dataset false.
    normalized : bool
        Whether to normalize the ifft. For the FastMRI this has to be true and for the Calgary-Campinas dataset false.
    Returns
    -------
    torch.Tensor: the fft of the data.
    """
    # # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    # # if not os.path.isfile('ref/fft2_inp.npy'): 
    # #     np.save('ref/fft2_inp.npy', data)

    # if not all((_ >= 0 and isinstance(_, int)) for _ in dim):
    #     raise TypeError(
    #         f"Currently fft2 does not support negative indexing. "
    #         f"Dim should contain only positive integers. Got {dim}."
    #     )

    # assert_complex(data, complex_last=True)

    # data = view_as_complex(data)
    # if centered:
    #     data = ifftshift(data, dim=dim)
    # # Verify whether half precision and if fft is possible in this shape. Else do a typecast.
    # if verify_fft_dtype_possible(data, dim):
    #     data = torch.fft.fftn(
    #         data,
    #         dim=dim,
    #         norm="ortho" if normalized else None,
    #     )
    # else:
    #     raise ValueError("Currently half precision FFT is not supported.")

    # if centered:
    #     data = fftshift(data, dim=dim)

    # data = view_as_real(data)


    # data_complex = data[:,:,:,:,1]
    # data_complex = torch.fft.fftn(data_complex)
    # data[:,:,:,:,1] = data_complex

    # if not os.path.isfile('ref/fft2_res.npy'): 
    #     np.save('ref/fft2_res.npy', data)

    fft = FFT2()
    data = fft.apply(data, False)
    return data


def ifft2(
    data: torch.Tensor,
    dim: Tuple[int, ...] = (1, 2),
    centered: bool = True,
    normalized: bool = True,
) -> torch.Tensor:
    """
    Apply centered two-dimensional Inverse Fast Fourier Transform. Can be performed in half precision when
    input shapes are powers of two.

    Version for PyTorch >= 1.7.0.

    Parameters
    ----------
    data : torch.Tensor
        Complex-valued input tensor. Should be of shape (*, 2) and dim is in *.
    dim : tuple, list or int
        Dimensions over which to compute. Should be positive. Negative indexing not supported
        Default is (1, 2), corresponding to ('height', 'width').
    centered : bool
        Whether to apply a centered ifft (center of kspace is in the center versus in the corners).
        For FastMRI dataset this has to be true and for the Calgary-Campinas dataset false.
    normalized : bool
        Whether to normalize the ifft. For the FastMRI this has to be true and for the Calgary-Campinas dataset false.
    Returns
    -------
    torch.Tensor: the ifft of the data.
    """
    # print('111111111111111111111111111111111111111111111111')
    # # if not os.path.isfile('ref/ifft2_inp.npy'): 
    # #     np.save('ref/ifft2_inp.npy', data)

    # if not all((_ >= 0 and isinstance(_, int)) for _ in dim):
    #     raise TypeError(
    #         f"Currently ifft2 does not support negative indexing. "
    #         f"Dim should contain only positive integers. Got {dim}."
    #     )
    # assert_complex(data, complex_last=True)

    # data = view_as_complex(data)
    # if centered:
    #     data = ifftshift(data, dim=dim)
    # # Verify whether half precision and if fft is possible in this shape. Else do a typecast.
    # if verify_fft_dtype_possible(data, dim):
    #     data = torch.fft.ifftn(
    #         data,
    #         dim=dim,
    #         norm="ortho" if normalized else None,
    #     )
    # else:
    #     raise ValueError("Currently half precision FFT is not supported.")

    # if centered:
    #     data = fftshift(data, dim=dim)

    # data = view_as_real(data)

    # # if not os.path.isfile('ref/ifft2_res.npy'): 
    # #     np.save('ref/ifft2_res.npy', data)

    return data


def safe_divide(input_tensor: torch.Tensor, other_tensor: torch.Tensor) -> torch.Tensor:
    """
    Divide input_tensor and other_tensor safely, set the output to zero where the divisor b is zero.

    Parameters
    ----------
    input_tensor : torch.Tensor
    other_tensor : torch.Tensor

    Returns
    -------
    torch.Tensor: the division.

    """

    data = torch.where(
        other_tensor == 0,
        torch.tensor([0.0], dtype=input_tensor.dtype).to(input_tensor.device),
        input_tensor / other_tensor,
    )

    return data


def align_as(input_tensor: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    """
    Permutes the dimensions of the input tensor to match the dimension order in the
    other tensor, adding size-one dims for any additional dimensions. The resulting
    tensor is a view on the original tensor.

    Example:
    --------
    >>> coils, height, width, complex = 3, 4, 5, 2
    >>> x = torch.randn(coils, height, width, complex)
    >>> y = torch.randn(height, width)
    >>> align_as(y, x).shape
    torch.Size([1, 4, 5, 1])
    >>> batch, coils, height, width, complex = 1, 2, 4, 5, 2
    >>> x = torch.randn(coils, height, width, complex)
    >>> y = torch.randn(batch, height, width,)
    >>> align_as(y, x).shape
    torch.Size([1, 4, 5, 1])

    Parameters:
    -----------
    input:  torch.Tensor
    other:  torch.Tensor

    Returns:
    --------
    torch.Tensor
    """
    one_dim = 1
    if not (
        (set(input_tensor.shape) - {one_dim}).issubset(set(other.shape))
        and np.prod(other.shape) % np.prod(input_tensor.shape) == 0
    ):
        raise ValueError(
            f"Dimensions mismatch. Tensor of shape {input_tensor.shape} cannot be aligned as tensor of shape "
            f"{other.shape}. Dimensions {list(input_tensor.shape)} should be contained in {list(other.shape)}."
        )
    input_shape = list(input_tensor.shape)
    other_shape = torch.tensor(other.shape, dtype=int)
    out_shape = torch.ones(len(other.shape), dtype=int)
    # TODO(gy): Fix to ensure complex_last when [2,..., 2] or [..., N,..., N,...] in other.shape,
    #  "-input_shape.count(dim):" is a hack and might cause problems.
    for dim in np.sort(np.unique(input_tensor.shape)):
        ind = torch.where(other_shape == dim)[0][-input_shape.count(dim) :]
        out_shape[ind] = dim
    return input_tensor.reshape(tuple(out_shape))


def modulus(data: torch.Tensor) -> torch.Tensor:
    """
    Compute modulus of complex input data. Assumes there is a complex axis (of dimension 2) in the data.

    Parameters
    ----------
    data : torch.Tensor

    Returns
    -------
    torch.Tensor: modulus of data.
    """
    # TODO: fix to specify dim of complex axis or make it work with complex_last=True.

    assert_complex(data, complex_last=False)
    complex_axis = -1 if data.size(-1) == 2 else 1

    return (data ** 2).sum(complex_axis).sqrt()  # noqa
    # return torch.view_as_complex(data).abs()


def modulus_if_complex(data: torch.Tensor) -> torch.Tensor:
    """
    Compute modulus if complex-valued.

    Parameters
    ----------
    data : torch.Tensor

    Returns
    -------
    torch.Tensor
    """
    # TODO: This can be merged with modulus if the tensor is real.
    try:
        return modulus(data)
    except ValueError:
        return data


def roll(
    data: torch.Tensor,
    shift: Union[int, Union[Tuple[int, ...], List[int]]],
    dims: Union[int, Union[Tuple, List]],
) -> torch.Tensor:
    """
    Similar to numpy roll but applies to pytorch tensors.
    Parameters
    ----------
    data : torch.Tensor
    shift: tuple, int
    dims : tuple, list or int

    Returns
    -------
    torch.Tensor
    """
    if isinstance(shift, (tuple, list)) and isinstance(dims, (tuple, list)):
        if len(shift) != len(dims):
            raise ValueError(f"Length of shifts and dimensions should be equal. Got {len(shift)} and {len(dims)}.")
        for curr_shift, curr_dim in zip(shift, dims):
            data = roll(data, curr_shift, curr_dim)
        return data
    dim_index = dims
    shift = shift % data.size(dims)

    if shift == 0:
        return data
    left_part = data.narrow(dim_index, 0, data.size(dims) - shift)
    right_part = data.narrow(dim_index, data.size(dims) - shift, shift)
    return torch.cat([right_part, left_part], dim=dim_index)


def fftshift(data: torch.Tensor, dim: Tuple[int, ...] = None) -> torch.Tensor:
    """
    Similar to numpy fftshift but applies to pytorch tensors.

    Parameters
    ----------
    data : torch.Tensor
    dim : tuple, list or int

    Returns
    -------
    torch.Tensor

    """
    if dim is None:
        dim = tuple(range(data.dim()))

    if isinstance(dim, int):
        dim = [dim]

    shift = [data.size(curr_dim) // 2 for curr_dim in dim]
    return roll(data, shift, dim)


def ifftshift(data: torch.Tensor, dim: Tuple[Union[str, int], ...] = None) -> torch.Tensor:
    """
    Similar to numpy ifftshift but applies to pytorch tensors.

    Parameters
    ----------
    data : torch.Tensor
    dim : tuple, list or int

    Returns
    -------
    torch.Tensor
    """
    if dim is None:
        dim = tuple(range(data.dim()))
        shift = [(dim + 1) // 2 for dim in data.shape]
    elif isinstance(dim, int):
        shift = (data.shape[dim] + 1) // 2
    else:
        shift = [(data.size(curr_dim) + 1) // 2 for curr_dim in dim]
    return roll(data, shift, dim)


def complex_multiplication(input_tensor: torch.Tensor, other_tensor: torch.Tensor) -> torch.Tensor:
    """
    Multiplies two complex-valued tensors. Assumes input tensors are complex (last axis has dimension 2).

    Parameters
    ----------
    input_tensor : torch.Tensor
        Input data
    other_tensor : torch.Tensor
        Input data

    Returns
    -------
    torch.Tensor
    """
    assert_complex(input_tensor, complex_last=True)
    assert_complex(other_tensor, complex_last=True)
    # multiplication = torch.view_as_complex(x) * torch.view_as_complex(y)
    # return torch.view_as_real(multiplication)

    complex_index = -1

    real_part = input_tensor[..., 0] * other_tensor[..., 0] - input_tensor[..., 1] * other_tensor[..., 1]
    imaginary_part = input_tensor[..., 0] * other_tensor[..., 1] + input_tensor[..., 1] * other_tensor[..., 0]

    multiplication = torch.cat(
        [
            real_part.unsqueeze(dim=complex_index),
            imaginary_part.unsqueeze(dim=complex_index),
        ],
        dim=complex_index,
    )

    return multiplication


def _complex_matrix_multiplication(input_tensor, other_tensor, mult_func):
    """
    Perform a matrix multiplication, helper function for complex_bmm and complex_mm.

    Parameters
    ----------
    x : torch.Tensor
    other_tensor : torch.Tensor
    mult_func : Callable
        Multiplication function e.g. torch.bmm or torch.mm

    Returns
    -------
    torch.Tensor
    """
    if not input_tensor.is_complex() or not other_tensor.is_complex():
        raise ValueError("Both input_tensor and other_tensor have to be complex-valued torch tensors.")

    output = (
        mult_func(input_tensor.real, other_tensor.real)
        - mult_func(input_tensor.imag, other_tensor.imag)
        + 1j * mult_func(input_tensor.real, other_tensor.imag)
        + 1j * mult_func(input_tensor.imag, other_tensor.real)
    )
    return output


def complex_mm(input_tensor, other_tensor):
    """
    Performs a matrix multiplication of the 2D complex matrices input_tensor and other_tensor.
    If input_tensor is a (n×m) tensor, other_tensor is a (m×p) tensor, out will be a (n×p) tensor.

    Parameters
    ----------
    input_tensor : torch.Tensor
    other_tensor : torch.Tensor
    Returns
    -------
    torch.Tensor
    """
    return _complex_matrix_multiplication(input_tensor, other_tensor, torch.mm)


def complex_bmm(input_tensor, other_tensor):
    """
    Complex batch multiplication.

    Parameters
    ----------
    input_tensor : torch.Tensor
    other_tensor : torch.Tensor
    Returns
    -------
    torch.Tensor
    """
    return _complex_matrix_multiplication(input_tensor, other_tensor, torch.bmm)


def conjugate(data: torch.Tensor) -> torch.Tensor:
    """
    Compute the complex conjugate of a torch tensor where the last axis denotes the real and complex part (last axis
    has dimension 2).

    Parameters
    ----------
    data : torch.Tensor

    Returns
    -------
    torch.Tensor
    """
    # assert_complex(data, complex_last=True)
    # data = torch.view_as_real(
    #     torch.view_as_complex(data).conj()
    # )
    assert_complex(data, complex_last=True)
    data = data.clone()  # Clone is required as the data in the next line is changed in-place.
    data[..., 1] = data[..., 1] * -1.0

    return data


def apply_mask(
    kspace: torch.Tensor,
    mask_func: Union[Callable, torch.Tensor],
    seed: Optional[int] = None,
    return_mask: bool = True,
) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Subsample kspace by setting kspace to zero as given by a binary mask.

    Parameters
    ----------
    kspace : torch.Tensor
        k-space as a complex-valued tensor.
    mask_func : callable or torch.tensor
        Masking function, taking a shape and returning a mask with this shape or can be broadcast as such
        Can also be a sampling mask.
    seed : int
        Seed for the random number generator
    return_mask : bool
        If true, mask will be returned

    Returns
    -------
    masked data (torch.Tensor), mask (torch.Tensor)
    """
    # TODO: Split the function to apply_mask_func and apply_mask

    assert_complex(kspace, complex_last=True)

    if not isinstance(mask_func, torch.Tensor):
        shape = np.array(kspace.shape)[1:]  # The first dimension is always the coil dimension.
        mask = mask_func(shape, seed)
    else:
        mask = mask_func

    masked_kspace = torch.where(mask == 0, torch.tensor([0.0], dtype=kspace.dtype), kspace)

    if not return_mask:
        return masked_kspace

    return masked_kspace, mask


def tensor_to_complex_numpy(data: torch.Tensor) -> np.ndarray:
    """
    Converts a complex pytorch tensor to a complex numpy array.
    The last axis denote the real and imaginary parts respectively.

    Parameters
    ----------
    data : torch.Tensor
        Input data

    Returns
    -------
    Complex valued np.ndarray
    """
    assert_complex(data)
    data = data.detach().cpu().numpy()
    return data[..., 0] + 1j * data[..., 1]


def root_sum_of_squares(data: torch.Tensor, dim: int = 0) -> torch.Tensor:
    r"""
    Compute the root sum of squares (RSS) transform along a given dimension of the input tensor.

    $$x_{\textrm{rss}} = \sqrt{\sum_{i \in \textrm{coil}} |x_i|^2}$$

    Parameters
    ----------
    data : torch.Tensor
        Input tensor

    dim : int
        Coil dimension. Default is 0 as the first dimension is always the coil dimension.

    Returns
    -------
    torch.Tensor : RSS of the input tensor.
    """
    try:
        assert_complex(data, complex_last=True)
        complex_index = -1
        return torch.sqrt((data ** 2).sum(complex_index).sum(dim))
    except ValueError:
        return torch.sqrt((data ** 2).sum(dim))


def center_crop(data: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """
    Apply a center crop along the last two dimensions.

    Parameters
    ----------
    data : torch.Tensor
    shape : Tuple[int, int]
        The output shape, should be smaller than the corresponding data dimensions.

    Returns
    -------
    torch.Tensor : The center cropped data.
    """
    # TODO: Make dimension independent.
    if not (0 < shape[0] <= data.shape[-2]) or not (0 < shape[1] <= data.shape[-1]):
        raise ValueError(f"Crop shape should be smaller than data. Requested {shape}, got {data.shape}.")

    width_lower = (data.shape[-2] - shape[0]) // 2
    width_upper = width_lower + shape[0]
    height_lower = (data.shape[-1] - shape[1]) // 2
    height_upper = height_lower + shape[1]

    return data[..., width_lower:width_upper, height_lower:height_upper]


def complex_center_crop(data_list, shape, offset=1, contiguous=False):
    """
    Apply a center crop to the input data, or to a list of complex images

    Parameters
    ----------
    data_list : List[torch.Tensor] or torch.Tensor
        The complex input tensor to be center cropped. It should have at least 3 dimensions
         and the cropping is applied along dimensions didx and didx+1 and the last dimensions should have a size of 2.
    shape : Tuple[int, int]
        The output shape. The shape should be smaller than the corresponding dimensions of data.
        If one value is None, this is filled in by the image shape.
    offset : int
        Starting dimension for cropping.
    contiguous : bool
        Return as a contiguous array. Useful for fast reshaping or viewing.

    Returns
    -------
    torch.Tensor or list[torch.Tensor]: The center cropped input_image
    """
    data_list = ensure_list(data_list)
    assert_same_shape(data_list)

    image_shape = list(data_list[0].shape)
    ndim = data_list[0].ndim
    bbox = [0] * ndim + image_shape

    # Allow for False in crop directions
    shape = [_ if _ else image_shape[idx + offset] for idx, _ in enumerate(shape)]
    for idx, _ in enumerate(shape):
        bbox[idx + offset] = (image_shape[idx + offset] - shape[idx]) // 2
        bbox[len(image_shape) + idx + offset] = shape[idx]

    if not all(_ >= 0 for _ in bbox[:ndim]):
        raise ValueError(
            f"Bounding box requested has negative values, "
            f"this is likely to data size being smaller than the crop size. Got {bbox} with image_shape {image_shape} "
            f"and requested shape {shape}."
        )

    output = [crop_to_bbox(data, bbox) for data in data_list]

    if contiguous:
        output = [_.contiguous() for _ in output]

    if len(output) == 1:  # Only one element:
        output = output[0]
    return output


def complex_random_crop(
    data_list,
    crop_shape,
    offset: int = 1,
    contiguous: bool = False,
    sampler: str = "uniform",
    sigma: bool = None,
):
    """
    Apply a random crop to the input data tensor or a list of complex.

    Parameters
    ----------
    data_list : Union[List[torch.Tensor], torch.Tensor]
        The complex input tensor to be center cropped. It should have at least 3 dimensions and the cropping is applied
        along dimensions -3 and -2 and the last dimensions should have a size of 2.
    crop_shape : Tuple[int, ...]
        The output shape. The shape should be smaller than the corresponding dimensions of data.
    offset : int
        Starting dimension for cropping.
    contiguous : bool
        Return as a contiguous array. Useful for fast reshaping or viewing.
    sampler : str
        Select the random indices from either a `uniform` or `gaussian` distribution (around the center)
    sigma : float or list of float
        Standard variance of the gaussian when sampler is `gaussian`. If not set will take 1/3th of image shape

    Returns
    -------
    torch.Tensor: The center cropped input tensor or list of tensors

    """
    if sampler == "uniform" and sigma is not None:
        raise ValueError(f"sampler `uniform` is incompatible with sigma {sigma}, has to be None.")

    data_list = ensure_list(data_list)
    assert_same_shape(data_list)

    image_shape = list(data_list[0].shape)

    ndim = data_list[0].ndim
    bbox = [0] * ndim + image_shape

    crop_shape = [_ if _ else image_shape[idx + offset] for idx, _ in enumerate(crop_shape)]
    crop_shape = np.asarray(crop_shape)

    limits = np.zeros(len(crop_shape), dtype=int)
    for idx, _ in enumerate(limits):
        limits[idx] = image_shape[offset + idx] - crop_shape[idx]

    if not all(_ >= 0 for _ in limits):
        raise ValueError(
            f"Bounding box limits have negative values, "
            f"this is likely to data size being smaller than the crop size. Got {limits}"
        )

    if sampler == "uniform":
        lower_point = np.random.randint(0, limits + 1).tolist()
    elif sampler == "gaussian":
        data_shape = np.asarray(image_shape[offset : offset + len(crop_shape)])
        if not sigma:
            sigma = data_shape / 6  # w, h
        if len(sigma) != 1 and len(sigma) != len(crop_shape):  # type: ignore
            raise ValueError(f"Either one sigma has to be set or same as the length of the bounding box. Got {sigma}.")
        lower_point = (
            np.random.normal(loc=data_shape / 2, scale=sigma, size=len(data_shape)) - crop_shape / 2
        ).astype(int)
        lower_point = np.clip(lower_point, 0, limits)

    else:
        raise ValueError(f"Sampler is either `uniform` or `gaussian`. Got {sampler}.")

    for idx, _ in enumerate(crop_shape):
        bbox[offset + idx] = lower_point[idx]
        bbox[offset + ndim + idx] = crop_shape[idx]

    output = [crop_to_bbox(data, bbox) for data in data_list]

    if contiguous:
        output = [_.contiguous() for _ in output]

    if len(output) == 1:
        return output[0]

    return output
