import functools
from typing import Any, Optional, Union

import torch
import numpy as np
from ans.autograd import Variable


class Function:

    @classmethod
    def apply(cls, *inputs: Union[Variable, Any], **params: Any) -> Variable:
        tensor_args = [i.data if isinstance(i, Variable) else i for i in inputs]
        output_data, cache = cls.forward(*tensor_args, **params)

        def grad_fn(dout: torch.Tensor) -> tuple[torch.Tensor, ...]:
            dinputs = cls.backward(dout, cache=cache)
            return tuple(dinputs[i] for i, inp in enumerate(inputs) if isinstance(inp, Variable))

        grad_fn.name = f"{cls.__name__}.backward"
        return Variable(
            output_data,
            parents=tuple(i for i in inputs if isinstance(i, Variable)),
            grad_fn=grad_fn
        )

    @staticmethod
    def forward(*inputs: torch.Tensor, **params: Any) -> tuple[torch.Tensor, tuple]:
        raise NotImplementedError

    @staticmethod
    def backward(doutput: torch.Tensor, cache=()) -> tuple[torch.Tensor, ...]:
        raise NotImplementedError

    def __str__(self):
        return f"{self.__class__.__name__}"

    def __repr__(self):
        return str(self)


class Linear(Function):

    @staticmethod
    def forward(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> tuple[torch.Tensor, tuple]:
        """
        Args:
            input: shape (num_samples, num_features)
            weight: shape (num_features, num_out)
            bias: shape (num_out,)
        Returns:
            output: shape (num_samples, num_out)
            cache: tuple of intermediate results to use in backward
        """

        ########################################
        # TODO: implement
        output = input.matmul(weight) + bias
        cache = input, weight, bias

        # ENDTODO
        ########################################

        return output, cache

    @staticmethod
    def backward(doutput: torch.Tensor, cache=()) -> tuple[torch.Tensor, ...]:
        """
        Args:
            doutput: gradient w.r.t. output of the forward pass; shape (num_samples, num_out)
            cache: cache from the forward pass
        Returns:
            tuple of gradient w.r.t. input, weight, bias in this order
        """

        ########################################
        # TODO: implement

        input, weight, bias = cache

        dinput = doutput.matmul(weight.t())
        dweight = input.t().matmul(doutput)
        dbias = doutput.sum(0)

        # ENDTODO
        ########################################

        return dinput, dweight, dbias


class Sigmoid(Function):

    @staticmethod
    def forward(input: torch.Tensor) -> tuple[torch.Tensor, tuple]:
        """
        Args:
            input: shape (num_samples, num_features)
        Returns:
            output: shape (num_samples, num_features)
            cache: tuple of intermediate results to use in backward
        """

        ########################################
        # TODO: implement

        output = 1 / (1 + torch.exp(-input))
        cache = output,

        # ENDTODO
        ########################################

        return output, cache

    @staticmethod
    def backward(doutput: torch.Tensor, cache=()) -> tuple[torch.Tensor, ...]:
        """
        Args:
            doutput: gradient w.r.t. output of the forward pass; shape (num_samples, num_out)
            cache: cache from the forward pass
        Returns:
            tuple of gradients w.r.t. input (single-element tuple)
        """

        ########################################
        # TODO: implement

        output = cache[0]
        dinput = doutput * output * (1 - output)

        # ENDTODO
        ########################################

        return dinput,


class SoftmaxCrossEntropy(Function):

    @staticmethod
    def forward(scores: torch.Tensor, targets: torch.Tensor, ignore_index: int = -1) -> tuple[torch.Tensor, tuple]:
        """
        Args:
            scores: shape (num_samples, num_out)
            targets: shape (num_samples,); dtype torch.int64
        Returns:
            output: shape () (scalar tensor)
            cache: tuple of intermediate results to use in backward
        """

        ########################################
        # TODO: implement

        scores = scores.float()  # Causes small errors in testing when this is not here???

        # Numerical stability
        max_scores, _ = torch.max(scores, dim=1, keepdim=True)
        scores -= max_scores
        exp_scores = torch.exp(scores)

        softmax_probs = exp_scores / torch.sum(exp_scores, dim=1, keepdim=True)
        target_probs = softmax_probs[torch.arange(scores.size(0)), targets]

        neg_log_likelihood = -torch.log(target_probs)

        if ignore_index >= 0:
            mask = (targets != ignore_index).double()
            neg_log_likelihood *= mask
        else:
            mask = None

        output = torch.mean(neg_log_likelihood).double()
        cache = (softmax_probs, targets, mask)

        # ENDTODO
        ########################################

        return output, cache

    @staticmethod
    def backward(doutput: torch.Tensor, cache=()) -> tuple[torch.Tensor, ...]:
        """
        Args:
            doutput: gradient w.r.t. output of the forward pass; shape ()
            cache: cache from the forward pass
        Returns:
            tuple of gradients w.r.t. scores (single-element tuple)
        """

        ########################################
        # TODO: implement

        softmax_probs, targets, mask = cache
        num_samples, num_out = softmax_probs.shape

        dscores = torch.zeros_like(softmax_probs)

        for i in range(num_samples):
            dscores[i, :] = softmax_probs[i, :]
            dscores[i, targets[i]] -= 1

        if mask is not None:
            dscores *= mask.view(-1, 1)

        dscores *= doutput / num_samples

        # ENDTODO
        ########################################

        return dscores,


class ReLU(Function):

    @staticmethod
    def forward(input: torch.Tensor) -> tuple[torch.Tensor, tuple]:
        """
        Args:
            input: shape (num_samples, ...)
        Returns:
            output: shape (num_samples, ...)
            cache: tuple of intermediate results to use in backward

        Operation is not inplace, i.e. it does not modify the input.
        """

        ########################################
        # TODO: implement

        output = torch.max(input, torch.zeros_like(input))
        cache = (input,)

        # ENDTODO
        ########################################

        return output, cache

    @staticmethod
    def backward(doutput: torch.Tensor, cache=()) -> tuple[torch.Tensor, ...]:
        """
        Args:
            doutput: gradient w.r.t. output of the forward pass; shape (num_samples, ...)
            cache: cache from the forward pass
        Returns:
            tuple of gradients w.r.t. input (single-element tuple)
        """

        ########################################
        # TODO: implement

        input, = cache
        dinput = (input > 0).float() * doutput

        # ENDTODO
        ########################################

        return dinput,


class Tanh(Function):

    @staticmethod
    def forward(input: torch.Tensor) -> tuple[torch.Tensor, tuple]:
        """
        Args:
            input: shape (num_samples, ...)
        Returns:
            output: shape (num_samples, ...)
            cache: tuple of intermediate results to use in backward
        """

        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
        ########################################

        return output, cache

    @staticmethod
    def backward(doutput: torch.Tensor, cache=()) -> tuple[torch.Tensor, ...]:
        """
        Args:
            doutput: gradient w.r.t. output of the forward pass; shape (num_samples, ...)
            cache: cache from the forward pass
        Returns:
            tuple of gradients w.r.t. input (single-element tuple)
        """

        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
        ########################################

        return dinput,


class Dropout(Function):

    @staticmethod
    def forward(
            input: torch.Tensor,
            p: float = 0.5,
            training: bool = False,
            seed: Optional[int] = None
    ) -> tuple[torch.Tensor, tuple]:
        """
        Args:
            input: shape (num_samples, ...)
            p: probability of element being zeroed out
            training: whether in training mode or eval mode
            seed: enable deterministic behavior (useful for gradient check)
        Returns:
            output: shape (num_samples, ...)
            cache: tuple of intermediate results to use in backward
        """

        # deterministic behavior for gradient check
        if seed is not None:
            torch.manual_seed(seed)

        ########################################
        # TODO: implement

        if training:
            with torch.no_grad():
                # During training, apply dropout
                mask = (torch.rand(input.size()) > p).float() / (1.0 - p)
                # Ensure that the mask is non-zero to prevent division by zero
                mask = torch.clamp(mask, min=1e-12)
                output = input * mask
                cache = mask
                assert not torch.isnan(output).any(), "NaN values in the output during training"
        else:
            # During evaluation, keep the input unchanged
            output = input
            cache = None

        # ENDTODO
        ########################################

        return output, cache

    @staticmethod
    def backward(doutput: torch.Tensor, cache=()) -> tuple[torch.Tensor, ...]:
        """
        Args:
            doutput: gradient w.r.t. output of the forward pass; shape (num_samples, ...)
            cache: cache from the forward pass
        Returns:
            tuple of gradients w.r.t. input (single-element tuple)
        """

        ########################################
        # TODO: implement

        dinput = doutput * cache if cache is not None else doutput

        # ENDTODO
        ########################################

        return dinput,


class BatchNorm1d(Function):

    @staticmethod
    def forward(
            input: torch.Tensor,
            gamma: Optional[torch.Tensor],
            beta: Optional[torch.Tensor],
            running_mean: Optional[torch.Tensor] = None,
            running_var: Optional[torch.Tensor] = None,
            momentum: float = 0.9,
            eps: float = 1e-05,
            training: bool = False
    ) -> tuple[torch.Tensor, tuple]:
        """

        Args:
            input: shape (num_samples, num_features)
            gamma: shape (num_features,)
            beta: shape (num_features,)
            running_mean: shape (num_features,)
            running_var: shape (num_features,)
            momentum: running average smoothing coefficient
            eps: for numerical stabilization
            training: whether in training mode or eval mode
        Returns:
            output: shape (num_samples, num_features)
            cache: tuple of intermediate results to use in backward
        """

        ########################################
        # TODO: implement

        import torch
        from typing import Optional, Tuple

        # Ensure double precision for running mean and variance
        if running_mean is not None:
            running_mean = running_mean.to(dtype=torch.float64)
        if running_var is not None:
            running_var = running_var.to(dtype=torch.float64)

        # Calculate mean and variance
        mean = input.mean(dim=0)
        var = input.var(dim=0, unbiased=False)

        # Update running mean and variance during training
        if training:
            if running_mean is not None and running_var is not None:
                running_mean.mul_(momentum).add_((1 - momentum) * mean)
                running_var.mul_(momentum).add_((1 - momentum) * var)

        # Normalize input
        x_hat = (input - mean) / torch.sqrt(var + eps)

        # Scale and shift
        if gamma is not None and beta is not None:
            output = gamma * x_hat + beta
        else:
            output = x_hat

        # Cache intermediate results for backward pass
        cache = (x_hat, gamma, beta, input, mean, var, eps)

        # ENDTODO
        ########################################

        return output, cache

    @staticmethod
    def backward(doutput: torch.Tensor, cache=()) -> tuple[torch.Tensor, ...]:
        """
        Args:
            doutput: gradient w.r.t. output of the forward pass; shape (num_samples, num_features)
            cache: cache from the forward pass
        Returns:
            tuple of gradients w.r.t. input (single-element tuple)
        """

        ########################################
        # TODO: implement

        x_hat, gamma, beta, input, mean, var, eps = cache
        N = input.shape[0]

        dgamma = (doutput * x_hat).sum(dim=0)
        dbeta = doutput.sum(dim=0)

        dx_hat = doutput * gamma
        dvar = -0.5 * (dx_hat * (input - mean)).sum(dim=0) * (var + eps) ** (-1.5)
        dmean = -dx_hat.sum(dim=0) / torch.sqrt(var + eps) - 2 * dvar * (input - mean).mean(dim=0)
        dinput = dx_hat / torch.sqrt(var + eps) + 2 * dvar * (input - mean) / N + dmean / N

        # ENDTODO
        ########################################

        return dinput, dgamma, dbeta


class BatchNorm2d(Function):

    @staticmethod
    def forward(
            input: torch.Tensor,
            gamma: Optional[torch.Tensor],
            beta: Optional[torch.Tensor],
            running_mean: Optional[torch.Tensor] = None,
            running_var: Optional[torch.Tensor] = None,
            momentum: float = 0.9,
            eps: float = 1e-05,
            training: bool = False
    ) -> tuple[torch.Tensor, tuple]:
        """
        Spatial BatchNorm for convolutional networks

        Args:
            input: shape (num_samples, num_channels, height, width)
            gamma: shape (num_channels,)
            beta: shape (num_channels,)
            running_mean: shape (num_channels,)
            running_var: shape (num_channels,)
            momentum: running average smoothing coefficient
            eps: for numerical stabilization
            training: whether in training mode or eval mode
        Returns:
            output: shape (num_samples, num_channels, height, width)
            cache: tuple of intermediate results to use in backward
        """

        ########################################
        # TODO: implement

        N, C, H, W = input.shape
        input_reshaped = input.permute(0, 2, 3, 1).contiguous().view(-1, C)

        output_reshaped, cache = BatchNorm1d.forward(
            input_reshaped, gamma, beta, running_mean, running_var, momentum, eps, training
        )

        output = output_reshaped.view(N, H, W, C).permute(0, 3, 1, 2).contiguous()

        # ENDTODO
        ########################################

        return output, cache

    @staticmethod
    def backward(doutput: torch.Tensor, cache=()) -> tuple[torch.Tensor, ...]:
        """
        Args:
            doutput: gradient w.r.t. output of the forward pass; shape (num_samples, num_channels, height, width)
            cache: cache from the forward pass
        Returns:
            tuple of gradients w.r.t. input (single-element tuple)
        """

        ########################################
        # TODO: implement

        N, C, H, W = doutput.shape
        doutput_reshaped = doutput.permute(0, 2, 3, 1).contiguous().view(-1, C)

        dinput_reshaped, dgamma, dbeta = BatchNorm1d.backward(doutput_reshaped, cache)

        dinput = dinput_reshaped.view(N, H, W, C).permute(0, 3, 1, 2).contiguous()

        # ENDTODO
        ########################################

        return dinput, dgamma, dbeta


class Conv2d(Function):

    @staticmethod
    def forward(
            input: torch.Tensor,
            weight: torch.Tensor,
            bias: Optional[torch.Tensor],
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            groups: int = 1
    ) -> tuple[torch.Tensor, tuple]:
        """
        Args:
            input: shape (num_samples(minibatch), num_channels(in_channels), height, width)
            weight: shape (num_filters(out_channels), num_channels, kernel_size[0], kernel_size[1])
            bias: shape (num_filters,)
            stride: convolution step size
            padding: how much should the input be padded on each side by zeroes
            dilation: see torch.nn.functional.conv2d
            groups: see torch.nn.functional.conv2d

        Returns:
            output: shape (num_samples, num_filters, output_height, output_width)
            cache: tuple of intermediate results to use in backward
        """
        ########################################
        # TODO: implement
        #output = torch.nn.functional.conv2d(input, weight, bias=torch.tensor(bias), stride=stride, padding=padding, dilation=dilation, groups=groups)        
        output = torch.nn.functional.conv2d(input, weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
        cache = (input, weight, bias, stride, padding, dilation, groups)
        # ENDTODO
        ########################################

        return output, cache

    @staticmethod
    def backward(doutput: torch.Tensor, cache=()) -> tuple[torch.Tensor, ...]:
        """
        Args:
            doutput: gradient w.r.t. output of the forward pass; shape (num_samples(in_channels), num_filters(out_channels), output_height, output_width)
            cache: cache from the forward pass
        Returns:
            tuple of gradients w.r.t. input, weight and bias
        """

        ########################################
        # TODO: implement

        input, weight, bias, stride, padding, dilation, groups = cache
        batch_size, channels, input_height, input_width = input.shape[0], input.shape[1], input.shape[2], input.shape[3]
        output_height, output_width = doutput.shape[2], doutput.shape[3]
        kernel_size_height, kernel_size_width = weight.shape[2], weight.shape[3]

        output_padding_height = input_height - (output_height - 1) * stride + 2 * padding - (
                kernel_size_height - 1) * dilation - 1
        output_padding_width = input_width - (output_width - 1) * stride + 2 * padding - (
                kernel_size_width - 1) * dilation - 1

        transpose_output = torch.nn.functional.conv_transpose2d(doutput, weight, stride=stride, padding=padding, dilation=dilation, groups=groups, output_padding=(output_padding_height, output_padding_width))  # WORKS for all

        num_filters, num_channels, kernel_size, kernel_size = weight.size()
        num_samples, num_channels, height, width = input.size()

        #grad_w = torch.zeros((num_filters, num_channels, kernel_size, kernel_size), dtype=doutput.dtype, device=torch.device('cpu'))  # USE ONLY FOR CPU !!!
        grad_w = torch.zeros((num_filters, num_channels, kernel_size, kernel_size), dtype=doutput.dtype, device=torch.device('cuda'))  # USE ONLY FOR CUDA !!!
        
        for s in range(num_samples):
            temp_input = input[s, :, :, :].unsqueeze(1)
            temp_doutput = doutput[s, :, :, :].unsqueeze(1)
            
            temp_conv2d = torch.nn.functional.conv2d(temp_input, temp_doutput, stride=dilation, padding=padding, dilation=stride, groups=groups).squeeze()
            cut_conv2d = torch.permute((temp_conv2d[:, :, :kernel_size, :kernel_size]), (1, 0, 2, 3))
            grad_w[:, :, :, :] += cut_conv2d
        
        grad_b = doutput.sum(dim=(0, 2, 3), keepdim=True).squeeze()  # WORKS for all
        dinput = transpose_output
        dweight = grad_w
        dbias = grad_b
        # ENDTODO
        ########################################

        return dinput, dweight, dbias


# conv_result = torch.nn.functional.conv2d(transpose_output[n, :, :, :].unsqueeze(0), weight[f].unsqueeze(0)).squeeze()
# conv_result = conv_result[:grad_w.shape[-2], :grad_w.shape[-1]]


class MaxPool2d(Function):

    @staticmethod
    def forward(input: torch.Tensor, window_size: int = 2) -> tuple[torch.Tensor, tuple]:
        """
        Args:
            input: shape (num_samples, num_channels, height, width)
            window_size: size of pooling window
        Returns:
            output: shape (num_samples, num_channels, height / window_size, width / window_size)
            cache: tuple of intermediate results to use in backward
        """

        ########################################
        # TODO: implement

        #import numpy as np
        batch_size, channels, height, width = input.size()
        w_trim_amount = (width - window_size) % window_size
        if w_trim_amount > 0:
            input = input[:, :, :, :-w_trim_amount]
        h_trim_amount = (height - window_size) % window_size
        if h_trim_amount > 0:
            input = input[:, :, :-h_trim_amount, :]


        #
        #
        #
        #
        #
        batch_size, channels, height, width = input.size()
        input_reshaped = input.unfold(2, window_size, window_size).unfold(3, window_size, window_size)
        input_reshaped = input_reshaped.reshape(batch_size, channels, height // window_size, width // window_size, -1)

        output, _ = input_reshaped.max(dim=-1)
        input_flat = input.reshape(batch_size, channels, -1).squeeze()

        input_flat = input_flat.double()
        output = output.double()

        sort_idx = torch.argsort(input_flat.reshape(-1))
        output_flat = sort_idx[torch.searchsorted(input_flat.reshape(-1), output.reshape(-1), sorter=sort_idx)]

        indices = torch.unravel_index(output_flat.view(output.shape), input_flat.shape)[-1]
        output = torch.tensor(output)
        indices = torch.tensor(indices)
        cache = (input.size(), window_size, indices, w_trim_amount, h_trim_amount)
        
        # ENDTODO
        ########################################

        return torch.tensor(np.float32(output.cpu())).to('cuda'), cache  # USE ONLY FOR CUDA !!!
        #return np.float32(output.cpu()), cache                          # USE ONLY FOR CPU !!!
        #return output, cache                                            # USE ONLY FOR TESTS !!!

    @staticmethod
    def backward(doutput: torch.Tensor, cache=()) -> tuple[torch.Tensor, ...]:
        """
        Args:
            doutput: gradient w.r.t. output of the forward pass; shape (num_samples, num_channels, height / window_size, width / window_size)
            cache: cache from the forward pass
        Returns:
            tuple of gradients w.r.t. input (single-element tuple)
        """

        ########################################
        # TODO: implement
        input_size, window_size, indices, w_trim_amount, h_trim_amount = cache
        dinput = torch.nn.functional.max_unpool2d(doutput, indices, kernel_size=window_size, stride=window_size, padding=0, output_size=None)

        dinput = torch.nn.functional.pad(dinput, (0, w_trim_amount, 0, h_trim_amount, 0, 0, 0, 0), value=0)

        # ENDTODO
        ########################################
        return dinput,
