import torch

from ans.autograd import Variable


class Optimizer:

    def __init__(self, parameters: list[Variable]) -> None:
        self.parameters = parameters

    def step(self) -> None:
        raise NotImplementedError

    def zero_grad(self) -> None:
        for param in self.parameters:
            param.grad = None


class SGD(Optimizer):

    def __init__(
            self,
            parameters: list[Variable],
            learning_rate: float = 1e-3,
            momentum: float = 0.,
            weight_decay: float = 0.
    ) -> None:
        super().__init__(parameters)

        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay

        ########################################
        # TODO: init _velocities to zeros

        raise NotImplementedError
        self._velocities: ...

        # ENDTODO
        ########################################

    def step(self) -> None:
        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
        ########################################


class Adam(Optimizer):

    def __init__(
            self,
            parameters: list[Variable],
            learning_rate: float = 1e-3,
            beta1: float = 0.9,
            beta2: float = 0.999,
            eps: float = 1e-08,
            weight_decay: float = 0.,
    ) -> None:
        super().__init__(parameters)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        ########################################
        # TODO: init _num_steps to zero, _m to zeros, _v to zeros

        raise NotImplementedError
        self._num_steps = ...
        self._m: dict[Variable, torch.Tensor] = ...
        self._v: dict[Variable, torch.Tensor] = ...

        # ENDTODO
        ########################################

    def step(self) -> None:
        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
        ########################################
