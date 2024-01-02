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
        
        self._velocities: Dict[Variable, torch.Tensor] = {param: torch.zeros_like(param.data) for param in self.parameters}
        
        # ENDTODO
        ########################################

    def step(self) -> None:
        ########################################
        # TODO: implement

        for param, velocity in self._velocities.items():
            if param.grad is not None:
                grad = param.grad + self.weight_decay * param.data
                velocity.data = self.momentum * velocity.data - self.learning_rate * grad
                param.data = param.data + velocity.data

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
        
        self._num_steps = 0
        self._m = {param: torch.zeros_like(param.data) for param in self.parameters}
        self._v = {param: torch.zeros_like(param.data) for param in self.parameters}

        # ENDTODO
        ########################################

    def step(self) -> None:
        ########################################
        # TODO: implement

        self._num_steps += 1

        for param in self.parameters:
            grad = param.grad.data

            if self.weight_decay != 0:
                grad = grad.add(self.weight_decay, param.data)

            self._m[param] = self.beta1 * self._m[param] + (1 - self.beta1) * grad
            self._v[param] = self.beta2 * self._v[param] + (1 - self.beta2) * (grad ** 2)

            m_hat = self._m[param] / (1 - self.beta1 ** self._num_steps)
            v_hat = self._v[param] / (1 - self.beta2 ** self._num_steps)

            update = -self.learning_rate * m_hat / (v_hat.sqrt() + self.eps)
            param.data.add_(update)

        # ENDTODO
        ########################################
