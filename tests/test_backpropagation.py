from typing import Callable, TypeAlias, Union

import matplotlib.pyplot as plt
import torch

import ans
from tests import ANSTestCase


TensorOrVariable: TypeAlias = Union[torch.Tensor, ans.autograd.Variable]


class TestBinaryOp(ANSTestCase):

    shapes = (tuple(), (2,), (2, 3))

    @staticmethod
    def forward(x: TensorOrVariable, y: TensorOrVariable) -> TensorOrVariable:
        raise NotImplementedError

    def test_operation(self):
        for shape in self.shapes:
            # forward pass
            x_var, y_var, z_var = example_1(shape, op=self.forward)  # with Variables
            z = self.forward(x_var.data, y_var.data)  # with Tensors
            self.assertTensorsClose(z_var.data, z)

            # backward pass
            dz = torch.randn(shape)
            z.backward(gradient=dz)
            dx, dy = z_var.grad_fn(dz)
            self.assertTensorsClose(dx, x_var.data.grad)
            self.assertTensorsClose(dy, y_var.data.grad)


class TestAdd(TestBinaryOp):

    @staticmethod
    def forward(x: TensorOrVariable, y: TensorOrVariable) -> TensorOrVariable:
        return x + y


class TestSub(TestAdd):

    @staticmethod
    def forward(x: TensorOrVariable, y: TensorOrVariable) -> TensorOrVariable:
        return x - y


class TestMul(TestAdd):

    @staticmethod
    def forward(x: TensorOrVariable, y: TensorOrVariable) -> TensorOrVariable:
        return x * y


class TestPow(ANSTestCase):

    shapes = (tuple(), (2,), (2, 3))
    powers = (-1, -0.5, 0, 0.5, 1, 2, 3)

    def test_powers(self):
        for shape in self.shapes:
            for power in self.powers:
                # forward pass
                x = torch.rand(shape, requires_grad=True)
                y = x ** power
                x_var = ans.autograd.Variable(x)
                y_var = x_var ** power
                self.assertTensorsClose(y_var.data, y)

                # backward pass
                dy = torch.randn(shape)
                y.backward(dy)
                dx, = y_var.grad_fn(dy)
                self.assertTensorsClose(dx, x.grad)


class TestTopologicalSort(ANSTestCase):

    shapes = (tuple(), (2,), (2, 3))

    @staticmethod
    def is_predecessor(who: ans.autograd.Variable, of_whom: ans.autograd.Variable) -> bool:
        for node in of_whom.parents:
            if node is who:
                return True
            elif node.parents:
                return TestTopologicalSort.is_predecessor(who, node)
        return False

    def test_examples(self):
        for example_fn in (example_1, example_2, example_3):
            for shape in self.shapes:
                variables = example_fn(shape)
                variables_sorted = variables[-1].predecessors()
                self.assertTrue(len(variables), len(variables_sorted))
                ranks = {var: variables_sorted.index(var) for var in variables}
                for var1 in variables:
                    for var2 in variables:
                        if var1 is var2:
                            continue
                        if self.is_predecessor(var1, var2):
                            self.assertLess(ranks[var1], ranks[var2])


class TestBackprop(ANSTestCase):

    shapes = (tuple(), (2,), (2, 3))

    def test_examples(self):
        for example_fn in (example_1, example_2, example_3):
            for shape in self.shapes:
                variables = example_fn(shape)
                dout = torch.randn(shape)
                variables[-1].backprop(dout=dout)  # ans backprop
                for var in variables:
                    var.data.retain_grad()  # to check intermediate gradients even though they don't really matter
                variables[-1].data.backward(gradient=dout)  # pytorch backprop
                for var in variables:
                    self.assertTensorsClose(var.grad, var.data.grad)


class TestToGraphviz(ANSTestCase):

    def test_to_graphviz(self):
        (self.params['w'] + self.params['q'] + self.params['z']).to_graphviz().view()
        ok = bool(input('ok (1/0)? '))
        self.assertTrue(ok)


def example_1(
        shape: tuple[int, ...],
        op: Callable[[TensorOrVariable, TensorOrVariable], TensorOrVariable] = TestAdd.forward
) -> tuple[ans.autograd.Variable, ...]:
    u = ans.autograd.Variable(torch.randn(shape, requires_grad=True))
    v = ans.autograd.Variable(torch.randn(shape, requires_grad=True))
    w = op(u, v)
    return u, v, w


def example_2(shape: tuple[int, ...]) -> tuple[ans.autograd.Variable, ...]:
    x1 = ans.autograd.Variable(torch.randn(shape, requires_grad=True))
    a = ans.autograd.Variable(torch.randn(shape, requires_grad=True))
    x2 = ans.autograd.Variable(torch.randn(shape, requires_grad=True))

    x2_ = a * x2
    y = x1 + x2_
    z = y * y
    return x1, a, x2, x2_, y, z


def example_3(shape: tuple[int, ...]) -> tuple[ans.autograd.Variable, ...]:
    x = ans.autograd.Variable(torch.randn(shape, requires_grad=True))
    o = ans.autograd.Variable(torch.ones(shape, requires_grad=True))

    s = x * x
    p = s + o
    m = s - o
    q = p * m
    return x, o, s, p, m, q
