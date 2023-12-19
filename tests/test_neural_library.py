import torch

import ans
from tests import ANSTestCase, randn_var


class TestLinearFunction(ANSTestCase):

    def test_implementaiton(self):
        self.assertNotCalling(ans.functional.Linear.forward, ['linear'])

    def test_forward(self):
        x_var = randn_var(10, 4)
        w_var = randn_var(4, 3)
        b_var = randn_var(3)

        output, cache = ans.functional.Linear.forward(x_var.data, w_var.data, b_var.data)
        self.assertIsInstance(output, torch.Tensor)
        self.assertIsInstance(cache, tuple)

        z_var = ans.functional.Linear.apply(x_var, w_var, b_var)
        z = torch.nn.functional.linear(x_var.data, w_var.data.t(), b_var.data)
        self.assertTensorsClose(z_var.data, z)

    def test_gradcheck(self):
        gradcheck_result = ans.autograd.gradcheck(
            ans.functional.Linear.apply,
            (randn_var(5, 4), randn_var(4, 3), randn_var(3)),
            verbose=False
        )
        self.assertTrue(gradcheck_result)


class TestSigmoidFunction(ANSTestCase):

    def test_implementation(self):
        self.assertNotCalling(ans.functional.Sigmoid.forward, ['sigmoid'])

    def test_forward(self):
        x_var = randn_var(10, 4)

        output, cache = ans.functional.Sigmoid.forward(x_var.data)
        self.assertIsInstance(output, torch.Tensor)
        self.assertIsInstance(cache, tuple)

        z_var = ans.functional.Sigmoid.apply(x_var)
        z = torch.sigmoid(x_var.data)

        self.assertTensorsClose(z_var.data, z)

    def test_gradcheck(self):
        gradcheck_result = ans.autograd.gradcheck(
            ans.functional.Sigmoid.apply,
            (randn_var(5, 4),),
            verbose=False
        )
        self.assertTrue(gradcheck_result)


class TestSoftmaxCrossEntropyFunction(ANSTestCase):

    def test_implementation(self):
        self.assertNotCalling(ans.functional.SoftmaxCrossEntropy.forward, ['cross_entropy'])

    def test_forward(self):
        x_var = randn_var(10, 4)
        y = torch.randint(4, (10,))

        output, cache = ans.functional.SoftmaxCrossEntropy.forward(x_var.data,y)
        self.assertIsInstance(output, torch.Tensor)
        self.assertIsInstance(cache, tuple)

        l_var = ans.functional.SoftmaxCrossEntropy.apply(x_var, y)
        l = torch.nn.functional.cross_entropy(x_var.data, y)

        self.assertTensorsClose(l_var.data, l)

    def test_gradcheck(self):
        x_var = randn_var(10, 4)
        y = torch.randint(4, (10,))

        gradcheck_result = ans.autograd.gradcheck(
            ans.functional.SoftmaxCrossEntropy.apply,
            (x_var, y),
            verbose=False
        )
        self.assertTrue(gradcheck_result)


class TestLinearModule(ANSTestCase):

    def test_init(self):
        ans.utils.seed_everything(0)
        linear = ans.modules.Linear(8, 5)
        expected_weight = torch.tensor([
            [-0.0026,  0.1897, -0.2910, -0.2602, -0.1362],
            [ 0.0948, -0.0070,  0.2803, -0.0314,  0.0936],
            [-0.1068, -0.0695, -0.3378, -0.2342, -0.1457],
            [ 0.0131,  0.1398,  0.2121, -0.2397, -0.1540],
            [ 0.1284,  0.2936, -0.0728,  0.2646, -0.0570],
            [ 0.0374,  0.3201, -0.3280, -0.2226, -0.0895],
            [-0.1378,  0.3055, -0.2292, -0.1628, -0.2470],
            [-0.3311, -0.2064,  0.3039,  0.1578,  0.1714]
        ])
        expected_bias = torch.tensor([0., 0., 0., 0., 0.])
        self.assertIsInstance(linear.weight, ans.autograd.Variable)
        self.assertIsInstance(linear.bias, ans.autograd.Variable)
        self.assertTensorsClose(linear.weight.data, expected_weight)
        self.assertTensorsClose(linear.bias.data, expected_bias)

    def test_implementaiton(self):
        self.assertCalling(ans.modules.Linear.__init__, ['rand', 'zeros'])
        self.assertCalling(ans.modules.Linear.forward, ['apply'])

    def test_forward(self):
        linear = ans.modules.Linear(7, 6).to(dtype=torch.float64)
        x_var = randn_var(10, 7)

        z_var = linear(x_var)
        self.assertIsInstance(z_var, ans.autograd.Variable)

        z = torch.nn.functional.linear(x_var.data, linear.weight.data.t(), linear.bias.data)
        self.assertTensorsClose(z_var.data, z)


class TestSigmoidModule(ANSTestCase):

    def test_implementaiton(self):
        self.assertCalling(ans.modules.Sigmoid.forward, ['apply'])

    def test_forward(self):
        sigmoid = ans.modules.Sigmoid()
        x_var = randn_var(10, 7)

        z_var = sigmoid(x_var)
        self.assertIsInstance(z_var, ans.autograd.Variable)

        z = torch.sigmoid(x_var.data)
        self.assertTensorsClose(z_var.data, z)


class TestSoftmaxCrossEntropyModule(ANSTestCase):

    def test_implementaiton(self):
        self.assertCalling(ans.modules.SoftmaxCrossEntropy.forward, ['apply'])

    def test_forward(self):
        sce = ans.modules.SoftmaxCrossEntropy()
        x_var = randn_var(10, 4)
        y = torch.randint(4, (10,))

        z_var = sce(x_var, y)
        self.assertIsInstance(z_var, ans.autograd.Variable)

        z = torch.nn.functional.cross_entropy(x_var.data, y)
        self.assertTensorsClose(z_var.data, z)
