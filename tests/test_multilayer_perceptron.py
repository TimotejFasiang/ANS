import torch

import ans
from tests import ANSTestCase
from tests.test_linear_classification import TestAccuracy  # used in notebook via this module
from tests import randn_var


class TestSGD(ANSTestCase):

    def setUp(self) -> None:
        self.ans_model = ans.modules.Sequential(
            ans.modules.Linear(6, 5),
            ans.modules.Sigmoid(),
            ans.modules.Linear(5, 4),
            ans.modules.Sigmoid(),
            ans.modules.Linear(4, 3)
        )
        self.torch_model = torch.nn.Sequential(
            torch.nn.Linear(6, 5),
            torch.nn.Sigmoid(),
            torch.nn.Linear(5, 4),
            torch.nn.Sigmoid(),
            torch.nn.Linear(4, 3)
        )
        for tpar, apar in zip(self.torch_model.parameters(), self.ans_model.parameters()):
            tpar.data = apar.data.clone().t()  # torch.nn.Linear does x * w.t() + b

    def test_init(self) -> None:
        model = ans.modules.Linear(4, 4).to(dtype=torch.float16)
        optimizer = ans.optim.SGD(model.parameters())
        v = next(iter(optimizer._velocities.values()))
        self.assertIsInstance(v, torch.Tensor)
        self.assertEqual(v.dtype, torch.float16)
        self.assertTrue(torch.all(v == 0))

    def _test_config(self, learning_rate: float, weight_decay: float, momentum: float) -> None:
        ans_optimizer = ans.optim.SGD(
            self.ans_model.parameters(),
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            momentum=momentum
        )
        torch_optimizer = torch.optim.SGD(
            self.torch_model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )
        for i in range(3):
            for tpar, apar in zip(self.torch_model.parameters(), self.ans_model.parameters()):
                apar.grad = 0.1 * torch.randn_like(apar.data)
                tpar.grad = apar.grad.clone().t()
            ans_optimizer.step()
            torch_optimizer.step()
        for tpar, apar in zip(self.torch_model.parameters(), self.ans_model.parameters()):
            self.assertTensorsClose(apar.data, tpar.t())

    def test_sgd(self) -> None:
        self._test_config(torch.rand(1).item(), 0., 0.)

    def test_weight_decay(self) -> None:
        self._test_config(torch.rand(1).item(), torch.rand(1).item(), 0.)

    def test_momentum(self) -> None:
        self._test_config(torch.rand(1).item(), 0., torch.rand(1).item())

    def test_weight_decay_momentum(self) -> None:
        self._test_config(torch.rand(1).item(), torch.rand(1).item(), torch.rand(1).item())


class TestPreprocess(ANSTestCase):

    def test_values(self) -> None:
        x, y = self.params['preprocess_fn'](torch.arange(2 * 3 * 2 * 2).reshape(2, 3, 2, 2), torch.randint(10, (2,)))
        try:
            expected_x_centered = torch.tensor([
                [0.0000, 0.0039, 0.0078, 0.0118, 0.0157, 0.0196, 0.0235, 0.0275, 0.0314, 0.0353, 0.0392, 0.0431],
                [0.0471, 0.0510, 0.0549, 0.0588, 0.0627, 0.0667, 0.0706, 0.0745, 0.0784, 0.0824, 0.0863, 0.0902]
            ])
            self.assertTensorsClose(x, expected_x_centered)
        except AssertionError as e:
            if not self.params.get('centered', False):
                expected_x = torch.tensor([
                    [-0.5000, -0.4961, -0.4922, -0.4882, -0.4843, -0.4804, -0.4765, -0.4725, -0.4686, -0.4647, -0.4608, -0.4569],
                    [-0.4529, -0.4490, -0.4451, -0.4412, -0.4373, -0.4333, -0.4294, -0.4255, -0.4216, -0.4176, -0.4137, -0.4098]])
                self.assertTensorsClose(x, expected_x)
            else:
                raise e

    def test_dtype(self) -> None:
        x, y = self.params['preprocess_fn'](
            torch.randn(2, 3),
            torch.randint(10, (2,)),
            dtype=torch.float16
        )
        self.assertEqual(x.dtype, torch.float16)
        self.assertEqual(y.dtype, torch.int64)

    def test_device(self) -> None:
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            x, y = self.params['preprocess_fn'](
                torch.randn(2, 3),
                torch.randint(10, (2,)),
                device=device
            )
            self.assertEqual(x.device, device)
            self.assertEqual(y.device, device)
        else:
            self.skipTest("CUDA device not available for testing")


class TestSteps(ANSTestCase):

    def setUp(self) -> None:
        ans.utils.seed_everything(0)
        self.model = ans.modules.Sequential(
            ans.modules.Linear(6, 5),
            ans.modules.Sigmoid(),
            ans.modules.Linear(5, 4),
            ans.modules.Sigmoid(),
            ans.modules.Linear(4, 3)
        )

    def test_implementation(self) -> None:
        self.assertCalling(
            self.params['train_step_fn'],
            ['preprocess', 'model', 'criterion', 'accuracy']
        )
        self.assertNoLoops(self.params['train_step_fn'])

    def test_train_step(self) -> None:
        inputs = torch.randn(13, 6)
        targets = torch.randint(0, 3, (13, 1)).squeeze()
        criterion = ans.modules.SoftmaxCrossEntropy()
        optimizer = ans.optim.SGD(self.model.parameters(), learning_rate=0.1 * torch.rand(1).item())

        for i in range(torch.randint(10, 20, (1,))):
            loss1, acc1 = self.params['train_step_fn'](inputs, targets, self.model, criterion, optimizer)
        try:
            # uncentered
            expected_loss1, expected_acc1 = 1.0481, 0.6154
            self.assertTensorsClose(loss1, expected_loss1)
            self.assertTensorsClose(acc1, expected_acc1)
        except AssertionError as e:
            # centered
            expected_loss1, expected_acc1 = 1.0493, 0.61538
            self.assertTensorsClose(loss1, expected_loss1)
            self.assertTensorsClose(acc1, expected_acc1)

        for i in range(torch.randint(10, 20, (1,))):
            loss2, acc2 = self.params['train_step_fn'](inputs, targets, self.model, criterion, optimizer)
        try:
            # uncentered
            expected_loss2, expected_acc2 = 0.9832, 0.6154
            self.assertTensorsClose(loss2, expected_loss2)
            self.assertTensorsClose(acc2, expected_acc2)
        except AssertionError as e:
            # centered
            expected_loss2, expected_acc2 = 0.98356, 0.61538
            self.assertTensorsClose(loss2, expected_loss2)
            self.assertTensorsClose(acc2, expected_acc2)

    def test_val_step(self) -> None:
        inputs = torch.randn(13, 6)
        targets = torch.randint(0, 3, (13, 1)).squeeze()
        criterion = ans.modules.SoftmaxCrossEntropy()
        loss, acc = self.params['val_step_fn'](inputs, targets, self.model, criterion)
        try:
            # uncentered
            expected_loss, expected_acc = 1.2205, 0.1538
            self.assertTensorsClose(loss, expected_loss)
            self.assertTensorsClose(acc, expected_acc)
        except AssertionError:
            # centered
            expected_loss, expected_acc = 1.224795, 0.1538
            self.assertTensorsClose(loss, expected_loss)
            self.assertTensorsClose(acc, expected_acc)


class TestReLU(ANSTestCase):

    def test_implementation(self):
        self.assertNotCalling(ans.functional.ReLU.forward, ['relu', 'relu_'])
        self.assertCalling(ans.modules.ReLU.forward, ['apply'])

    def test_forward_function(self):
        x_var = randn_var(10, 4, std=10.)

        output, cache = ans.functional.ReLU.forward(x_var.data)
        self.assertIsInstance(output, torch.Tensor)
        self.assertIsInstance(cache, tuple)

        z_var = ans.functional.ReLU.apply(x_var)
        z = torch.relu(x_var.data)

        self.assertTensorsClose(z_var.data, z)

    def test_forward_module(self):
        relu = ans.modules.ReLU()
        x_var = randn_var(10, 7, std=10.)

        z_var = relu(x_var)
        self.assertIsInstance(z_var, ans.autograd.Variable)

        z = torch.relu(x_var.data)
        self.assertTensorsClose(z_var.data, z)

    def test_gradcheck(self):
        gradcheck_result = ans.autograd.gradcheck(
            ans.functional.ReLU.apply,
            (randn_var(10, 4, std=10.),),
            verbose=False
        )
        self.assertTrue(gradcheck_result)


class TestBatchNorm1d(ANSTestCase):

    def test_implementation(self):
        self.assertNotCalling(ans.functional.BatchNorm1d.forward, ['batch_norm'])
        self.assertCalling(ans.modules.Sigmoid.forward, ['apply'])

    @staticmethod
    def random_inputs(n: int, d: int, affine: bool = True) -> tuple[tuple, dict]:
        x_var = randn_var(n, d, mean=torch.randn(1).item(), std=torch.rand(1).item())
        gamma = randn_var(d) if affine else None
        beta = randn_var(d) if affine else None
        params = {
            'running_mean': torch.randn(d, dtype=x_var.data.dtype),
            'running_var': torch.rand(d, dtype=x_var.data.dtype),
            'momentum': torch.rand(1).item(),
            'eps': 1e-5 * torch.rand(1).item()
        }
        return (x_var, gamma, beta), params

    def test_output_types(self):
        (x_var, gamma, beta), params = TestBatchNorm1d.random_inputs(10, 4)
        output, cache = ans.functional.BatchNorm1d.forward(x_var.data, gamma.data, beta.data, **params, training=True)
        self.assertIsInstance(output, torch.Tensor)
        self.assertIsInstance(cache, tuple)

    def test_forward_function_affine(self):
        (x_var, gamma, beta), params = TestBatchNorm1d.random_inputs(10, 4)
        torch_running_mean = params['running_mean'].clone()
        torch_running_var = params['running_var'].clone()

        for i, training in enumerate([True, True, False]):
            z_var = ans.functional.BatchNorm1d.apply(x_var, gamma, beta, **params, training=training)
            z = torch.nn.functional.batch_norm(x_var.data, torch_running_mean, torch_running_var, gamma.data,
                                               beta.data, training=training, momentum=1-params['momentum'],
                                               eps=params['eps'])
            self.assertTensorsClose(params['running_mean'].data, torch_running_mean, msg=f"training[{i}]={training}")
            self.assertTensorsClose(params['running_var'].data, torch_running_var, msg=f"training[{i}]={training}")
            self.assertTensorsClose(z_var.data, z, msg=f"training[{i}]={training}")

    def test_forward_function_linear(self):
        (x_var, gamma, beta), params = TestBatchNorm1d.random_inputs(10, 4, affine=False)
        torch_running_mean = params['running_mean'].clone()
        torch_running_var = params['running_var'].clone()

        for i, training in enumerate([True, True, False]):
            z_var = ans.functional.BatchNorm1d.apply(x_var, None, None, **params, training=training)
            z = torch.nn.functional.batch_norm(x_var.data, torch_running_mean, torch_running_var, None, None,
                                               training=training, momentum=1-params['momentum'], eps=params['eps'])
            self.assertTensorsClose(params['running_mean'].data, torch_running_mean, msg=f"training[{i}]={training}")
            self.assertTensorsClose(params['running_var'].data, torch_running_var, msg=f"training[{i}]={training}")
            self.assertTensorsClose(z_var.data, z, msg=f"training[{i}]={training}")

    def test_forward_module(self):
        momentum = torch.rand(1).item()
        eps = torch.rand(1).item()
        ans_batchnorm = ans.modules.BatchNorm1d(7, momentum=momentum, eps=eps).to(dtype=torch.float64)
        torch_batchnorm = torch.nn.BatchNorm1d(7, eps=eps, momentum=1. - momentum, dtype=torch.float64)

        self.assertTensorsClose(ans_batchnorm.gamma.data, torch_batchnorm.weight)
        self.assertTensorsClose(ans_batchnorm.beta.data, torch_batchnorm.bias)
        self.assertTensorsClose(ans_batchnorm.running_mean, torch_batchnorm.running_mean)
        self.assertTensorsClose(ans_batchnorm.running_var, torch_batchnorm.running_var)

        for i in range(3):
            x_var = randn_var(10, 7, mean=torch.randn(1).item(), std=torch.rand(1).item())
            z_var = ans_batchnorm(x_var)
            z = torch_batchnorm(x_var.data)

            self.assertIsInstance(z_var, ans.autograd.Variable)
            self.assertTensorsClose(z_var.data, z, msg=f"i={i}")
            self.assertTensorsClose(ans_batchnorm.running_mean, torch_batchnorm.running_mean)
            self.assertTensorsClose(ans_batchnorm.running_var, torch_batchnorm.running_var)

    def test_gradcheck(self):
        for training in [True, False]:
            (x_var, gamma, beta), params = TestBatchNorm1d.random_inputs(11, 6)
            gradcheck_result = ans.autograd.gradcheck(
                ans.functional.BatchNorm1d.apply,
                (x_var, gamma, beta),
                params=dict(**params, training=training),
                verbose=False
            )
            self.assertTrue(gradcheck_result, msg=f"training={training}")


class TestDropout(ANSTestCase):

    def test_implementation(self):
        self.assertNotCalling(ans.functional.Dropout.forward, ['dropout', 'dropout_'])
        self.assertCalling(ans.modules.Dropout.forward, ['apply'])

    def test_forward_output_types(self):
        x_var = randn_var(100, 100)
        output, cache = ans.functional.ReLU.forward(x_var.data)
        self.assertIsInstance(output, torch.Tensor)
        self.assertIsInstance(cache, tuple)

    def test_forward_function_training(self):
        x_var = randn_var(100, 1000)
        p = torch.rand(1).item()
        z_var = ans.functional.Dropout.apply(x_var, p=p, training=True)
        null_ratio = (z_var.data.abs() < 1e-6).to(dtype=torch.float64).mean()
        self.assertTensorsClose(null_ratio, torch.tensor(p).double(), rtol=1e-2, atol=1e-2)

    def test_forward_function_eval(self):
        x_var = randn_var(100, 1000)
        p = torch.rand(1).item()
        z_var = ans.functional.Dropout.apply(x_var, p=p, training=False)
        self.assertTensorsClose(z_var.data, x_var.data)

    def test_forward_module(self):
        p = torch.rand(1).item()
        dropout = ans.modules.Dropout(p=p)
        x_var = randn_var(100, 1000, std=10.)

        z_var = dropout(x_var)
        self.assertIsInstance(z_var, ans.autograd.Variable)

        null_ratio = (z_var.data.abs() < 1e-6).to(dtype=torch.float64).mean()
        self.assertTensorsClose(null_ratio, torch.tensor(p).double(), rtol=1e-2, atol=1e-2)

    def test_gradcheck(self):
        for training in [True, False]:
            gradcheck_result = ans.autograd.gradcheck(
                ans.functional.Dropout.apply,
                (randn_var(10, 4),),
                params=dict(training=training, seed=torch.randint(1000, (1,)).item()),
                verbose=False
            )
            self.assertTrue(gradcheck_result, msg=f"training={training}")



class TestAdam(ANSTestCase):

    def setUp(self) -> None:
        self.ans_model = ans.modules.Sequential(
            ans.modules.Linear(6, 5),
            ans.modules.Sigmoid(),
            ans.modules.Linear(5, 4),
            ans.modules.Sigmoid(),
            ans.modules.Linear(4, 3)
        )
        self.torch_model = torch.nn.Sequential(
            torch.nn.Linear(6, 5),
            torch.nn.Sigmoid(),
            torch.nn.Linear(5, 4),
            torch.nn.Sigmoid(),
            torch.nn.Linear(4, 3)
        )
        for tpar, apar in zip(self.torch_model.parameters(), self.ans_model.parameters()):
            tpar.data = apar.data.clone().t()  # torch.nn.Linear does x * w.t() + b

    def test_init(self) -> None:
        model = ans.modules.Linear(4, 4).to(dtype=torch.float16)
        optimizer = ans.optim.Adam(model.parameters())
        for a in ('_m', '_v'):
            val = next(iter(getattr(optimizer, a).values()))
            self.assertIsInstance(val, torch.Tensor)
            self.assertEqual(val.dtype, torch.float16)
            self.assertTrue(torch.all(val == 0))

    def _test_config(self, learning_rate: float, weight_decay: float, beta1: float, beta2: float):
        ans_optimizer = ans.optim.Adam(
            self.ans_model.parameters(),
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            beta1=beta1,
            beta2=beta2,
            eps=torch.rand(1).item()
        )
        torch_optimizer = torch.optim.Adam(
            self.torch_model.parameters(),
            lr=learning_rate,
            betas=(beta1, beta2),
            weight_decay=weight_decay,
            eps=ans_optimizer.eps
        )
        for i in range(3):
            for tpar, apar in zip(self.torch_model.parameters(), self.ans_model.parameters()):
                apar.grad = 0.1 * torch.randn_like(apar.data)
                tpar.grad = apar.grad.clone().t()
            ans_optimizer.step()
            torch_optimizer.step()
        for tpar, apar in zip(self.torch_model.parameters(), self.ans_model.parameters()):
            self.assertTensorsClose(apar.data, tpar.t())

    def test_adam(self):
        self._test_config(torch.rand(1).item(), torch.rand(1).item(), torch.rand(1).item(), torch.rand(1).item())
        self._test_config(torch.rand(1).item(), torch.rand(1).item(), torch.rand(1).item(), torch.rand(1).item())


class TestBasicMLPAccuracy(ANSTestCase):

    def test_basic_mlp_accuracy(self) -> None:
        ok = bool(int(input('basic mlp val acc > 50 % (1/0)? ')))
        self.assertTrue(ok)


class TestAdvancedMLPAccuracy(ANSTestCase):

    def test_advanced_mlp_accuracy(self) -> None:
        ok = bool(int(input('advanced mlp val acc > 55 % (1/0)? ')))
        self.assertTrue(ok)
