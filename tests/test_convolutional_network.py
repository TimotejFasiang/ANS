import random

import torch
import torchvision.datasets

import ans
from tests import ANSTestCase
from tests import randn_var
from tests.test_linear_classification import TestAccuracy


class TestConv2dFunction(ANSTestCase):

    def test_implementaiton(self) -> None:
        self.assertCalling(ans.functional.Conv2d.forward, ['conv2d'])
        self.assertNoLoops(ans.functional.Conv2d.forward)
        self.assertCalling(ans.functional.Conv2d.backward, ['conv2d', 'conv_transpose2d'])
        self.assertNotCalling(ans.functional.Conv2d.backward, ['conv2d_input', 'conv2d_weight', 'conv2d_bias'])

    def test_output_types(self) -> None:
        x_var = randn_var(11, 4, 13, 16)
        w_var = randn_var(8, 4, 5, 5)
        b_var = randn_var(8)
        output, cache = ans.functional.Conv2d.forward(x_var.data, w_var.data, b_var.data)
        self.assertIsInstance(output, torch.Tensor)
        self.assertIsInstance(cache, tuple)

    def _test_forward(self, n: int, c: int, m: int, m_: int, f: int, k: int, s: int = 1, p: int = 0, d: int = 1, g: int = 1) -> None:
        x_var = randn_var(n, c, m, m_)
        w_var = randn_var(f, c // g, k, k)
        b_var = randn_var(f)
        z_var = ans.functional.Conv2d.apply(x_var, w_var, b_var, stride=s, padding=p, dilation=d, groups=g)
        z = torch.nn.functional.conv2d(x_var.data, w_var.data, b_var.data, stride=s, padding=p, dilation=d, groups=g)
        self.assertTensorsClose(z_var.data, z)

    def test_forward(self) -> None:
        self._test_forward(11, 4, 13, 16, 8, 5)
        self._test_forward(11, 4, 13, 16, 8, 5, s=2)
        self._test_forward(11, 4, 13, 16, 8, 5, s=2, p=3)
        self._test_forward(11, 4, 13, 16, 8, 5, s=3, p=2)
        self._test_forward(11, 4, 13, 16, 8, 5, s=2, p=4, d=2)

    def _test_backward(self, n: int, c: int, m: int, m_: int, f: int, k: int, s: int = 1, p: int = 0, d: int = 1, g: int = 1) -> None:
        gradcheck_result = ans.autograd.gradcheck(
            ans.functional.Conv2d.apply,
            (randn_var(n, c, m, m_), randn_var(f, c // g, k, k), randn_var(f)),
            params=dict(stride=s, padding=p, dilation=d, groups=g),
            verbose=False
        )
        self.assertTrue(gradcheck_result)

    def test_backward(self) -> None:
        self._test_backward(11, 5, 13, 16, 8, 5)
        self._test_backward(11, 5, 13, 16, 8, 5, s=2)
        self._test_backward(11, 5, 13, 16, 8, 5, s=2, p=3)
        self._test_backward(11, 5, 13, 16, 8, 5, s=3, p=2)
        self._test_backward(11, 5, 13, 16, 8, 5, s=3, p=2, d=2)


class TestConv2dFunctionOptional(TestConv2dFunction):

    def test_forward(self) -> None:
        pass

    def test_backward(self) -> None:
        pass

    def test_forward_groups(self) -> None:
        self._test_forward(11, 4, 13, 16, 8, 5, g=2, s=3, p=2, d=2)

    def test_backward_groups(self) -> None:
        self._test_backward(11, 4, 13, 16, 8, 5)
    
    def test_backward_no_loops(self) -> None:
        self.assertNoLoops(ans.functional.Conv2d.backward)


class TestConv2dModule(ANSTestCase):

    def test_init(self):
        conv = ans.modules.Conv2d(256, 512, 5, padding=2)
        self.assertIsInstance(conv.weight, ans.autograd.Variable)
        self.assertIsInstance(conv.bias, ans.autograd.Variable)
        self.assertTensorsClose(conv.weight.data.mean(), torch.tensor(0.))
        self.assertGreaterEqual(conv.weight.data.min(), -0.0125)
        self.assertLessEqual(conv.weight.data.max(), 0.0125)

        h = conv.weight.data.histogram(bins=10, range=(-0.0125, 0.0125))[0] / conv.weight.data.numel()
        expected_h = 0.1 * torch.ones(10)
        self.assertTensorsClose(h, expected_h, rtol=1e-2, atol=1e-2)

        expected_bias = torch.zeros(512)
        self.assertTensorsClose(conv.bias.data, expected_bias)

    def test_implementaiton(self):
        self.assertCalling(ans.modules.Conv2d.__init__, ['rand', 'zeros'])
        self.assertCalling(ans.modules.Conv2d.forward, ['apply'])

    def test_forward(self):
        conv = ans.modules.Conv2d(4, 10, 3, padding=1).to(dtype=torch.float64)
        x_var = randn_var(10, 4, 6, 9)

        z_var = conv(x_var)
        self.assertIsInstance(z_var, ans.autograd.Variable)

        z = torch.nn.functional.conv2d(x_var.data, conv.weight.data, conv.bias.data, padding=1)
        self.assertTensorsClose(z_var.data, z)


class TestReLU(ANSTestCase):

    def test_implementation(self):
        self.assertNotCalling(ans.functional.ReLU.forward, ['relu', 'relu_'])
        self.assertCalling(ans.modules.ReLU.forward, ['apply'])

    def _test_forward_function(self, *shape: int) -> None:
        x_var = randn_var(*shape, std=10.)

        output, cache = ans.functional.ReLU.forward(x_var.data)
        self.assertIsInstance(output, torch.Tensor)
        self.assertIsInstance(cache, tuple)

        z_var = ans.functional.ReLU.apply(x_var)
        z = torch.relu(x_var.data)

        self.assertTensorsClose(z_var.data, z)

    def test_forward_function(self) -> None:
        self._test_forward_function(10, 4)
        self._test_forward_function(10, 4, 3)
        self._test_forward_function(10, 4, 3, 8)

    def _test_forward_module(self, *shape: int) -> None:
        relu = ans.modules.ReLU()
        x_var = randn_var(*shape, std=10.)

        z_var = relu(x_var)
        self.assertIsInstance(z_var, ans.autograd.Variable)

        z = torch.relu(x_var.data)
        self.assertTensorsClose(z_var.data, z)

    def test_forward_module(self) -> None:
        self._test_forward_module(10, 4)
        self._test_forward_module(10, 4, 3)
        self._test_forward_module(10, 4, 3, 8)

    def _test_gradcheck(self, *shape: int) -> None:
        gradcheck_result = ans.autograd.gradcheck(
            ans.functional.ReLU.apply,
            (randn_var(*shape, std=10.),),
            eps=1e-3,
            verbose=False
        )
        self.assertTrue(gradcheck_result)

    def test_gradcheck(self) -> None:
        self._test_gradcheck(10, 4)
        self._test_gradcheck(10, 4, 3)
        self._test_gradcheck(10, 4, 3, 8)


class TestMaxPool2dFunction(ANSTestCase):

    def test_implementaiton(self) -> None:
        self.assertNotCalling(ans.functional.MaxPool2d.forward, ['max_pool2d'])
        self.assertNoLoops(ans.functional.MaxPool2d.forward)
        self.assertNoLoops(ans.functional.MaxPool2d.backward)

    def test_output_types(self) -> None:
        x_var = randn_var(11, 4, 5, 8)
        output, cache = ans.functional.MaxPool2d.forward(x_var.data, 2)
        self.assertIsInstance(output, torch.Tensor)
        self.assertIsInstance(cache, tuple)

    def _test_forward(self, n: int, c: int, h: int, w: int, k: int) -> None:
        x_var = randn_var(n, c, h, w)
        z_var = ans.functional.MaxPool2d.apply(x_var, window_size=k)
        z = torch.nn.functional.max_pool2d(x_var.data, k)
        self.assertTensorsClose(z_var.data, z)

    def test_forward(self) -> None:
        self._test_forward(6, 4, 5, 8, 2)
        self._test_forward(6, 4, 5, 8, 3)
        self._test_forward(6, 4, 5, 8, 5)

    def _test_backward(self, n: int, c: int, h: int, w: int, k: int) -> None:
        gradcheck_result = ans.autograd.gradcheck(
            ans.functional.MaxPool2d.apply,
            (randn_var(n, c, h, w),),
            params=dict(window_size=k),
            eps=1e-4,
            verbose=False
        )
        self.assertTrue(gradcheck_result)

    def test_backward(self) -> None:
        self._test_backward(6, 4, 5, 8, 2)
        self._test_backward(6, 4, 5, 8, 3)
        self._test_backward(6, 4, 5, 8, 5)


class TestMaxPool2dModule(ANSTestCase):

    def test_implementaiton(self) -> None:
        self.assertCalling(ans.modules.MaxPool2d.forward, ['apply'])

    def test_forward(self) -> None:
        maxpool = ans.modules.MaxPool2d(3).to(dtype=torch.float64)
        x_var = randn_var(10, 4, 6, 9)

        z_var = maxpool(x_var)
        self.assertIsInstance(z_var, ans.autograd.Variable)

        z = torch.nn.functional.max_pool2d(x_var.data, 3)
        self.assertTensorsClose(z_var.data, z)


class TestReshape(ANSTestCase):

    def test_implementaiton(self) -> None:
        self.assertNoLoops(ans.autograd.Variable.reshape)

    def _test_forward(self, x_var, *shape: int) -> None:
        z_var = x_var.reshape(*shape)
        self.assertIsInstance(z_var, ans.autograd.Variable)
        self.assertEqual(len(z_var.data.shape), len(shape))
        shape_numel = torch.prod(torch.tensor(shape)).abs().item()
        expected_shape = [d if d > -1 else x_var.data.numel() / shape_numel for d in shape]
        self.assertListEqual(list(z_var.data.shape), expected_shape)

    def test_forward(self) -> None:
        x_var = randn_var(6, 4, 5, 8, 3)
        self._test_forward(x_var, 6, 4, 5, 8 * 3)
        self._test_forward(x_var, 6 * 4, 5, 8 * 3)
        self._test_forward(x_var, 6, 4 * 5, 8 * 3)
        self._test_forward(x_var, 6, 4 * 5, -1)
        self._test_forward(x_var, -1)

    def _test_backward(self, x_var: ans.autograd.Variable, *shape: int) -> None:
        gradcheck_result = ans.autograd.gradcheck(
            lambda var: var.reshape(*shape),
            (x_var,),
            verbose=False
        )
        self.assertTrue(gradcheck_result)

    def test_backward(self) -> None:
        x_var = randn_var(6, 4, 5, 8, 3)
        self._test_backward(x_var, 6, 4, 5, 8 * 3)
        self._test_backward(x_var, 6 * 4, 5, 8 * 3)
        self._test_backward(x_var, 6, 4 * 5, 8 * 3)
        self._test_backward(x_var, 6, 4 * 5, -1)
        self._test_backward(x_var, -1)


class TestFlattenModule(ANSTestCase):

    def test_implementaiton(self) -> None:
        self.assertCalling(ans.modules.Flatten.forward, ['reshape'])

    def test_forward(self):
        flatten = ans.modules.Flatten()
        x_var = randn_var(10, 4, 6, 9)

        z_var = flatten(x_var)
        self.assertIsInstance(z_var, ans.autograd.Variable)

        z = torch.nn.Flatten()(x_var.data)
        self.assertTensorsClose(z_var.data, z)


class TestPreprocess(ANSTestCase):

    def test_values(self) -> None:
        ans.utils.seed_everything(0)
        x, y = self.params['preprocess_fn'](255 * torch.rand(2 * 3 * 2 * 2).reshape(2, 3, 2, 2), torch.randint(10, (2,)))
        expected_x = torch.tensor([
            [[[-0.0037, -0.1926, -0.0444],
              [-0.4115, -0.0099, -0.1511]],
             [[ 0.2682,  0.1341,  0.1323],
              [-0.3680,  0.3964, -0.0983]]],
            [[[-0.4777,  0.1977,  0.1816],
              [-0.2061, -0.3390, -0.1029]],
             [[-0.3311,  0.3000,  0.4152],
              [ 0.0185, -0.2177,  0.3742]]]
        ])
        self.assertTensorsClose(x, expected_x)

    def test_dtype(self) -> None:
        x, y = self.params['preprocess_fn'](
            torch.randn(2, 3, 4, 5),
            torch.randint(10, (2,)),
            dtype=torch.float16
        )
        self.assertEqual(x.dtype, torch.float16)
        self.assertEqual(y.dtype, torch.int64)

    def test_device(self) -> None:
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            x, y = self.params['preprocess_fn'](
                torch.randn(2, 3, 4, 5),
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
            ans.modules.Conv2d(3, 7, 5, padding=2),
            ans.modules.Sigmoid(),
            ans.modules.Flatten(),
            ans.modules.Linear(448, 10)
        )
        # self.model.to(dtype=torch.float64)

    def test_implementation(self) -> None:
        self.assertCalling(
            self.params['train_step_fn'],
            ['model', 'criterion', 'accuracy']
        )
        self.assertNoLoops(self.params['train_step_fn'])

    def test_train_step(self) -> None:
        # inputs = 255 * torch.rand(10, 616)
        # targets = torch.randint(0, 4, (10, 1)).squeeze()
        cifar = torchvision.datasets.CIFAR10('../data', train=True)
        inputs = torch.tensor(cifar.data[:5, ::4, ::4, :].transpose(0, 3, 1, 2) / 255 - 0.5).float()
        targets = torch.tensor(cifar.targets[:5]).long()
        criterion = ans.modules.SoftmaxCrossEntropy()
        optimizer = ans.optim.SGD(self.model.parameters(), learning_rate=0.03)

        for i in range(10):
            loss1, acc1 = self.params['train_step_fn'](inputs, targets, self.model, criterion, optimizer)
            # print(loss1, acc1)
        expected_loss1, expected_acc1 = 1.405117, 0.4
        self.assertTensorsClose(loss1, expected_loss1)
        self.assertTensorsClose(acc1, expected_acc1)

        for i in range(90):
            loss2, acc2 = self.params['train_step_fn'](inputs, targets, self.model, criterion, optimizer)
            # print(loss2, acc2)
        expected_loss2, expected_acc2 = 0.9610665, 0.6
        self.assertTensorsClose(loss2, expected_loss2)
        self.assertTensorsClose(acc2, expected_acc2)

    def test_val_step(self) -> None:
        # inputs = 255 * torch.rand(10, 3, 8, 11)
        # targets = torch.randint(0, 4, (10, 1)).squeeze()
        cifar = torchvision.datasets.CIFAR10('../data', train=False)
        inputs = torch.tensor(cifar.data[:5, ::4, ::4, :].transpose(0, 3, 1, 2) / 255 - 0.5).float()
        targets = torch.tensor(cifar.targets[:5]).long()
        criterion = ans.modules.SoftmaxCrossEntropy()
        loss, acc = self.params['val_step_fn'](inputs, targets, self.model, criterion)
        expected_loss, expected_acc = 2.3937, 0.
        self.assertTensorsClose(loss, expected_loss)
        self.assertTensorsClose(acc, expected_acc)


class TestVGG7(ANSTestCase):

    def setUp(self) -> None:
        self.model = self.params['model_cls'](10).to(dtype=torch.float64)
        for name, param in self.model.named_parameters():
            ans.utils.seed_everything(param.data.numel())
            param.data.copy_(0.1 * torch.randn_like(param.data))

    def test_num_params(self) -> None:
        self.assertEqual(self.model.num_params(), 4178570)

    def test_forward(self) -> None:
        ans.utils.seed_everything(0)
        scores = self.model(0.1 * torch.randn(1, 3, 32, 32, dtype=torch.float64)).data
        expected_scores = torch.tensor([
            [29.9054, -66.1315,   4.1798,  88.4492,  -1.0886, -29.2324,  25.1272, -58.6127,   2.6198, -10.0502]
        ], dtype=torch.float64)
        self.assertTensorsClose(scores, expected_scores, atol=0.1)


class TestVGG7ValAccuracy(ANSTestCase):

    def test_vgg7_val_accuracy(self) -> None:
        ok = bool(int(input('VGG7 val acc > 75 % (1/0)? ')))
        self.assertTrue(ok)


class TestAugmentation(ANSTestCase):

    def test_train_dataset(self):
        for _ in range(10):
            idx = random.randint(0, len(self.params['train_dataset']) - 1)
            x, y = self.params['train_dataset'][idx]
            self.assertListEqual(list(x.shape), [3, 32, 32])
            ok = False
            for _ in range(20):
                x_, y_ = self.params['train_dataset'][idx]
                self.assertEqual(y_, y)  # targets unchanged
                self.assertListEqual(list(x_.shape), list(x.shape))  # same shape
                if not torch.allclose(x_, x):
                    ok = True  # but different values
                    break
            self.assertTrue(ok)

    def test_val_dataset(self):
        for _ in range(10):
            idx = random.randint(0, len(self.params['val_dataset']) - 1)
            x, y = self.params['val_dataset'][idx]
            self.assertIsInstance(y, int)
            self.assertListEqual(list(x.shape), [3, 32, 32])
            for _ in range(20):
                x_, y_ = self.params['val_dataset'][idx]
                self.assertEqual(y_, y)  # targets unchanged
                self.assertTensorsClose(x_, x)  # same shape and values


class TestBatchNorm2d(ANSTestCase):

    def test_implementation(self):
        self.assertNotCalling(ans.functional.BatchNorm2d.forward, ['batch_norm'])
        self.assertCalling(ans.modules.Sigmoid.forward, ['apply'])

    @staticmethod
    def random_inputs(n: int, c: int, h: int, w: int, affine: bool = True) -> tuple[tuple, dict]:
        x_var = randn_var(n, c, h, w, mean=torch.randn(1).item(), std=torch.rand(1).item())
        gamma = randn_var(c) if affine else None
        beta = randn_var(c) if affine else None
        params = {
            'running_mean': torch.randn(c, dtype=x_var.data.dtype),
            'running_var': torch.rand(c, dtype=x_var.data.dtype),
            'momentum': torch.rand(1).item(),
            'eps': 1e-5 * torch.rand(1).item()
        }
        return (x_var, gamma, beta), params

    def test_output_types(self):
        (x_var, gamma, beta), params = TestBatchNorm2d.random_inputs(10, 4, 8, 13)
        output, cache = ans.functional.BatchNorm2d.forward(x_var.data, gamma.data, beta.data, **params, training=True)
        self.assertIsInstance(output, torch.Tensor)
        self.assertIsInstance(cache, tuple)

    def test_forward_function_affine(self):
        (x_var, gamma, beta), params = TestBatchNorm2d.random_inputs(10, 4, 8, 13)
        torch_running_mean = params['running_mean'].clone()
        torch_running_var = params['running_var'].clone()

        for i, training in enumerate([True, True, False]):
            z_var = ans.functional.BatchNorm2d.apply(x_var, gamma, beta, **params, training=training)
            z = torch.nn.functional.batch_norm(x_var.data, torch_running_mean, torch_running_var, gamma.data,
                                               beta.data, training=training, momentum=1-params['momentum'],
                                               eps=params['eps'])
            self.assertTensorsClose(params['running_mean'].data, torch_running_mean, msg=f"training[{i}]={training}")
            self.assertTensorsClose(params['running_var'].data, torch_running_var, msg=f"training[{i}]={training}")
            self.assertTensorsClose(z_var.data, z, msg=f"training[{i}]={training}")

    def test_forward_function_linear(self):
        (x_var, gamma, beta), params = TestBatchNorm2d.random_inputs(10, 4, 8, 13, affine=False)
        torch_running_mean = params['running_mean'].clone()
        torch_running_var = params['running_var'].clone()

        for i, training in enumerate([True, True, False]):
            z_var = ans.functional.BatchNorm2d.apply(x_var, None, None, **params, training=training)
            z = torch.nn.functional.batch_norm(x_var.data, torch_running_mean, torch_running_var, None, None,
                                               training=training, momentum=1-params['momentum'], eps=params['eps'])
            self.assertTensorsClose(params['running_mean'].data, torch_running_mean, msg=f"training[{i}]={training}")
            self.assertTensorsClose(params['running_var'].data, torch_running_var, msg=f"training[{i}]={training}")
            self.assertTensorsClose(z_var.data, z, msg=f"training[{i}]={training}")

    def test_forward_module(self):
        momentum = torch.rand(1).item()
        eps = torch.rand(1).item()
        ans_batchnorm = ans.modules.BatchNorm2d(7, momentum=momentum, eps=eps).to(dtype=torch.float64)
        torch_batchnorm = torch.nn.BatchNorm2d(7, eps=eps, momentum=1. - momentum, dtype=torch.float64)

        self.assertTensorsClose(ans_batchnorm.gamma.data, torch_batchnorm.weight)
        self.assertTensorsClose(ans_batchnorm.beta.data, torch_batchnorm.bias)
        self.assertTensorsClose(ans_batchnorm.running_mean, torch_batchnorm.running_mean)
        self.assertTensorsClose(ans_batchnorm.running_var, torch_batchnorm.running_var)

        for i in range(3):
            x_var = randn_var(6, 7, 5, 11, mean=torch.randn(1).item(), std=torch.rand(1).item())
            z_var = ans_batchnorm(x_var)
            z = torch_batchnorm(x_var.data)

            self.assertIsInstance(z_var, ans.autograd.Variable)
            self.assertTensorsClose(z_var.data, z, msg=f"i={i}")
            self.assertTensorsClose(ans_batchnorm.running_mean, torch_batchnorm.running_mean)
            self.assertTensorsClose(ans_batchnorm.running_var, torch_batchnorm.running_var)

    def test_gradcheck(self):
        for training in [True, False]:
            (x_var, gamma, beta), params = TestBatchNorm2d.random_inputs(11, 4, 5, 7)
            gradcheck_result = ans.autograd.gradcheck(
                ans.functional.BatchNorm2d.apply,
                (x_var, gamma, beta),
                params=dict(**params, training=training),
                verbose=False
            )
            self.assertTrue(gradcheck_result, msg=f"training={training}")


class TestVGG7BN(TestVGG7):

    def test_num_params(self) -> None:
        self.assertEqual(self.model.num_params(), 4180042)

    def test_forward(self) -> None:
        ans.utils.seed_everything(0)
        scores = self.model(0.1 * torch.randn(1, 3, 32, 32, dtype=torch.float64)).data
        expected_scores = torch.tensor([
            [0.0439, -1.1975, -0.3539,  0.2358,  0.0134, -0.8286,  0.6840,  0.2537, -0.3125,  0.3412]
        ], dtype=torch.float64)
        self.assertTensorsClose(scores, expected_scores, atol=0.1)


class TestResnet9(TestVGG7):

    def test_num_params(self) -> None:
        self.assertEqual(self.model.num_params(), 6573130)

    def test_forward(self) -> None:
        ans.utils.seed_everything(0)
        scores = self.model(0.1 * torch.randn(1, 3, 32, 32, dtype=torch.float64)).data
        expected_scores = torch.tensor([
            [-0.0288, -1.5824,  0.3436, -0.2640, -0.6936,  0.0891,  0.7058, -1.2480, -2.1264, -0.7670]
        ], dtype=torch.float64)
        self.assertTensorsClose(scores, expected_scores, atol=0.1)


class TestResnet9ValAccuracy(ANSTestCase):

    def test_resnet9_val_accuracy(self) -> None:
        ok = bool(int(input('Resnet9 val acc > 75 % (1/0)? ')))
        self.assertTrue(ok)
