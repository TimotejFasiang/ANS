import torch

import ans
from tests import ANSTestCase


class TestBatchLoader(ANSTestCase):

    def setUp(self) -> None:
        self.x = torch.tile(torch.arange(10)[:, None], (1, 2))
        self.y = torch.arange(10)

    def test_output_tuple_unsupervised(self):
        loader = ans.data.BatchLoader(self.x.clone())
        batch = next(iter(loader))
        self.assertIsInstance(batch, tuple)
        self.assertEqual(len(batch), 1)

    def test_output_tuple_supervised(self):
        loader = ans.data.BatchLoader(self.x.clone(), self.y.clone())
        batch = next(iter(loader))
        self.assertIsInstance(batch, tuple)
        self.assertEqual(len(batch), 2)

    def test_defaults(self) -> None:
        loader = ans.data.BatchLoader(self.x.clone(), self.y.clone())
        xb, yb = next(iter(loader))
        self.assertTupleEqual(xb.shape, self.x.shape)
        self.assertTupleEqual(yb.shape, self.y.shape)
        self.assertTensorsClose(xb, self.x)
        self.assertTensorsClose(yb, self.y)

    def test_batch_size_even(self) -> None:
        batch_size = 2
        loader = ans.data.BatchLoader(self.x.clone(), self.y.clone(), batch_size=batch_size)
        batches = [(xb, yb) for xb, yb in loader]
        self.assertEqual(len(batches), len(self.x) // batch_size)
        self.assertEqual(sum(len(xb) for xb, yb in batches), len(self.x))
        expected_batch_sizes = [batch_size] * (len(self.x) // batch_size)
        self.assertListEqual([len(xb) for xb, yb in batches], expected_batch_sizes)
        self.assertListEqual([len(yb) for xb, yb in batches], expected_batch_sizes)

    def test_batch_size_uneven(self) -> None:
        batch_size = 3
        loader = ans.data.BatchLoader(self.x.clone(), self.y.clone(), batch_size=batch_size)
        batches = [(xb, yb) for xb, yb in loader]
        self.assertEqual(len(batches), -(-len(self.x) // batch_size))  # -(-a//b)) ... ceil
        self.assertEqual(sum(len(xb) for xb, yb in batches), len(self.x))
        expected_batch_sizes = [batch_size] * (len(self.x) // batch_size)
        if sum(expected_batch_sizes) < len(self.x):
            expected_batch_sizes.append(len(self.x) - sum(expected_batch_sizes))
        self.assertListEqual([len(xb) for xb, yb in batches], expected_batch_sizes)
        self.assertListEqual([len(yb) for xb, yb in batches], expected_batch_sizes)

    def test_shuffle(self):
        ans.utils.seed_everything(0)
        loader = ans.data.BatchLoader(self.x.clone(), self.y.clone(), shuffle=True)
        xb, yb = next(iter(loader))
        self.assertFalse(torch.all(xb == self.x))
        self.assertFalse(torch.all(yb == self.y))
        self.assertTensorsClose(xb, self.x[xb[:, 0]])
        self.assertTensorsClose(yb, self.y[xb[:, 0]])


class TestPreprocess(ANSTestCase):

    def test_preprocess(self):
        x = self.params['preprocess_fn'](torch.arange(2 * 3 * 2 * 2).reshape(2, 3, 2, 2))
        expected_x = torch.tensor([
            [0.0000, 0.0039, 0.0078, 0.0118, 0.0157, 0.0196, 0.0235, 0.0275, 0.0314, 0.0353, 0.0392, 0.0431],
            [0.0471, 0.0510, 0.0549, 0.0588, 0.0627, 0.0667, 0.0706, 0.0745, 0.0784, 0.0824, 0.0863, 0.0902]
        ])
        self.assertTensorsClose(x, expected_x)


class TestInit(ANSTestCase):

    def test_init_params(self):
        ans.utils.seed_everything(0)
        expected_weight = torch.tensor([
            [ 0.0154, -0.0029],
            [-0.0218,  0.0057],
            [-0.0108, -0.0140]
        ])
        expected_bias = torch.tensor([0., 0.])
        weight, bias = self.params['init_params_fn'](3, 2, multiplier=1e-2)
        self.assertTensorsClose(weight, expected_weight)
        self.assertTensorsClose(bias.squeeze(), expected_bias)
        if len(bias.shape) > 1:
            self.assertLessEqual(len(bias.shape), 2)
            self.assertGreaterEqual(bias.shape[1], bias.shape[0])


class TestCalcLinearScores(ANSTestCase):

    def test_implementation(self):
        self.assertNotCalling(self.params['calc_linear_scores_fn'], ['linear'])
        self.assertNoLoops(self.params['calc_linear_scores_fn'])

    def test_calc_linear_scores(self):
        x, w, b = torch.randn(5, 3), torch.randn(3, 2), torch.randn(2)
        scores = self.params['calc_linear_scores_fn'](x, w, b)
        expected_scores = torch.nn.functional.linear(x, w.t(), b)
        self.assertTensorsClose(scores, expected_scores)


class TestSoftmaxCrossEntropy(ANSTestCase):

    def test_implementation(self):
        self.assertNotCalling(self.params['softmax_cross_entropy_fn'], ['cross_entropy', 'nll_loss'])
        self.assertNoLoops(self.params['softmax_cross_entropy_fn'])

    def test_softmax_cross_entropy(self):
        ans.utils.seed_everything(0)
        probs = torch.randn(4, 8)
        targets = torch.randint(0, 8, (4, 1)).squeeze()
        loss = self.params['softmax_cross_entropy_fn'](probs, targets)
        expected_loss = torch.nn.functional.cross_entropy(probs, targets, reduction='mean')
        self.assertTensorsClose(loss, expected_loss)


class TestAccuracy(ANSTestCase):

    def test_implementation(self):
        self.assertNotCalling(self.params['accuracy_fn'], ['accuracy'])
        self.assertNoLoops(self.params['accuracy_fn'])

    def test_accuracy(self):
        ans.utils.seed_everything(0)
        probs = torch.rand(19, 5)
        targets = torch.randint(0, 5, (19, 1)).squeeze()
        acc = self.params['accuracy_fn'](probs, targets)
        expected_acc = torch.tensor(0.2105)
        self.assertTensorsClose(acc, expected_acc)


class TestSoftmaxCrossEntropyGradients(ANSTestCase):

    def test_implementation(self):
        self.assertNoLoops(self.params['softmax_cross_entropy_gradients_fn'])

    def test_softmax_cross_entropy_gradients(self):
        inputs = torch.randn(5, 4)
        targets = torch.tensor([1, 1, 0, 2, 0])
        weight = torch.randn(4, 3, requires_grad=True)
        bias = torch.randn(3, requires_grad=True)
        logits = torch.nn.functional.linear(inputs, weight.t(), bias=bias)
        loss = torch.nn.functional.cross_entropy(logits, targets)
        loss.backward()
        expected_dweight = weight.grad
        expected_dbias = bias.grad
        dweight, dbias = self.params['softmax_cross_entropy_gradients_fn'](inputs, logits, targets)
        self.assertTensorsClose(dweight, expected_dweight)
        self.assertTensorsClose(dbias, expected_dbias)


class TestUpdateParamInplace(ANSTestCase):

    def test_implementation(self):
        self.assertNoLoops(self.params['update_param_inplace_fn'])

    def test_update_param_inplace(self):
        ans.utils.seed_everything(0)
        weight = torch.randn(5, 3)
        dweight = torch.randn(*weight.shape)
        lr = 1e-2 * torch.rand(1).item()
        self.params['update_param_inplace_fn'](weight, dweight, lr)
        expected_weight = torch.tensor([
            [ 1.5401, -0.2893, -2.1815],
            [ 0.5749, -1.0736, -1.4013],
            [ 0.3928,  0.8312, -0.7249],
            [-0.4047, -0.6058,  0.1914],
            [-0.8509,  1.1008, -1.0660]
        ])
        self.assertTensorsClose(weight, expected_weight)


class TestTrainStepSoftmax(ANSTestCase):

    def test_implementation(self):
        self.assertCalling(
            self.params['train_step_softmax_fn'],
            ['preprocess', 'calc_linear_scores', 'softmax_cross_entropy', 'accuracy', 'softmax_cross_entropy_gradients',
             'update_param_inplace']
        )
        self.assertNoLoops(self.params['train_step_softmax_fn'])

    def test_train_step_softmax(self):
        ans.utils.seed_everything(0)
        inputs = torch.randn(8, 5)
        targets = torch.randint(0, 3, (8, 1)).squeeze()
        weight = torch.randn(5, 3)
        bias = torch.randn(3)
        learning_rate = 1e-2 * torch.rand(1).item()
        loss, acc = self.params['train_step_softmax_fn'](inputs, targets, weight, bias, learning_rate=learning_rate)
        expected_loss, expected_acc = 1.5571, 0.375
        expected_weight = torch.tensor([
            [0.1920, 0.5428, -2.2188],
            [0.2590, -1.0297, -0.5008],
            [0.2734, -0.9181, -0.0404],
            [0.2881, -0.0075, -0.9145],
            [-1.0886, -0.2666, 0.1894]
        ])
        expected_bias = torch.tensor([-0.2174,  2.0533, -0.0328])
        self.assertTensorsClose(loss, expected_loss)
        self.assertTensorsClose(acc, expected_acc)
        self.assertTensorsClose(weight, expected_weight)
        self.assertTensorsClose(bias, expected_bias)


class TestValStep(ANSTestCase):

    def test_implementation(self):
        self.assertCalling(
            self.params['val_step_fn'],
            ['preprocess', 'calc_linear_scores', 'loss_fn', 'accuracy']
        )
        self.assertNotCalling(self.params['val_step_fn'], ['softmax_cross_entropy_gradients', 'update_param_inplace'])
        self.assertNoLoops(self.params['val_step_fn'])

    def test_val_step(self):
        ans.utils.seed_everything(0)
        inputs = torch.randn(8, 5)
        targets = torch.randint(0, 3, (8, 1)).squeeze()
        weight = torch.randn(5, 3)
        bias = torch.randn(3)
        loss, acc = self.params['val_step_fn'](inputs, targets, weight, bias, self.params['loss_fn'])
        expected_loss, expected_acc = 1.5571, 0.375
        self.assertTensorsClose(loss, expected_loss)
        self.assertTensorsClose(acc, expected_acc)


class TestSoftmaxValAccuracy(ANSTestCase):

    def test_softmax_val_acc(self):
        val_acc = float(input('softmax val acc: '))
        self.assertGreaterEqual(val_acc, 0.4)


class TestHingeLoss(ANSTestCase):

    def test_implementation(self):
        self.assertNotCalling(self.params['hinge_loss_fn'], ['multi_margin_loss'])

    def test_no_loops(self):
        self.assertNoLoops(self.params['hinge_loss_fn'])

    def test_hinge_loss(self):
        scores = torch.randn(8, 5)
        targets = torch.randint(0, 5, (8, 1)).squeeze()
        loss = self.params['hinge_loss_fn'](scores, targets)
        expected_loss = scores.shape[1] * torch.nn.functional.multi_margin_loss(scores, targets, reduction='mean')
        self.assertTensorsClose(loss, expected_loss)


class TestHingeLossGradients(ANSTestCase):

    def test_hinge_loss_gradients(self):
        inputs = torch.randn(8, 5)
        targets = torch.randint(0, 3, (8, 1)).squeeze()
        weight = torch.randn(5, 3, requires_grad=True)
        bias = torch.randn(3, requires_grad=True)
        scores = torch.nn.functional.linear(inputs, weight.t(), bias)
        loss = scores.shape[1] * torch.nn.functional.multi_margin_loss(scores, targets, reduction='mean')
        loss.backward()
        dweight, dbias = self.params['hinge_loss_gradients_fn'](inputs, scores, targets)
        expected_dweight, expected_dbias = weight.grad, bias.grad
        self.assertTensorsClose(dweight, expected_dweight)
        self.assertTensorsClose(dbias, expected_dbias)


class TestTrainStepSVM(ANSTestCase):

    def test_implementation(self):
        self.assertCalling(
            self.params['train_step_svm_fn'],
            ['preprocess', 'calc_linear_scores', 'hinge_loss', 'accuracy', 'hinge_loss_gradients',
             'update_param_inplace']
        )
        self.assertNoLoops(self.params['train_step_svm_fn'])

    def test_train_step_svm(self):
        ans.utils.seed_everything(0)
        inputs = torch.randn(8, 5)
        targets = torch.randint(0, 3, (8, 1)).squeeze()
        weight = torch.randn(5, 3)
        bias = torch.randn(3)
        learning_rate = 1e-2 * torch.rand(1).item()
        loss, acc = self.params['train_step_svm_fn'](inputs, targets, weight, bias, learning_rate=learning_rate)
        expected_loss, expected_acc = 2.57787, 0.375
        expected_weight = torch.tensor([
            [ 0.1920,  0.5428, -2.2188],
            [ 0.2590, -1.0297, -0.5008],
            [ 0.2734, -0.9181, -0.0404],
            [ 0.2881, -0.0075, -0.9145],
            [-1.0886, -0.2666,  0.1894]
        ])
        expected_bias = torch.tensor([-0.2178,  2.0515, -0.0306])
        self.assertTensorsClose(loss, expected_loss)
        self.assertTensorsClose(acc, expected_acc)
        self.assertTensorsClose(weight, expected_weight)
        self.assertTensorsClose(bias, expected_bias)


class TestSVMValAccuracy(ANSTestCase):

    def test_softmax_val_acc(self):
        val_acc = float(input('svm val acc: '))
        self.assertGreaterEqual(val_acc, 0.35)
