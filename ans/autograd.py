import inspect
import re
from typing import Any, Callable, Optional, Union

import graphviz
import torch


class Variable:

    def __init__(
            self,
            data: torch.Tensor,
            parents: tuple['Variable', ...] = (),
            grad_fn: Optional[Callable[[torch.Tensor], tuple[torch.Tensor, ...]]] = None,
            name: Optional[str] = None
    ) -> None:
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data)
        self.data = data
        self.grad: Optional[torch.Tensor] = None
        self.parents = parents
        self.grad_fn = grad_fn
        self.name = name

    def __repr__(self):
        if hasattr(self.grad_fn, 'func'):
            grad_fn_repr = self.grad_fn.func.__qualname__
        elif self.grad_fn is not None:
            grad_fn_repr = self.grad_fn.__qualname__
        else:
            grad_fn_repr = 'None'
        if self.data.ndim > 0:  # tensor
            return f"{self.__class__.__name__}(shape={tuple(self.data.shape)}, grad_fn={grad_fn_repr})"
        else:  # scalar
            return f"{self.__class__.__name__}({self.data.item()}, grad_fn={grad_fn_repr})"

    def __add__(self, other: 'Variable') -> 'Variable':
        ########################################
        # TODO: implement

        import numpy as np

        def grad_fn(dout: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return dout.clone(), dout.clone()

        result = Variable(
            self.data + other.data,
            parents=(self, other),
            grad_fn=grad_fn
        )

        return result

        # ENDTODO
        ########################################
    
    def __radd__(self, other: 'Variable') -> 'Variable':
        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
        ########################################

    def __sub__(self, other: 'Variable') -> 'Variable':
        ########################################
        # TODO: implement

        def grad_fn(dout: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return dout.clone(), -dout.clone()

        result = Variable(
            self.data - other.data,
            parents=(self, other),
            grad_fn=grad_fn
        )

        return result

        # ENDTODO
        ########################################
    
    def __rsub__(self, other: 'Variable') -> 'Variable':
        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
        ########################################

    def __mul__(self, other: 'Variable') -> 'Variable':
        ########################################
        # TODO: implement

        def grad_fn(dout: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return dout * other.data, dout * self.data

        result = Variable(
            self.data * other.data,
            parents=(self, other),
            grad_fn=grad_fn
        )

        return result

        # ENDTODO
        ########################################
    
    def __rmul__(self, other: 'Variable') -> 'Variable':
        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
        ########################################

    def __pow__(self, power, modulo=None):
        ########################################
        # TODO: implement

        def grad_fn(dout: torch.Tensor) -> tuple[torch.Tensor]:
            grad = dout * power * self.data**(power - 1)
            return grad,

        result = Variable(
            self.data**power,
            parents=(self,),
            grad_fn=grad_fn
        )

        return result

        # ENDTODO
        ########################################
    
    def __matmul__(self, other: 'Variable') -> 'Variable':
        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
        ########################################
    
    def __rmatmul__(self, other: 'Variable') -> 'Variable':
        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
        ########################################

    def __getitem__(self, item) -> 'Variable':
        def grad_fn(dout: torch.Tensor) -> tuple[torch.Tensor]:
            dinput = torch.zeros_like(self.data)
            dinput[item] = dout
            return dinput,
        return Variable(
            self.data[item],
            parents=(self,),
            grad_fn=grad_fn
        )

    @property
    def shape(self) -> torch.Size:
        return self.data.shape

    def reshape(self, *shape: int) -> 'Variable':
        def grad_fn(dout: torch.Tensor) -> tuple[torch.Tensor]:
            return dout.reshape(*self.data.shape),
        grad_fn.name = 'reshape'
        return Variable(
            self.data.reshape(*shape).clone(),
            parents=(self,),
            grad_fn=grad_fn
        )

    def predecessors(self) -> list['Variable']:
        """

        Returns:
            predecessors: Topologically sorted list of node's predecessors including self.
        """

        predecessors = []

        ########################################
        # TODO: implement

        def visit(node: 'Variable', visited: set):
            if node not in visited:
                visited.add(node)
                for parent in node.parents:
                    visit(parent, visited)
                predecessors.append(node)

        visit(self, set())

        # ENDTODO
        ########################################

        return predecessors

    def backprop(self, dout: Optional[torch.Tensor] = None) -> None:
        """
        Runs full backpropagation starting from self. Fills the grad attribute with dself/dpredecessor for all
        predecessors of self.

        Args:
            dout: Incoming gradient on self; if None, then set to tensor of ones with proper shape and dtype

        """

        ########################################
        # TODO: implement

        if dout is None:
            dout = torch.ones_like(self.data)

        self.grad = dout.clone()
        for node in reversed(self.predecessors()):
            if node.grad_fn is not None:
                grads = node.grad_fn(node.grad)

                for i, parent in enumerate(node.parents):
                    if parent.grad is None:
                        parent.grad = grads[i].clone()
                    else:
                        parent.grad += grads[i].clone()

        # ENDTODO
        ########################################

    def to_graphviz(self, show_data: bool = False) -> graphviz.Digraph:
        def get_node_info(
                node: 'Variable',
                default_name: str = '',
                show_data: bool = False,
                stack_offset: int = 2
        ) -> tuple[str, str]:
            """
            Auxiliary function for graphviz Digraph creation to display a computation graph history of some Variable.
            Function automatically determines unique identifier and label for any node that is represented by a Variable object.

            Args:
                node: Variable object
                default_name: Name that will be displayed if automatic name determination from Variable inspection fails
                show_data: Whether to include Variable's data and grad as string in returned label
                stack_offset: Magic argument to inspect. If name determination fails, try setting this to 1 or 3.

            Returns:
                node_uid: Unique identifier of node
                node_label: Text caption of the node to be displayed as label in some graphviz.Digraph
            """
            node_uid = str(id(node))
            if node.name is not None:
                return node_uid, node.name
            try:
                namespace = dict(inspect.getmembers(inspect.stack()[stack_offset][0]))["f_globals"]
                node_name = next(k for k, v in namespace.items() if v is node)
            except StopIteration:
                node_name = default_name
            if node.data.ndim > 0:  # tensor
                node_label = f"{node_name} {tuple(node.data.shape)}"
                if show_data:
                    node_label += f" | data: {node.data} | grad: {node.grad}"
            else:  # scalar
                node_label = f"{node_name} = {node.data.item():.3f}"
                if show_data and node.grad is not None:
                    node_label += f" | grad = {node.grad.item():.3f}"
            return node_uid, node_label

        def get_func_info(node:'Variable') -> tuple[str, str]:
            """
            The function is analogous to get_node_info, but used for graph nodes that represent operations such as summation or
            multiplication. Automatically determines unique identifier and display label.
            It will check input node's grad_fn attribute and infer the operation name of which the node is a result.
            Args:
                node: Variable object

            Returns:
                func_uid: Unique identifier of node
                func_label: Text caption of the node to be displayed as label in some graphviz.Digraph
            """
            func_uid = str(id(node)) + '.grad_fn'
            if hasattr(node.grad_fn, 'name'):
                func_label = node.grad_fn.name.removesuffix('.backward')  # coming from Function.apply
            else:
                func_label = re.search('__(\w+)__', node.grad_fn.__qualname__).group(1)  # coming from Variable.__*__
            return func_uid, func_label

        dot = graphviz.Digraph(graph_attr={'rankdir': 'LR'})

        ########################################
        # TODO: implement

        visited = set()
        for node in self.predecessors():
            node_uid, node_label = get_node_info(node, show_data=show_data)
            dot.node(node_uid, label=node_label, shape='record')
            if node.grad_fn is not None:
                func_uid, func_label = get_func_info(node)
                dot.node(func_uid, label=func_label, shape='ellipse')
                dot.edge(node_uid, func_uid)

        # ENDTODO
        ########################################

        return dot


def numeric_gradient(
        func: Callable[[torch.Tensor], torch.Tensor],
        input: torch.Tensor,
        dout: torch.Tensor,
        eps: Optional[float] = None
) -> torch.Tensor:
    if eps is None:
        eps = max(1e-6, 0.01 * input.std())
    if not torch.is_floating_point(input):
        raise TypeError(input.dtype)
    input = input.contiguous()
    flat = input.view(-1)  # shares memory
    grad = torch.zeros_like(flat)
    for i in range(flat.shape[0]):
        inp_i = flat[i].item()  # must be deep-copied, since `flat[i]` returns a view to the i-th element
        flat[i] = inp_i + eps
        pos = func(input)
        flat[i] = inp_i - eps
        neg = func(input)
        flat[i] = inp_i  # restore
        grad[i] = torch.sum((pos - neg) * dout) / (2. * eps)
    return grad.reshape(input.shape)


def gradcheck(
        func: Callable[[Any, ...], Variable],
        inputs: tuple[Union[torch.Tensor, Variable], ...],
        params: Optional[dict[str, Any]] = None,
        dout: Optional[torch.Tensor] = None,
        eps: Optional[float] = None,
        rtol: float = 1e-3,
        atol: float = 1e-5,
        verbose: bool = True
) -> bool:
    if params is None:
        params = {}

    # Analytic gradients
    output = func(*inputs, **params)
    for input in inputs:
        if isinstance(input, Variable):
            input.grad = None
    if dout is None:
        dout = torch.randn_like(output.data)
    output.backprop(dout=dout)
    grads_ana = [i.grad for i in inputs if isinstance(i, Variable)]

    # Numeric gradients
    ok = True
    for i, input in enumerate(inputs):
        if not isinstance(input, Variable):
            continue
        grad_num = numeric_gradient(lambda _: func(*inputs, **params).data, input.data, dout, eps=eps)
        abs_err = torch.abs(grads_ana[i] - grad_num)
        rel_err = abs_err / torch.abs(grad_num + 1e-6)
        allowed = atol + rtol * torch.abs(grad_num)
        passes = torch.all(abs_err < allowed)  # torch.allclose
        ok = ok and passes.item()
        if verbose:
            input_name = input.name if input.name is not None else f"input{i}"
            if passes:
                print(f"d{input_name} ok: rel_err={rel_err.max().item():.3e}, abs_err={abs_err.max().item():.3e}")
            else:
                print(f"\033[31md{input_name} FAIL: "  # begin red color
                      f"rel_err={rel_err.max().item():.3e}, abs_err={abs_err.max().item():.3e}\n")
                print('Analytic gradient: ')
                print(grads_ana[i], '\n')
                print('Numeric gradient: ')
                print(grad_num, '\n')
                print('Relative error: ')
                print(rel_err)
                print('\033[30m')  # end red color

    return ok
