#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import unittest
import numpy as np
from op_test import OpTest, skip_check_grad_ci, convert_float_to_uint16
import paddle
import paddle.fluid.core as core
import paddle.fluid as fluid
from paddle.fluid import compiler, Program, program_guard
from paddle.fluid.framework import convert_np_dtype_to_dtype_


def get_places(self):
    return [paddle.CustomPlace('intel_gpu', 0)]


OpTest._get_places = get_places

@skip_check_grad_ci("")
class TestSumOp(OpTest):
    def setUp(self):
        self.python_api = paddle.sum
        self.op_type = "reduce_sum"
        self.inputs = {'X': np.random.random((5, 6, 10)).astype("float32")}
        self.outputs = {'Out': self.inputs['X'].sum(axis=0)}
        self.attrs = {'dim': [0]}

    def test_check_output(self):
        self.check_output(check_eager=False)

    # def test_check_grad(self):
    #     self.check_grad(['X'], 'Out', check_eager=False)


# class TestSumOp_fp16(OpTest):
#     def setUp(self):
#         self.python_api = paddle.sum
#         self.op_type = "reduce_sum"
#         self.inputs = {
#             'X': np.random.uniform(0, 0.1, (5, 6, 10)).astype("float16")
#         }
#         self.attrs = {'dim': [0, 1, 2]}
#         self.outputs = {
#             'Out': self.inputs['X'].sum(axis=tuple(self.attrs['dim']))
#         }
#         self.gradient = self.calc_gradient()

#     def test_check_output(self):
#         self.check_output(check_eager=False)

#     def calc_gradient(self):
#         x = self.inputs["X"]
#         grad = np.ones(x.shape, dtype=x.dtype)
#         return grad,

#     def test_check_grad(self):
#         self.check_grad(
#             ['X'], 'Out', user_defined_grads=self.gradient, check_eager=False)

# @unittest.skipIf(not core.is_compiled_with_cuda(),
#                  "core is not compiled with CUDA")
# class TestSumOp_bf16(OpTest):
#     def setUp(self):
#         np.random.seed(100)
#         self.python_api = paddle.sum
#         self.op_type = "reduce_sum"
#         self.dtype = np.uint16
#         self.x = np.random.uniform(0, 0.1, (2, 5, 10)).astype(np.float32)
#         self.attrs = {'dim': [0, 1, 2]}
#         self.out = self.x.sum(axis=tuple(self.attrs['dim']))
#         self.gradient = self.calc_gradient()

#         self.inputs = {'X': convert_float_to_uint16(self.x)}
#         self.outputs = {'Out': convert_float_to_uint16(self.out)}
#         self.gradient = self.calc_gradient()

#     def test_check_output(self):
#         place = core.CustomPlace('intel_gpu', 0)
#         self.check_output_with_place(place, check_eager=False)

#     def test_check_grad(self):
#         place = core.CustomPlace('intel_gpu', 0)
#         self.check_grad_with_place(
#             place, ['X'],
#             'Out',
#             user_defined_grads=self.gradient,
#             check_eager=False)

#     def calc_gradient(self):
#         x = self.x
#         grad = np.ones(x.shape, dtype=x.dtype)
#         return [grad]

# class TestSumOp_fp16_withInt(OpTest):
#     def setUp(self):
#         self.python_api = paddle.sum
#         self.op_type = "reduce_sum"
#         self.inputs = {
#             # ref to https://en.wikipedia.org/wiki/Half-precision_floating-point_format
#             # Precision limitations on integer values between 0 and 2048 can be exactly represented
#             'X': np.random.randint(0, 30, (10, 10)).astype("float16")
#         }
#         self.attrs = {'dim': [0, 1]}
#         self.outputs = {
#             'Out': self.inputs['X'].sum(axis=tuple(self.attrs['dim']))
#         }
#         self.gradient = self.calc_gradient()

#     def test_check_output(self):
#         self.check_output(check_eager=False)

#     def calc_gradient(self):
#         x = self.inputs["X"]
#         grad = np.ones(x.shape, dtype=x.dtype)
#         return grad,

#     def test_check_grad(self):
#         self.check_grad(
#             ['X'], 'Out', user_defined_grads=self.gradient, check_eager=False)

@skip_check_grad_ci("")
class TestSumOp4D(OpTest):
    def setUp(self):
        self.python_api = paddle.sum
        self.op_type = "reduce_sum"
        self.inputs = {
            'X': np.random.random((1, 5, 6, 10)).astype("float32")
        }
        self.attrs = {'dim': [0]}
        self.outputs = {'Out': self.inputs['X'].sum(axis=0)}

    def test_check_output(self):
        self.check_output(check_eager=False)

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', check_eager=False)

@skip_check_grad_ci("")
class TestSumOp5D(OpTest):
    def setUp(self):
        self.python_api = paddle.sum
        self.op_type = "reduce_sum"
        self.inputs = {
            'X': np.random.random((1, 2, 5, 6, 10)).astype("float32")
        }
        self.attrs = {'dim': [0]}
        self.outputs = {'Out': self.inputs['X'].sum(axis=0)}

    def test_check_output(self):
        self.check_output(check_eager=False)

    # def test_check_grad(self):
    #     self.check_grad(['X'], 'Out', check_eager=False)

@skip_check_grad_ci("")
class TestSumOp6D(OpTest):
    def setUp(self):
        self.python_api = paddle.sum
        self.op_type = "reduce_sum"
        self.inputs = {
            'X': np.random.random((1, 1, 2, 5, 6, 10)).astype("float32")
        }
        self.attrs = {'dim': [0]}
        self.outputs = {'Out': self.inputs['X'].sum(axis=0)}

    def test_check_output(self):
        self.check_output(check_eager=False)

    # def test_check_grad(self):
    #     self.check_grad(['X'], 'Out', check_eager=False)

@skip_check_grad_ci("")
class TestSumOp6DRed(OpTest):
    def setUp(self):
        self.python_api = paddle.sum
        self.op_type = "reduce_sum"
        self.inputs = {
            'X': np.random.random((1, 3, 1, 2, 1, 4)).astype("float32")
        }
        self.attrs = {'dim': (0, 3)}
        self.outputs = {'Out': self.inputs['X'].sum(axis=(0, 3))}

    def test_check_output(self):
        self.check_output(check_eager=False)

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', check_eager=False)


@skip_check_grad_ci(
    reason="reduce_max is discontinuous non-derivable function,"
    " its gradient check is not supported by unittest framework.")
class TestMaxOp(OpTest):
    """Remove Max with subgradient from gradient check to confirm the success of CI."""

    def setUp(self):
        self.op_type = "reduce_max"
        self.python_api = paddle.max
        self.inputs = {'X': np.random.random((5, 6, 10)).astype("float32")}
        self.attrs = {'dim': [-1]}
        self.outputs = {
            'Out': self.inputs['X'].max(axis=tuple(self.attrs['dim']))
        }

    def test_check_output(self):
        self.check_output(check_eager=False)


@skip_check_grad_ci(
    reason="reduce_min is discontinuous non-derivable function,"
    " its gradient check is not supported by unittest framework.")
class TestMinOp(OpTest):
    """Remove Min with subgradient from gradient check to confirm the success of CI."""

    def setUp(self):
        self.op_type = "reduce_min"
        self.python_api = paddle.min
        self.inputs = {'X': np.random.random((5, 6, 10)).astype("float32")}
        self.attrs = {'dim': [2]}
        self.outputs = {
            'Out': self.inputs['X'].min(axis=tuple(self.attrs['dim']))
        }

    def test_check_output(self):
        self.check_output(check_eager=False)


@skip_check_grad_ci("")
class TestMin6DOp(OpTest):
    """Remove Min with subgradient from gradient check to confirm the success of CI."""

    def setUp(self):
        self.op_type = "reduce_min"
        self.python_api = paddle.min
        self.inputs = {
            'X': np.random.random((2, 4, 3, 5, 6, 10)).astype("float32")
        }
        self.attrs = {'dim': [2, 4]}
        self.outputs = {
            'Out': self.inputs['X'].min(axis=tuple(self.attrs['dim']))
        }

    def test_check_output(self):
        self.check_output(check_eager=False)


# @skip_check_grad_ci("")
# class TestMin8DOp(OpTest):
#     """Remove Min with subgradient from gradient check to confirm the success of CI."""

#     def setUp(self):
#         self.op_type = "reduce_min"
#         self.python_api = paddle.min
#         self.inputs = {
#             'X': np.random.random((2, 4, 3, 5, 6, 3, 2, 4)).astype("float32")
#         }
#         self.attrs = {'dim': [2, 3, 4]}
#         self.outputs = {
#             'Out': self.inputs['X'].min(axis=tuple(self.attrs['dim']))
#         }

#     def test_check_output(self):
#         self.check_output(check_eager=False)


# class TestAllOp(OpTest):
#     def setUp(self):
#         self.op_type = "reduce_all"
#         self.python_api = paddle.all
#         self.inputs = {'X': np.random.randint(0, 2, (5, 6, 10)).astype("bool")}
#         self.outputs = {'Out': self.inputs['X'].all()}
#         self.attrs = {'reduce_all': True}

#     def test_check_output(self):
#         self.check_output(check_eager=False)

# class TestAll8DOp(OpTest):
#     def setUp(self):
#         self.op_type = "reduce_all"
#         self.python_api = paddle.all
#         self.inputs = {
#             'X': np.random.randint(0, 2,
#                                    (2, 5, 3, 2, 2, 3, 4, 2)).astype("bool")
#         }
#         self.attrs = {'reduce_all': True, 'dim': (2, 3, 4)}
#         self.outputs = {'Out': self.inputs['X'].all(axis=self.attrs['dim'])}

#     def test_check_output(self):
#         self.check_output(check_eager=False)

# class TestAllOpWithDim(OpTest):
#     def setUp(self):
#         self.op_type = "reduce_all"
#         self.python_api = paddle.all
#         self.inputs = {'X': np.random.randint(0, 2, (5, 6, 10)).astype("bool")}
#         self.attrs = {'dim': (1, )}
#         self.outputs = {'Out': self.inputs['X'].all(axis=self.attrs['dim'])}

#     def test_check_output(self):
#         self.check_output(check_eager=False)

# class TestAll8DOpWithDim(OpTest):
#     def setUp(self):
#         self.op_type = "reduce_all"
#         self.python_api = paddle.all
#         self.inputs = {
#             'X': np.random.randint(0, 2,
#                                    (2, 5, 3, 2, 2, 3, 4, 2)).astype("bool")
#         }
#         self.attrs = {'dim': (1, 3, 4)}
#         self.outputs = {'Out': self.inputs['X'].all(axis=self.attrs['dim'])}

#     def test_check_output(self):
#         self.check_output(check_eager=False)

# class TestAllOpWithKeepDim(OpTest):
#     def setUp(self):
#         self.op_type = "reduce_all"
#         self.python_api = paddle.all
#         self.inputs = {'X': np.random.randint(0, 2, (5, 6, 10)).astype("bool")}
#         self.attrs = {'dim': [1], 'keep_dim': True}
#         self.outputs = {
#             'Out': np.expand_dims(
#                 self.inputs['X'].all(axis=1), axis=1)
#         }

#     def test_check_output(self):
#         self.check_output(check_eager=False)

# class TestAll8DOpWithKeepDim(OpTest):
#     def setUp(self):
#         self.op_type = "reduce_all"
#         self.python_api = paddle.all
#         self.inputs = {
#             'X': np.random.randint(0, 2,
#                                    (2, 5, 3, 2, 2, 3, 4, 2)).astype("bool")
#         }
#         self.attrs = {'dim': (5, ), 'keep_dim': True}
#         self.outputs = {
#             'Out': np.expand_dims(
#                 self.inputs['X'].all(axis=self.attrs['dim']), axis=5)
#         }

#     def test_check_output(self):
#         self.check_output(check_eager=False)

class TestAllOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):
            # The input type of reduce_all_op must be Variable.
            input1 = 12
            self.assertRaises(TypeError, fluid.layers.reduce_all, input1)
            # The input dtype of reduce_all_op must be bool.
            input2 = fluid.layers.data(
                name='input2', shape=[12, 10], dtype="int32")
            self.assertRaises(TypeError, fluid.layers.reduce_all, input2)

@skip_check_grad_ci("")
class Test1DReduce(OpTest):
    def setUp(self):
        self.op_type = "reduce_sum"
        self.inputs = {'X': np.random.random(120).astype("float32")}
        self.outputs = {'Out': self.inputs['X'].sum(axis=0)}

    def test_check_output(self):
        self.check_output()

    # def test_check_grad(self):
    #     self.check_grad(['X'], 'Out')


@skip_check_grad_ci("")
class Test2DReduce0(Test1DReduce):
    def setUp(self):
        self.op_type = "reduce_sum"
        self.attrs = {'dim': [0]}
        self.inputs = {'X': np.random.random((20, 10)).astype("float32")}
        self.outputs = {'Out': self.inputs['X'].sum(axis=0)}


@skip_check_grad_ci("")
class Test2DReduce1(Test1DReduce):
    def setUp(self):
        self.op_type = "reduce_sum"
        self.attrs = {'dim': [1]}
        self.inputs = {'X': np.random.random((20, 10)).astype("float32")}
        self.outputs = {
            'Out': self.inputs['X'].sum(axis=tuple(self.attrs['dim']))
        }

@skip_check_grad_ci("")
class Test3DReduce0(Test1DReduce):
    def setUp(self):
        self.op_type = "reduce_sum"
        self.attrs = {'dim': [1]}
        self.inputs = {'X': np.random.random((5, 6, 7)).astype("float32")}
        self.outputs = {
            'Out': self.inputs['X'].sum(axis=tuple(self.attrs['dim']))
        }


@skip_check_grad_ci("")
class Test3DReduce1(Test1DReduce):
    def setUp(self):
        self.op_type = "reduce_sum"
        self.attrs = {'dim': [2]}
        self.inputs = {'X': np.random.random((5, 6, 7)).astype("float32")}
        self.outputs = {
            'Out': self.inputs['X'].sum(axis=tuple(self.attrs['dim']))
        }


@skip_check_grad_ci("")
class Test3DReduce2(Test1DReduce):
    def setUp(self):
        self.op_type = "reduce_sum"
        self.attrs = {'dim': [-2]}
        self.inputs = {'X': np.random.random((5, 6, 7)).astype("float32")}
        self.outputs = {
            'Out': self.inputs['X'].sum(axis=tuple(self.attrs['dim']))
        }


@skip_check_grad_ci("")
class Test3DReduce3(Test1DReduce):
    def setUp(self):
        self.op_type = "reduce_sum"
        self.attrs = {'dim': [1, 2]}
        self.inputs = {'X': np.random.random((5, 6, 7)).astype("float32")}
        self.outputs = {
            'Out': self.inputs['X'].sum(axis=tuple(self.attrs['dim']))
        }


# @skip_check_grad_ci("")
# class Test8DReduce0(Test1DReduce):
#     def setUp(self):
#         self.op_type = "reduce_sum"
#         self.attrs = {'dim': (4, 2, 3)}
#         self.inputs = {
#             'X': np.random.random((2, 5, 3, 2, 2, 3, 4, 2)).astype("float32")
#         }
#         self.outputs = {
#             'Out': self.inputs['X'].sum(axis=tuple(self.attrs['dim']))
#         }


@skip_check_grad_ci("")
class TestKeepDimReduce(Test1DReduce):
    def setUp(self):
        self.op_type = "reduce_sum"
        self.inputs = {'X': np.random.random((5, 6, 10)).astype("float32")}
        self.attrs = {'dim': [1], 'keep_dim': True}
        self.outputs = {
            'Out': self.inputs['X'].sum(axis=tuple(self.attrs['dim']),
                                        keepdims=self.attrs['keep_dim'])
        }


# @skip_check_grad_ci("")
# class TestKeepDim8DReduce(Test1DReduce):
#     def setUp(self):
#         self.op_type = "reduce_sum"
#         self.inputs = {
#             'X': np.random.random((2, 5, 3, 2, 2, 3, 4, 2)).astype("float32")
#         }
#         self.attrs = {'dim': (3, 4, 5), 'keep_dim': True}
#         self.outputs = {
#             'Out': self.inputs['X'].sum(axis=tuple(self.attrs['dim']),
#                                         keepdims=self.attrs['keep_dim'])
#         }


@skip_check_grad_ci(
    reason="reduce_anyreduce_any is discontinuous non-derivable function,"
    " its gradient check is not supported by unittest framework.")
class TestReduceMaxOpMultiAxises(OpTest):
    """Remove Max with subgradient from gradient check to confirm the success of CI."""

    def setUp(self):
        self.op_type = "reduce_max"
        self.python_api = paddle.max
        self.inputs = {'X': np.random.random((5, 6, 10)).astype("float32")}
        self.attrs = {'dim': [-2, -1]}
        self.outputs = {
            'Out': self.inputs['X'].max(axis=tuple(self.attrs['dim']))
        }

    def test_check_output(self):
        self.check_output(check_eager=False)


@skip_check_grad_ci(
    reason="reduce_min is discontinuous non-derivable function,"
    " its gradient check is not supported by unittest framework.")
class TestReduceMinOpMultiAxises(OpTest):
    """Remove Min with subgradient from gradient check to confirm the success of CI."""

    def setUp(self):
        self.op_type = "reduce_min"
        self.python_api = paddle.min
        self.inputs = {'X': np.random.random((5, 6, 10)).astype("float32")}
        self.attrs = {'dim': [1, 2]}
        self.outputs = {
            'Out': self.inputs['X'].min(axis=tuple(self.attrs['dim']))
        }

    def test_check_output(self):
        self.check_output(check_eager=False)


@skip_check_grad_ci("")
class TestKeepDimReduceSumMultiAxises(OpTest):
    def setUp(self):
        self.op_type = "reduce_sum"
        self.inputs = {'X': np.random.random((5, 6, 10)).astype("float32")}
        self.attrs = {'dim': [-2, -1], 'keep_dim': True}
        self.outputs = {
            'Out':
            self.inputs['X'].sum(axis=tuple(self.attrs['dim']), keepdims=True)
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


@skip_check_grad_ci("")
class TestReduceSumWithDimOne(OpTest):
    def setUp(self):
        self.op_type = "reduce_sum"
        self.inputs = {'X': np.random.random((100, 1, 1)).astype("float32")}
        self.attrs = {'dim': [1, 2], 'keep_dim': True}
        self.outputs = {
            'Out': self.inputs['X'].sum(axis=tuple(self.attrs['dim']),
                                        keepdims=True)
        }

    def test_check_output(self):
        self.check_output()

    # def test_check_grad(self):
    #     self.check_grad(['X'], 'Out')


@skip_check_grad_ci("")
class TestReduceSumWithNumelOne(OpTest):
    def setUp(self):
        self.op_type = "reduce_sum"
        self.inputs = {'X': np.random.random((100, 1)).astype("float32")}
        self.attrs = {'dim': [1], 'keep_dim': False}
        self.outputs = {
            'Out': self.inputs['X'].sum(axis=tuple(self.attrs['dim']),
                                        keepdims=False)
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


@skip_check_grad_ci("")
class TestReduceAll(OpTest):
    def setUp(self):
        self.op_type = "reduce_sum"
        self.inputs = {'X': np.random.random((100, 1, 1)).astype("float32")}
        self.attrs = {'reduce_all': True, 'keep_dim': False}
        self.outputs = {'Out': self.inputs['X'].sum()}

    def test_check_output(self):
        self.check_output()

    # def test_check_grad(self):
    #     self.check_grad(['X'], 'Out')


@skip_check_grad_ci("")
class Test1DReduceWithAxes1(OpTest):
    def setUp(self):
        self.op_type = "reduce_sum"
        self.inputs = {'X': np.random.random(100).astype("float32")}
        self.attrs = {'dim': [0], 'keep_dim': False}
        self.outputs = {'Out': self.inputs['X'].sum(axis=0)}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


@skip_check_grad_ci("")
class TestReduceWithDtype(OpTest):
    def setUp(self):
        self.op_type = "reduce_sum"
        self.inputs = {'X': np.random.random((6, 2, 10)).astype("float64")}
        self.outputs = {'Out': self.inputs['X'].sum().astype('float64')}
        self.attrs = {'reduce_all': True}
        self.attrs.update({
            'in_dtype': int(convert_np_dtype_to_dtype_(np.float32)),
            'out_dtype': int(convert_np_dtype_to_dtype_(np.float64))
        })

    def test_check_output(self):
        self.check_output()

    # def test_check_grad(self):
    #     self.check_grad(['X'], 'Out')


@skip_check_grad_ci("")
class TestReduceWithDtype1(TestReduceWithDtype):
    def setUp(self):
        self.op_type = "reduce_sum"
        self.inputs = {'X': np.random.random((6, 2, 10)).astype("float64")}
        self.outputs = {'Out': self.inputs['X'].sum(axis=1)}
        self.attrs = {'dim': [1]}
        self.attrs.update({
            'in_dtype': int(convert_np_dtype_to_dtype_(np.float32)),
            'out_dtype': int(convert_np_dtype_to_dtype_(np.float64))
        })


@skip_check_grad_ci("")
class TestReduceWithDtype2(TestReduceWithDtype):
    def setUp(self):
        self.op_type = "reduce_sum"
        self.inputs = {'X': np.random.random((6, 2, 10)).astype("float64")}
        self.outputs = {'Out': self.inputs['X'].sum(axis=1, keepdims=True)}
        self.attrs = {'dim': [1], 'keep_dim': True}
        self.attrs.update({
            'in_dtype': int(convert_np_dtype_to_dtype_(np.float32)),
            'out_dtype': int(convert_np_dtype_to_dtype_(np.float64))
        })


@skip_check_grad_ci("")
class TestReduceSumOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):
            # The input type of reduce_sum_op must be Variable.
            x1 = fluid.create_lod_tensor(
                np.array([[-1]]), [[1]], fluid.CustomPlace('intel_gpu', 0))
            self.assertRaises(TypeError, fluid.layers.reduce_sum, x1)
            # The input dtype of reduce_sum_op  must be float32 or float64 or int32 or int64.
            x2 = fluid.layers.data(name='x2', shape=[4], dtype="uint8")
            self.assertRaises(TypeError, fluid.layers.reduce_sum, x2)


@skip_check_grad_ci("")
class API_TestSumOp(unittest.TestCase):
    def run_static(self,
                   shape,
                   x_dtype,
                   attr_axis,
                   attr_dtype=None,
                   np_axis=None):
        if np_axis is None:
            np_axis = attr_axis

        places = [fluid.CustomPlace('intel_gpu', 0)]
        for place in places:
            with fluid.program_guard(fluid.Program(), fluid.Program()):
                data = fluid.data("data", shape=shape, dtype=x_dtype)
                result_sum = paddle.sum(x=data,
                                        axis=attr_axis,
                                        dtype=attr_dtype)

                exe = fluid.Executor(place)
                input_data = np.random.rand(*shape).astype(x_dtype)
                res, = exe.run(feed={"data": input_data},
                               fetch_list=[result_sum])

            self.assertTrue(
                np.allclose(
                    res, np.sum(input_data.astype(attr_dtype), axis=np_axis)))

    def test_static(self):
        shape = [10, 10]
        axis = 1

        self.run_static(shape, "float32", axis)

        shape = [5, 5, 5]
        self.run_static(shape, "float32", (0, 1))
        self.run_static(shape, "float32", (), np_axis=(0, 1, 2))

    def test_dygraph(self):
        np_x = np.random.random([2, 3, 4]).astype('float32')
        with fluid.dygraph.guard(paddle.CustomPlace('intel_gpu', 0)):
            x = fluid.dygraph.to_variable(np_x)
            out0 = paddle.sum(x).numpy()
            out1 = paddle.sum(x, axis=0).numpy()
            out2 = paddle.sum(x, axis=(0, 1)).numpy()
            out3 = paddle.sum(x, axis=(0, 1, 2)).numpy()
        self.assertTrue(np.allclose(out0, np.sum(np_x, axis=(0, 1, 2)), 1e-5, 1e-5))
        self.assertTrue(np.allclose(out1, np.sum(np_x, axis=0), 1e-5, 1e-5))
        self.assertTrue(np.allclose(out2, np.sum(np_x, axis=(0, 1)), 1e-5, 1e-5))
        self.assertTrue(np.allclose(out3, np.sum(np_x, axis=(0, 1, 2)), 1e-5, 1e-5))

if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()