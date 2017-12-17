from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import argparse
import time

import paddle.v2 as paddle
import paddle.v2.fluid as fluid
import paddle.v2.fluid.profiler as profiler

SEED = 1
DTYPE = "float32"
BATCH_SIZE = 128
PASS = 5


def parse_args():
    parser = argparse.ArgumentParser("mnist model benchmark.")
    parser.add_argument(
        '--batch_size', type=int, default=128, help='The minibatch size.')
    parser.add_argument(
        '--iterations', type=int, default=35, help='The number of minibatches.')
    parser.add_argument(
        '--pass_num', type=int, default=5, help='The number of passes.')
    parser.add_argument(
        '--device',
        type=str,
        default='CPU',
        choices=['CPU', 'GPU'],
        help='The device type.')
    parser.add_argument(
        '--infer_only', action='store_true', help='If set, run forward only.')
    parser.add_argument(
        '--use_cprof', action='store_true', help='If set, use cProfile.')
    parser.add_argument(
        '--use_nvprof',
        action='store_false',
        help='If set, use nvprof for CUDA.')
    args = parser.parse_args()
    return args


def print_arguments(args):
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).iteritems()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


def paddle_random_normal(shape, loc=.0, scale=1., seed=1, dtype="float32"):
    program = fluid.framework.Program()
    block = program.global_block()
    w = block.create_var(
        dtype=dtype,
        shape=shape,
        lod_level=0,
        name="param",
        initializer=fluid.initializer.NormalInitializer(
            loc=.0, scale=scale, seed=seed))
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    out = exe.run(program, fetch_list=[w])
    return np.array(out[0])


def v2_fluid_init_parameters(parameters,
                             f,
                             exclude_params=[],
                             scale=1.,
                             seed=1,
                             dtype="float32"):
    tar_param = parameters.from_tar(f)
    for pname in tar_param.names():
        if pname in parameters.names() and pname not in exclude_params:
            shape = tar_param.get(pname).shape
            para = paddle_random_normal(
                shape, scale=scale, seed=seed, dtype=dtype)
            parameters.set(pname, para)


def cnn_model(data):
    # first conv layer
    conv_pool_1 = paddle.networks.simple_img_conv_pool(
        input=data,
        filter_size=5,
        num_filters=20,
        num_channel=1,
        pool_size=2,
        pool_stride=2,
        act=paddle.activation.Relu())
    # second conv layer
    conv_pool_2 = paddle.networks.simple_img_conv_pool(
        input=conv_pool_1,
        filter_size=5,
        num_filters=50,
        num_channel=20,
        pool_size=2,
        pool_stride=2,
        act=paddle.activation.Relu())
    # fully-connected layer
    predict = paddle.layer.fc(input=conv_pool_2,
                              size=10,
                              act=paddle.activation.Softmax())
    return predict


def run_benchmark(model, args):
    if args.use_cprof:
        pr = cProfile.Profile()
        pr.enable()
    start_time = time.time()

    paddle.init(use_gpu=False, trainer_count=1)
    # define network topology
    images = paddle.layer.data(
        name='pixel', type=paddle.data_type.dense_vector(784))
    label = paddle.layer.data(
        name='label', type=paddle.data_type.integer_value(10))
    predict = model(images)

    cost = paddle.layer.classification_cost(input=predict, label=label)
    parameters = paddle.parameters.create(cost)

    optimizer = paddle.optimizer.Adam(beta1=0.9, beta2=0.999)
    #optimizer = paddle.optimizer.Momentum(
    #    learning_rate=0.1 / 128.0,
    #    momentum=0.9,
    #    regularization=paddle.optimizer.L2Regularization(rate=0.0005 * 128))

    trainer = paddle.trainer.SGD(cost=cost,
                                 parameters=parameters,
                                 update_equation=optimizer)

    # init v2 parameter with fluid init
    with open('params_pass_0.tar', 'r') as f:
        v2_fluid_init_parameters(parameters, f, seed=SEED, dtype=DTYPE)

    class Namespace:
        pass

    ns = Namespace()
    ns.start = time.clock()

    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            #if event.batch_id % 100 == 0:
            end = time.clock()
            dir(event.metrics)
            print("pass=%d, batch=%d, loss=%f, error=%f, elapse=%f" %
                  (event.pass_id, event.batch_id, event.cost,
                   event.metrics.values()[0], (end - ns.start) / 1000))
            ns.start = time.clock()
            for p in parameters:
                # print("parameters:", parameters.get(p))
                para = parameters.get(p)
                print("para min:%f, max:%f, max_abs:%f:" % (para.min(), para.max(),  max(para.min(), para.max(), key=abs)))

                # grad = parameters.get_grad(p)
                # print("gradients:", grad)
                # print("gradients max abs:" + str(
                #     max(grad.min(), grad.max(), key=abs)))

        if isinstance(event, paddle.event.EndPass):
            result = trainer.test(reader=paddle.batch(
                paddle.dataset.mnist.test(), batch_size=128))
            print("Test with Pass %d, Cost %f, %s\n" %
                  (event.pass_id, result.cost, result.metrics))

    trainer.train(
        reader=paddle.batch(
            paddle.dataset.mnist.train(), batch_size=BATCH_SIZE),
        event_handler=event_handler,
        num_passes=PASS)


if __name__ == '__main__':
    args = parse_args()
    print_arguments(args)
    if args.use_nvprof and args.device == 'GPU':
        with profiler.cuda_profiler("cuda_profiler.txt", 'csv') as nvprof:
            run_benchmark(cnn_model, args)
    else:
        run_benchmark(cnn_model, args)
