from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import argparse
import time
import os
import distutils.util
import paddle.v2 as paddle
import paddle.fluid as fluid
import paddle.fluid.profiler as profiler

SEED = 1
DTYPE = "float32"

# random seed must set before configuring the network.
fluid.default_startup_program().random_seed = SEED


def parse_args():
    parser = argparse.ArgumentParser("mnist model benchmark.")
    parser.add_argument(
        '--batch_size', type=int, default=128, help='The minibatch size.')
    parser.add_argument(
        '--per_gpu_batch_size',
        type=int,
        default=128,
        help='The minibatch size.')
    parser.add_argument(
        '--skip_batch_num',
        type=int,
        default=5,
        help='The first num of minibatch num to skip, for better performance test'
    )
    parser.add_argument(
        '--iterations', type=int, default=35, help='The number of minibatches.')
    parser.add_argument(
        '--pass_num', type=int, default=5, help='The number of passes.')
    parser.add_argument(
        '--device',
        type=str,
        default='GPU',
        choices=['CPU', 'GPU'],
        help='The device type.')
    parser.add_argument(
        '--parallel',
        type=distutils.util.strtobool,
        default=True,
        help='use memory optimize or not.')
    parser.add_argument(
        '--use_mem_opt',
        type=distutils.util.strtobool,
        default=True,
        help='use memory optimize or not.')
    parser.add_argument(
        '--use_nccl',
        type=distutils.util.strtobool,
        default=True,
        help='use memory optimize or not.')
    parser.add_argument(
        '--use_parallel_mode',
        type=str,
        default='parallel_do',
        choices=['parallel_do', 'parallel_exe'],
        help='The parallel mode("parallel_do" or "parallel_exe").')
    args = parser.parse_args()
    return args


def cnn_model(data):
    conv_pool_1 = fluid.nets.simple_img_conv_pool(
        input=data,
        filter_size=5,
        num_filters=20,
        pool_size=2,
        pool_stride=2,
        act="relu")
    conv_pool_2 = fluid.nets.simple_img_conv_pool(
        input=conv_pool_1,
        filter_size=5,
        num_filters=50,
        pool_size=2,
        pool_stride=2,
        act="relu")

    # TODO(dzhwinter) : refine the initializer and random seed settting
    SIZE = 10
    input_shape = conv_pool_2.shape
    param_shape = [reduce(lambda a, b: a * b, input_shape[1:], 1)] + [SIZE]
    scale = (2.0 / (param_shape[0]**2 * SIZE))**0.5

    predict = fluid.layers.fc(
        input=conv_pool_2,
        size=SIZE,
        act="softmax",
        param_attr=fluid.param_attr.ParamAttr(
            initializer=fluid.initializer.NormalInitializer(
                loc=0.0, scale=scale)))
    return predict


def add_optimizer(args, avg_cost):
    opt = fluid.optimizer.AdamOptimizer(
        learning_rate=0.001, beta1=0.9, beta2=0.999)
    opt.minimize(avg_cost)
    if args.use_mem_opt:
        fluid.memory_optimize(fluid.default_main_program())


def train_parallel_exe(args, model):

    # Input data
    images = fluid.layers.data(name='pixel', shape=[1, 28, 28], dtype=DTYPE)
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    # Train program
    predict = model(images)
    cost = fluid.layers.cross_entropy(input=predict, label=label)
    avg_cost = fluid.layers.mean(x=cost)

    # Evaluator
    batch_size_tensor = fluid.layers.create_tensor(dtype='int64')
    batch_acc = fluid.layers.accuracy(
        input=predict, label=label, total=batch_size_tensor)

    # Optimization
    add_optimizer(args, avg_cost)

    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    train_exe = fluid.ParallelExecutor(
        loss_name=avg_cost.name, use_cuda=True, allow_op_delay=True)

    # Reader
    train_reader = paddle.batch(
        paddle.dataset.mnist.train(), batch_size=args.batch_size)

    accuracy = fluid.average.WeightedAverage()
    iters, num_samples, start_time = 0, 0, time.time()
    for pass_id in range(args.pass_num):
        accuracy.reset()
        train_accs = []
        train_losses = []
        for batch_id, data in enumerate(train_reader()):
            if iters == args.skip_batch_num:
                start_time = time.time()
                num_samples = 0
            if iters == args.iterations:
                break
            img_data = np.array(
                map(lambda x: x[0].reshape([1, 28, 28]), data)).astype(DTYPE)
            y_data = np.array(map(lambda x: x[1], data)).astype("int64")
            y_data = y_data.reshape([len(y_data), 1])

            outs = train_exe.run(
                [avg_cost.name, batch_acc.name, batch_size_tensor.name],  #
                feed_dict={"pixel": img_data,
                           "label": y_data}
            )  # The accuracy is the accumulation of batches, but not the current batch.
            accuracy.add(value=np.array(outs[1]),
                         weight=np.sum(np.array(outs[2])))
            # accuracy.add(value=np.mean(np.array(outs[1])), weight=np.sum(np.array(outs[2])))
            iters += 1
            num_samples += len(y_data)
            loss = np.mean(np.array(outs[0]))
            acc = np.mean(np.array(outs[1]))
            train_losses.append(loss)
            train_accs.append(acc)
            print("Pass: %d, Iter: %d, Loss: %f, Accuracy: %f" %
                  (pass_id, iters, loss, acc))

        print("Pass: %d, Loss: %f, Train Accuray: %f\n" %
              (pass_id, np.mean(train_losses), np.mean(train_accs)))
        train_elapsed = time.time() - start_time
        examples_per_sec = num_samples / train_elapsed

        print('\nTotal examples: %d, total time: %.5f, %.5f examples/sed\n' %
              (num_samples, train_elapsed, examples_per_sec))

        if iters == args.iterations:
            break
        #exit(0)


def train_parallel_do(args, model):

    # Input data
    images = fluid.layers.data(name='pixel', shape=[1, 28, 28], dtype=DTYPE)
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    if args.parallel:
        places = fluid.layers.get_places()
        pd = fluid.layers.ParallelDo(places, use_nccl=args.use_nccl)

        with pd.do():
            image_ = pd.read_input(images)
            label_ = pd.read_input(label)
            predict = model(image_)
            cost = fluid.layers.cross_entropy(input=predict, label=label_)
            avg_cost = fluid.layers.mean(x=cost)

            batch_size_tensor = fluid.layers.create_tensor(dtype='int64')
            batch_acc = fluid.layers.accuracy(
                input=predict, label=label_, total=batch_size_tensor)

            pd.write_output(avg_cost)
            pd.write_output(batch_acc)
            pd.write_output(batch_size_tensor)

        avg_cost, batch_acc, batch_size_tensor = pd()
        avg_cost = fluid.layers.mean(x=avg_cost)
        batch_acc = fluid.layers.mean(x=batch_acc)
        batch_size_tensor = fluid.layers.sums(input=batch_size_tensor)
    else:
        # Train program
        predict = model(images)
        cost = fluid.layers.cross_entropy(input=predict, label=label)
        avg_cost = fluid.layers.mean(x=cost)

        # Evaluator
        batch_size_tensor = fluid.layers.create_tensor(dtype='int64')
        batch_acc = fluid.layers.accuracy(
            input=predict, label=label, total=batch_size_tensor)

    # Optimization
    add_optimizer(args, avg_cost)

    # Initialize executor
    place = fluid.CPUPlace() if args.device == 'CPU' else fluid.CUDAPlace(0)
    exe = fluid.Executor(place)

    # Parameter initialization
    exe.run(fluid.default_startup_program())

    # Reader
    train_reader = paddle.batch(
        paddle.dataset.mnist.train(), batch_size=args.batch_size)

    accuracy = fluid.average.WeightedAverage()
    iters, num_samples, start_time = 0, 0, time.time()
    for pass_id in range(args.pass_num):
        accuracy.reset()
        train_accs = []
        train_losses = []
        for batch_id, data in enumerate(train_reader()):
            if iters == args.skip_batch_num:
                start_time = time.time()
                num_samples = 0
            if iters == args.iterations:
                break
            img_data = np.array(
                map(lambda x: x[0].reshape([1, 28, 28]), data)).astype(DTYPE)
            y_data = np.array(map(lambda x: x[1], data)).astype("int64")
            y_data = y_data.reshape([len(y_data), 1])

            outs = exe.run(
                fluid.default_main_program(),
                feed={"pixel": img_data,
                      "label": y_data},
                fetch_list=[avg_cost, batch_acc, batch_size_tensor]
            )  # The accuracy is the accumulation of batches, but not the current batch.
            accuracy.add(value=outs[1], weight=np.sum(outs[2]))
            iters += 1
            num_samples += len(y_data)
            loss = np.array(outs[0])
            acc = np.array(outs[1])
            train_losses.append(loss)
            train_accs.append(acc)
            print("Pass: %d, Iter: %d, Loss: %f, Accuracy: %f" %
                  (pass_id, iters, loss, acc))

        print("Pass: %d, Loss: %f, Train Accuray: %f\n" %
              (pass_id, np.mean(train_losses), np.mean(train_accs)))
        train_elapsed = time.time() - start_time
        examples_per_sec = num_samples / train_elapsed

        print('\nTotal examples: %d, total time: %.5f, %.5f examples/sed\n' %
              (num_samples, train_elapsed, examples_per_sec))

        if iters == args.iterations:
            break

        #exit(0)


def print_arguments(args):
    print('----------- mnist Configuration Arguments -----------')
    for arg, value in sorted(vars(args).iteritems()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


if __name__ == '__main__':
    args = parse_args()
    cards = os.getenv("CUDA_VISIBLE_DEVICES") or ""
    cards_num = len(cards.split(","))
    if cards_num == 1:
        args.parallel = False
    args.batch_size = args.per_gpu_batch_size * cards_num
    print_arguments(args)
    print("the number of card:%d" % (cards_num))

    if args.use_parallel_mode == "parallel_exe":
        train_parallel_exe(args, cnn_model)
    else:
        train_parallel_do(args, cnn_model)
