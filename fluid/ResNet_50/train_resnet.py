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

import os
import time
import argparse
import distutils.util
import numpy as np
from functools import partial

from config import *
from models import resnet
import paddle
import paddle.fluid as fluid
import paddle.dataset.flowers as flowers
import paddle.fluid.profiler as profiler


def parse_args():
    parser = argparse.ArgumentParser('ResNet-50 parallel profile.')
    parser.add_argument('--batch_size', type=int, default=12, help='batch size')
    parser.add_argument(
        '--batch_size_per_trainer', type=int, default=12, help='')
    parser.add_argument(
        '--use_mem_opt',
        type=distutils.util.strtobool,
        default=True,
        help='use memory optimize or not.')
    parser.add_argument('--pass_num', type=int, default=10, help='')
    parser.add_argument('--skip_first_steps', type=int, default=5, help='')
    parser.add_argument(
        '--use_gpu', type=distutils.util.strtobool, default=True, help='')
    parser.add_argument(
        '--use_py_reader',
        type=distutils.util.strtobool,
        default=False,
        help='.')
    parser.add_argument(
        '--reduce_mode',
        type=distutils.util.strtobool,
        default=False,
        help='balance parameter opt between cards')
    parser.add_argument(
        '--with_test', type=distutils.util.strtobool, default=False, help='')
    parser.add_argument(
        '--fuse_adjacent_ops',
        type=distutils.util.strtobool,
        default=False,
        help='')
    parser.add_argument(
        '--fix_seed', type=distutils.util.strtobool, default=True, help='')
    args = parser.parse_args()
    return args


args = parse_args()


def print_arguments():
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).iteritems()):
        print('%s=%s' % (arg, value))


def make_all_py_reader_inputs(is_test=False):
    print 'feed ', input_fields
    reader = fluid.layers.py_reader(
        capacity=20,
        name="test_reader" if is_test else "train_reader",
        shapes=[input_descs[input_field][0] for input_field in input_fields],
        dtypes=[input_descs[input_field][1] for input_field in input_fields],
        lod_levels=[
            input_descs[input_field][2]
            if len(input_descs[input_field]) == 3 else 0
            for input_field in input_fields
        ])
    return fluid.layers.read_file(reader), reader


def get_image_label(is_test=False):
    reader = None
    if args.use_py_reader:
        all_inputs, reader = make_all_py_reader_inputs(is_test=is_test)
        image, label = all_inputs
    else:
        image = fluid.layers.data(
            name='image', shape=image_shape, dtype='float32')
        label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    return image, label, reader


def net_conf(image, label, class_dim):
    out = resnet.resnet_imagenet(input=image, class_dim=class_dim)
    cost = fluid.layers.cross_entropy(input=out, label=label)
    avg_cost = fluid.layers.mean(x=cost)
    return out, avg_cost


def run_use_py_reader(py_reader,
                      executor,
                      fetch_list,
                      display_metric,
                      feed_list=None):
    batch_time = []
    py_reader.start()
    time.sleep(1)
    batch_id = 0
    while True:
        # print py_reader.queue.size()
        beg = time.time()
        try:
            outs = executor.run(fetch_list=fetch_list)
        except fluid.core.EOFException:
            # The current pass is over.
            print("The current pass is over.")
            py_reader.reset()
            break

        batch_time.append(time.time() - beg)
        batch_id += 1
        display_metric(outs, batch_id, time.time() - beg)

    if len(batch_time) > args.skip_first_steps:
        batch_time[0:args.skip_first_steps] = []
        print("drop the first %d batch time" % (args.skip_first_steps))
    else:
        print("the number of step is %d, "
              "but the skip_first_steps is %d. "
              "So you didn't skip the first steps" %
              (len(batch_time), args.skip_first_steps))

    print("Average time cost: %f" % (np.mean(batch_time)))


def run_use_feed(reader, executor, fetch_list, display_metric, feed_list):
    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    feeder = fluid.DataFeeder(place=place, feed_list=feed_list)
    batch_time = []
    for batch_id, data in enumerate(reader()):
        beg = time.time()
        outs = executor.run(fetch_list, feed=feeder.feed(data))
        batch_time.append(time.time() - beg)
        display_metric(outs, batch_id, time.time() - beg)

    if len(batch_time) > args.skip_first_steps:
        batch_time[0:args.skip_first_steps] = []
    else:
        print("the number of step is %d, "
              "but the skip_first_steps is %d. "
              "So you didn't skip the first steps" %
              (len(batch_time), args.skip_first_steps))

    print("Average time cost: %f" % (np.mean(batch_time)))


def test_parallel_exe(trainer):
    # Define testing Program
    test_prog = fluid.Program()
    test_startup_prog = fluid.Program()
    if args.fix_seed:
        test_startup_prog.random_seed = 1
    with fluid.program_guard(test_prog, test_startup_prog):
        with fluid.unique_name.guard():
            test_image, test_label, test_reader = get_image_label(is_test=True)
            test_prediction, test_avg_cost = net_conf(test_image, test_label,
                                                      class_dim)
            batch_size_tensor = fluid.layers.create_tensor(dtype='int64')
            batch_acc = fluid.layers.accuracy(
                input=test_prediction,
                label=test_label,
                total=batch_size_tensor)

    test_prog = test_prog.clone(for_test=True)

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    fluid.Executor(place).run(test_startup_prog)

    if args.use_mem_opt:
        fluid.memory_optimize(test_prog)

    feed_list = None
    if args.use_py_reader:
        # Init Reader
        test_reader.decorate_paddle_reader(
            paddle.batch(
                flowers.test(), batch_size=args.batch_size_per_trainer))
    else:
        test_reader = paddle.batch(flowers.test(), batch_size=args.batch_size)
        feed_list = [test_image, test_label]

    # Init ParallelExecutor
    # Create train_exe 
    exec_strategy = fluid.ExecutionStrategy()
    # exec_strategy.allow_op_delay = False
    build_strategy = fluid.BuildStrategy()
    build_strategy.reduce_strategy = \
            fluid.BuildStrategy.ReduceStrategy.Reduce \
            if args.reduce_mode \
            else fluid.BuildStrategy.ReduceStrategy.AllReduce
    if args.fuse_adjacent_ops:
        build_strategy.fuse_adjacent_ops = True

    tester = fluid.ParallelExecutor(
        use_cuda=True if args.use_gpu else False,
        main_program=test_prog,
        share_vars_from=trainer,
        build_strategy=build_strategy,
        exec_strategy=exec_strategy)

    def test(pass_id, tester=tester, test_reader=test_reader):
        batch_accs = []
        batch_size = []

        def test_display_metric(output, batch_id, time_consum, pass_id):
            batch_acc_val, batch_size_var = \
                np.array(output[0]), np.array(output[1])
            acc = float(
                (batch_acc_val * batch_size_var).sum() / batch_size_var.sum())
            batch_accs.append(acc)
            batch_size.append(batch_size_var.sum())
            print(
                "pass:%d, batch: %d, acc: %s, batch_size_val: %s, speed:%f img/sec"
                % (pass_id, batch_id, acc, batch_size_var,
                   batch_size_var.sum() / time_consum))

        if args.use_py_reader:
            test_one_pass = run_use_py_reader
        else:
            test_one_pass = run_use_feed

        test_one_pass(
            test_reader,
            tester,
            fetch_list=[batch_acc.name, batch_size_tensor.name],
            display_metric=partial(
                test_display_metric, pass_id=pass_id),
            feed_list=feed_list)

        print("test accuracy:%f" %
              (np.dot(batch_accs, batch_size) / np.sum(batch_size)))

    return test


def train_parallel_exe():
    # Define training Program
    train_prog = fluid.Program()
    train_startup_prog = fluid.Program()
    if args.fix_seed:
        train_prog.random_seed = 1
        train_startup_prog.random_seed = 1
    with fluid.program_guard(train_prog, train_startup_prog):
        with fluid.unique_name.guard():
            train_image, train_label, train_reader = get_image_label(
                is_test=False)
            train_prediction, avg_cost = net_conf(train_image, train_label,
                                                  class_dim)
            optimizer = fluid.optimizer.Momentum(
                learning_rate=0.01,
                momentum=0.9,
                regularization=fluid.regularizer.L2Decay(1e-6))
            optimizer.minimize(avg_cost)
            batch_size_tensor = fluid.layers.create_tensor(dtype='int64')
            batch_acc = fluid.layers.accuracy(
                input=train_prediction,
                label=train_label,
                total=batch_size_tensor)

    # Optimize Memory
    if args.use_mem_opt:
        fluid.memory_optimize(train_prog)

    # Init parameter
    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    fluid.Executor(place).run(train_startup_prog)

    # Init ParallelExecutor
    exec_strategy = fluid.ExecutionStrategy()
    # exec_strategy.allow_op_delay = False
    build_strategy = fluid.BuildStrategy()
    build_strategy.reduce_strategy = \
            fluid.BuildStrategy.ReduceStrategy.Reduce \
            if args.reduce_mode \
            else fluid.BuildStrategy.ReduceStrategy.AllReduce
    build_strategy.fuse_adjacent_ops = True if args.fuse_adjacent_ops else False

    # Create train_exe 
    trainer = fluid.ParallelExecutor(
        use_cuda=True if args.use_gpu else False,
        loss_name=avg_cost.name,
        main_program=train_prog,
        build_strategy=build_strategy,
        exec_strategy=exec_strategy)

    if args.with_test:
        test = test_parallel_exe(trainer)

    feed_list = None
    if args.use_py_reader:
        # Init Reader
        # train_reader.decorate_paddle_reader(
        # paddle.v2.reader.shuffle(
        # paddle.batch(mnist.train(), 512), buf_size=8192))
        train_reader.decorate_paddle_reader(
            paddle.batch(
                flowers.train(), batch_size=args.batch_size_per_trainer))
    else:
        train_reader = paddle.batch(flowers.train(), batch_size=args.batch_size)
        feed_list = [train_image, train_label]

    def train_display_metric(output, batch_id, time_consum, pass_id=1):
        avg_cost_val, batch_acc_val, batch_size_var = \
                np.array(output[0]), np.array(output[1]), np.array(output[2])
        acc = float(
            (batch_acc_val * batch_size_var).sum() / batch_size_var.sum())
        loss = np.dot(avg_cost_val, batch_size_var) / batch_size_var.sum()
        print("epoch: %d, batch: %d, batch_acc_val: %f, batch_size_val: %s,"
              " avg loss: %f, speed:%f img/sec" %
              (pass_id, batch_id, acc, batch_size_var, loss,
               batch_size_var.sum() / time_consum))

    for pass_id in xrange(args.pass_num):
        if args.use_py_reader:
            train_one_pass = run_use_py_reader
        else:
            train_one_pass = run_use_feed

        train_one_pass(
            train_reader,
            trainer,
            fetch_list=[avg_cost.name, batch_acc.name, batch_size_tensor.name],
            display_metric=partial(
                train_display_metric, pass_id=pass_id),
            feed_list=feed_list)
        if args.with_test:
            test(pass_id=pass_id)


if __name__ == '__main__':
    if args.use_gpu:
        cards = os.getenv("CUDA_VISIBLE_DEVICES") or ""
        trainer_num = len(cards.split(","))
    else:
        trainer_num = int(os.getenv("CPU_NUM"))

    args.batch_size = args.batch_size_per_trainer * trainer_num
    print_arguments()

    print("trainer_num=" + str(trainer_num))
    train_parallel_exe()
