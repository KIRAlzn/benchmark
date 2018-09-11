import os
import numpy as np
import time
import sys
import paddle
import paddle.fluid as fluid
from se_resnext_v2 import SE_ResNeXt
import reader
from config import *

import argparse
import functools
from utility import add_arguments, print_arguments
from paddle.fluid.layers.learning_rate_scheduler import _decay_step_counter
from paddle.fluid.initializer import init_on_cpu
import math

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('batch_size',   int,  256, "Minibatch size.")
add_arg('batch_size_per_trainer',   int,  32, "Minibatch size.")
add_arg('run_pass',   int,  10, "run number passes.")
add_arg('num_layers',   int,  50,  "How many layers for SE-ResNeXt model.")
add_arg('with_mem_opt', bool, False, "Whether to use memory optimization or not.")
add_arg('use_py_reader', bool, True, "")
add_arg('use_gpu', bool, True, "")
# add_arg('parallel_exe', bool, True, "Whether to use ParallelExecutor to train or not.")

def cosine_decay(learning_rate, step_each_epoch, epochs = 120):
    """Applies cosine decay to the learning rate.
    lr = 0.05 * (math.cos(epoch * (math.pi / 120)) + 1)
    """
    global_step = _decay_step_counter()
    with init_on_cpu():
        epoch = fluid.layers.floor(global_step / step_each_epoch)
        lr = learning_rate / 2.
        decayed_lr = lr * (fluid.layers.cos(epoch * (math.pi / epochs)) + 1)
    return decayed_lr


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


def train_parallel_exe(args,
                       learning_rate,
                       batch_size,
                       num_passes,
                       init_model=None,
                       model_save_dir='model',
                       parallel=True,
                       use_nccl=True,
                       lr_strategy=None,
                       layers=50):
    class_dim = 1000
    image_shape = [3, 224, 224]

    # Define training Program
    train_prog = fluid.Program()
    train_startup_prog = fluid.Program()
    with fluid.program_guard(train_prog, train_startup_prog):
        with fluid.unique_name.guard():
            image, label, py_reader = get_image_label()
            # image = fluid.layers.data(name='image', shape=image_shape, dtype='float32')
            # label = fluid.layers.data(name='label', shape=[1], dtype='int64')
            out = SE_ResNeXt(input=image, class_dim=class_dim, layers=layers)
            acc_top1 = fluid.layers.accuracy(input=out, label=label, k=1)
            acc_top5 = fluid.layers.accuracy(input=out, label=label, k=5)
            cost = fluid.layers.cross_entropy(input=out, label=label)
            avg_cost = fluid.layers.mean(x=cost)

            if "piecewise_decay" in lr_strategy:
                bd = lr_strategy["piecewise_decay"]["bd"]
                lr = lr_strategy["piecewise_decay"]["lr"]
                optimizer = fluid.optimizer.Momentum(
                    learning_rate=fluid.layers.piecewise_decay(
                        boundaries=bd, values=lr),
                    momentum=0.9,
                    regularization=fluid.regularizer.L2Decay(1e-4))
            elif "cosine_decay" in lr_strategy:
                print('cosine_decay')
                step_each_epoch = lr_strategy["cosine_decay"]["step_each_epoch"]
                epochs = lr_strategy["cosine_decay"]["epochs"]
                optimizer = fluid.optimizer.Momentum(
                    learning_rate=cosine_decay(learning_rate=learning_rate,
                        step_each_epoch=step_each_epoch, epochs=epochs),
                    momentum=0.9,
                    regularization=fluid.regularizer.L2Decay(1e-4))
            else:
                optimizer = fluid.optimizer.Momentum(
                    learning_rate=learning_rate,
                    momentum=0.9,
                    regularization=fluid.regularizer.L2Decay(1e-4))
            opts = optimizer.minimize(avg_cost)

    if args.with_mem_opt:
        fluid.memory_optimize(train_prog)

    # Init parameter
    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    fluid.Executor(place).run(train_startup_prog)

    if init_model is not None:
        fluid.io.load_persistables(exe, init_model)

    feed_list = None
    if args.use_py_reader:
        # Init Reader
        # train_reader.decorate_paddle_reader(
        # paddle.v2.reader.shuffle(
        # paddle.batch(mnist.train(), 512), buf_size=8192))
        train_reader.decorate_paddle_reader(
            paddle.batch(
                reader.train(), batch_size=args.batch_size_per_trainer))
    else:
        train_reader = paddle.batch(reader.train(), batch_size=batch_size)
        train_reader = paddle.batch(flowers.train(), batch_size=args.batch_size)
        feeder = fluid.DataFeeder(place=place, feed_list=[image, label])
        feed_list = [train_image, train_label]

    # Init ParallelExecutor
    exec_strategy = fluid.ExecutionStrategy()
    build_strategy = fluid.BuildStrategy()
    build_strategy.fuse_adjacent_ops = True if args.fuse_adjacent_ops else False

    # Create train_exe 
    train_exe = fluid.ParallelExecutor(
        use_cuda=True if args.use_gpu else False,
        loss_name=avg_cost.name,
        main_program=train_prog,
        build_strategy=build_strategy,
        exec_strategy=exec_strategy)

    fetch_list = [avg_cost.name, acc_top1.name, acc_top5.name]

    for pass_id in range(num_passes):
        if pass_id == args.run_pass:
            break
        batch_time = []

        py_reader.start()
        time.sleep(1)
        batch_id = 0
        while True:
            # print py_reader.queue.size()
            beg = time.time()
            try:
                outs = train_exe.run(fetch_list=fetch_list)
            except fluid.core.EOFException:
                # The current pass is over.
                print("The current pass is over.")
                py_reader.reset()
                break

            batch_time.append(time.time() - beg)
            batch_id += 1

            loss = np.mean(np.array(outs[0]))
            acc1 = np.mean(np.array(outs[1]))
            acc5 = np.mean(np.array(outs[2]))
            train_info[0].append(loss)
            train_info[1].append(acc1)
            train_info[2].append(acc5)
            print("Pass {0}, trainbatch {1}, loss {2}, \
                       acc1 {3}, acc5 {4} time {5}"
                                                   .format(pass_id, \
                       batch_id, loss, acc1, acc5, \
                       "%2.2f sec" % period))

        if batch_id  == args.skip_first_steps:
            batch_time[0:args.skip_first_steps] = []
        print("Average time cost: %f" % (np.mean(batch_time)))


if __name__ == '__main__':
    args = parser.parse_args()
    print_arguments(args)

    total_images = 1281167
    batch_size = 256
    step = int(total_images / batch_size + 1)
    num_epochs = 120

    # mode: piecewise_decay, cosine_decay
    learning_rate_mode = "cosine_decay"
    #learning_rate_mode = "piecewise_decay"
    lr_strategy = {}
    if learning_rate_mode == "piecewise_decay":
        epoch_points = [30, 60, 90]
        bd = [e * step for e in epoch_points]
        lr = [0.1, 0.01, 0.001, 0.0001]
        lr_strategy[learning_rate_mode] = {
            "bd": bd,
            "lr": lr
        }
    elif learning_rate_mode == "cosine_decay":
        lr_strategy[learning_rate_mode] = {
            "step_each_epoch":step,
            "epochs":120
        }
    else:
        lr_strategy = None

    use_nccl = True
    # layers: 50, 152
    layers = args.num_layers
    method = train_parallel_exe # if args.parallel_exe else train_parallel_do
    method(args,
           learning_rate=0.1,
           batch_size=batch_size,
           num_passes=num_epochs,
           init_model=None,
           parallel=True,
           use_nccl=True,
           lr_strategy=lr_strategy,
           layers=layers)
