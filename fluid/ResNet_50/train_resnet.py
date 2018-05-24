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

from models import resnet
import paddle
import paddle.fluid as fluid
# import paddle.dataset.flowers as flowers
import paddle.fluid.profiler as profiler

fluid.default_startup_program().random_seed = 111

warm_up_num = 20
display_step = 10
skip_first_steps = 10
number_iteration = 150

batch_size = 64
batch_size_per_gpu = 64

fix_data_in_gpu = True
use_mem_opt = True
show_record_time = False
balance_parameter_opt_between_cards = False


def net_conf(image, label, class_dim):
    out = resnet.resnet_imagenet(input=image, class_dim=class_dim)
    cost = fluid.layers.cross_entropy(input=out, label=label)
    avg_cost = fluid.layers.mean(x=cost)
    return out, avg_cost


def add_optimizer(avg_cost):
    optimizer = fluid.optimizer.Momentum(learning_rate=0.01, momentum=0.9)
    optimizer.minimize(avg_cost)

    if use_mem_opt:
        fluid.memory_optimize(fluid.default_main_program())


def train_parallel_exe():

    class_dim = 1000
    image_shape = [3, 224, 224]

    def fake_reader():
        while True:
            img = np.random.rand(3, 224, 224)
            lab = np.random.randint(0, 999)
            yield img, lab

    def train():
        return fake_reader

    image = fluid.layers.data(name='image', shape=image_shape, dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    place = fluid.CUDAPlace(0)
    feeder = fluid.DataFeeder(place=place, feed_list=[image, label])
    train_reader = feeder.decorate_reader(
        paddle.batch(
            train(), batch_size=batch_size_per_gpu),
        multi_devices=True)

    train_reader_iter = train_reader()
    if fix_data_in_gpu:
        data = train_reader_iter.next()
        feed_data = data

    prediction, avg_cost = net_conf(image, label, class_dim)

    add_optimizer(avg_cost)

    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    exec_strategy = fluid.ExecutionStrategy()
    exec_strategy.allow_op_delay = True

    build_strategy = fluid.BuildStrategy()
    build_strategy.reduce_strategy = fluid.BuildStrategy.ReduceStrategy.Reduce if balance_parameter_opt_between_cards else fluid.BuildStrategy.ReduceStrategy.AllReduce

    exe = fluid.ParallelExecutor(
        loss_name=avg_cost.name,
        use_cuda=True,
        build_strategy=build_strategy,
        exec_strategy=exec_strategy)

    # warm up
    for batch_id in xrange(warm_up_num):
        cost_val = exe.run([avg_cost.name]
                           if (batch_id + 1) % display_step == 0 else [],
                           feed=feed_data
                           if fix_data_in_gpu else train_reader_iter.next())

    time_record = []
    img_count = 0
    train_start = time.time()
    for batch_id in xrange(number_iteration):
        cost_val = exe.run([avg_cost.name]
                           if (batch_id + 1) % display_step == 0 else [],
                           feed=feed_data
                           if fix_data_in_gpu else train_reader_iter.next())

        img_count += batch_size

        if (batch_id + 1) % display_step == 0:
            step_time = time.time() - train_start
            time_record.append(step_time)

            print("iter=%d, cost=%s, elapse=%f, img/sec=%f" %
                  ((batch_id + 1), np.array(cost_val[0]), step_time,
                   img_count / step_time))

            img_count = 0
            train_start = time.time()

    skip_time_record = skip_first_steps / display_step
    time_record[0:skip_time_record] = []

    if show_record_time:
        for i, ele in enumerate(time_record):
            print("iter:{0}, time consume:{1}".format(i, ele))

    img_count = (number_iteration - skip_first_steps) * batch_size

    print("average time:{0}, img/sec:{1}".format(
        np.mean(time_record), img_count / np.sum(time_record)))


if __name__ == '__main__':
    cards = os.getenv("CUDA_VISIBLE_DEVICES") or ""
    cards_num = len(cards.split(","))
    batch_size = batch_size_per_gpu * cards_num

    print("cards_num=" + str(cards_num))

    train_parallel_exe()
