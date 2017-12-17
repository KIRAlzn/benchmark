import os
from PIL import Image
import numpy as np
import paddle.v2 as paddle
import paddle.v2.fluid as fluid

with_gpu = os.getenv('WITH_GPU', '0') != '0'

BATCH_SIZE = 128
PASS = 5


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


def convolutional_neural_network(img):
    # first conv layer
    conv_pool_1 = paddle.networks.simple_img_conv_pool(
        input=img,
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


def main():
    paddle.init(use_gpu=with_gpu, trainer_count=1)

    # define network topology
    images = paddle.layer.data(
        name='pixel', type=paddle.data_type.dense_vector(784))
    label = paddle.layer.data(
        name='label', type=paddle.data_type.integer_value(10))

    predict = convolutional_neural_network(images)
    cost = paddle.layer.classification_cost(input=predict, label=label)
    parameters = paddle.parameters.create(cost)

    optimizer = paddle.optimizer.Momentum(
        learning_rate=0.1 / 128.0,
        momentum=0.9,
        regularization=paddle.optimizer.L2Regularization(rate=0.0005 * 128))

    trainer = paddle.trainer.SGD(cost=cost,
                                 parameters=parameters,
                                 update_equation=optimizer)

    # init v2 parameter with fluid init
    with open('params_pass_0.tar', 'r') as f:
        v2_fluid_init_parameters(parameters, f)

    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 100 == 0:
                print "Pass %d, Batch %d, Cost %f, %s" % (
                    event.pass_id, event.batch_id, event.cost, event.metrics)
        if isinstance(event, paddle.event.EndPass):
            # save parameters
            with open('params_pass_%d.tar' % event.pass_id, 'w') as f:
                trainer.save_parameter_to_tar(f)

            result = trainer.test(reader=paddle.batch(
                paddle.dataset.mnist.test(), batch_size=128))
            print "Test with Pass %d, Cost %f, %s\n" % (
                event.pass_id, result.cost, result.metrics)

    trainer.train(
        reader=paddle.batch(
            paddle.reader.shuffle(
                paddle.dataset.mnist.train(), buf_size=BATCH_SIZE * 64),
            batch_size=BATCH_SIZE),
        event_handler=event_handler,
        num_passes=PASS)


if __name__ == '__main__':
    main()
