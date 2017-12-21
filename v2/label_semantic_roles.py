import math, os
import numpy as np
import paddle.v2 as paddle
import paddle.v2.dataset.conll05 as conll05
import paddle.v2.evaluator as evaluator
#import paddle.v2.chunk_evaluator  as chunk_evaluator
from paddle.trainer_config_helpers  import *

with_gpu = os.getenv('WITH_GPU', '0') != '0'

word_dict, verb_dict, label_dict = conll05.get_dict()
word_dict_len = len(word_dict)
label_dict_len = len(label_dict)
pred_len = len(verb_dict)

mark_dict_len = 2
word_dim = 32
mark_dim = 5
hidden_dim = 512
depth = 8
default_std = 1 / math.sqrt(hidden_dim) / 3.0
mix_hidden_lr = 1e-3

IS_SPARSE = True
PASS_NUM = 10
BATCH_SIZE = 20


def d_type(size):
    return paddle.data_type.integer_value_sequence(size)


def load_parameter(file_name, h, w):
    with open(file_name, 'rb') as f:
        f.read(16)  # skip header.
        return np.fromfile(f, dtype=np.float32).reshape(h, w)


def db_lstm():
    #8 features
    word = paddle.layer.data(name='word_data', type=d_type(word_dict_len))
    predicate = paddle.layer.data(name='verb_data', type=d_type(pred_len))

    ctx_n2 = paddle.layer.data(name='ctx_n2_data', type=d_type(word_dict_len))
    ctx_n1 = paddle.layer.data(name='ctx_n1_data', type=d_type(word_dict_len))
    ctx_0 = paddle.layer.data(name='ctx_0_data', type=d_type(word_dict_len))
    ctx_p1 = paddle.layer.data(name='ctx_p1_data', type=d_type(word_dict_len))
    ctx_p2 = paddle.layer.data(name='ctx_p2_data', type=d_type(word_dict_len))
    mark = paddle.layer.data(name='mark_data', type=d_type(mark_dict_len))

    emb_para = paddle.attr.Param(name='emb', initial_std=0., is_static=True)
    std_0 = paddle.attr.Param(initial_std=0.)
    std_default = paddle.attr.Param(initial_std=default_std)

    predicate_embedding = paddle.layer.embedding(
        size=word_dim,
        input=predicate,
        param_attr=paddle.attr.Param(
            name='vemb', initial_std=default_std))
    mark_embedding = paddle.layer.embedding(
        size=mark_dim, input=mark, param_attr=std_0)

    word_input = [word, ctx_n2, ctx_n1, ctx_0, ctx_p1, ctx_p2]
    emb_layers = [
        paddle.layer.embedding(
            size=word_dim, input=x, param_attr=emb_para) for x in word_input
    ]
    emb_layers.append(predicate_embedding)
    emb_layers.append(mark_embedding)

    hidden_0 = paddle.layer.mixed(
        size=hidden_dim,
        bias_attr=std_default,
        input=[
            paddle.layer.full_matrix_projection(
                input=emb, param_attr=std_default) for emb in emb_layers
        ])

    lstm_para_attr = paddle.attr.Param(initial_std=0.0, learning_rate=1.0)
    hidden_para_attr = paddle.attr.Param(
        initial_std=default_std, learning_rate=mix_hidden_lr)

    lstm_0 = paddle.layer.lstmemory(
        input=hidden_0,
        act=paddle.activation.Relu(),
        gate_act=paddle.activation.Sigmoid(),
        state_act=paddle.activation.Sigmoid(),
        bias_attr=std_0,
        param_attr=lstm_para_attr)

    #stack L-LSTM and R-LSTM with direct edges
    input_tmp = [hidden_0, lstm_0]

    for i in range(1, depth):
        mix_hidden = paddle.layer.mixed(
            size=hidden_dim,
            bias_attr=std_default,
            input=[
                paddle.layer.full_matrix_projection(
                    input=input_tmp[0], param_attr=hidden_para_attr),
                paddle.layer.full_matrix_projection(
                    input=input_tmp[1], param_attr=lstm_para_attr)
            ])

        lstm = paddle.layer.lstmemory(
            input=mix_hidden,
            act=paddle.activation.Relu(),
            gate_act=paddle.activation.Sigmoid(),
            state_act=paddle.activation.Sigmoid(),
            reverse=((i % 2) == 1),
            bias_attr=std_0,
            param_attr=lstm_para_attr)

        input_tmp = [mix_hidden, lstm]

    feature_out = paddle.layer.mixed(
        size=label_dict_len,
        bias_attr=std_default,
        input=[
            paddle.layer.full_matrix_projection(
                input=input_tmp[0], param_attr=hidden_para_attr),
            paddle.layer.full_matrix_projection(
                input=input_tmp[1], param_attr=lstm_para_attr)
        ], )

    return feature_out


def main():
    paddle.init(use_gpu=with_gpu, trainer_count=1)

    # define network topology
    feature_out = db_lstm()
    target = paddle.layer.data(name='target', type=d_type(label_dict_len))
    crf_cost = paddle.layer.crf(size=label_dict_len,
                                input=feature_out,
                                label=target,
                                param_attr=paddle.attr.Param(
                                    name='crfw',
                                    initial_std=default_std,
                                    learning_rate=mix_hidden_lr))

    crf_dec = paddle.layer.crf_decoding(
        size=label_dict_len,
        input=feature_out,
        #label=target,
        param_attr=paddle.attr.Param(name='crfw'))
    # evaluator.sum(input=crf_dec)
    chunk_evaluator(
        input=crf_dec,
        label=target,
        chunk_scheme="IOB",
        num_chunk_types=int(math.ceil((label_dict_len - 1) / 2.0)))
 
    # create parameters
    parameters = paddle.parameters.create(crf_cost)
    parameters.set('emb', load_parameter(conll05.get_embedding(), 44068, 32))

    # create optimizer
    # optimizer = paddle.optimizer.Momentum(
    #     momentum=0,
    #     learning_rate=2e-2,
    #     regularization=paddle.optimizer.L2Regularization(rate=8e-4),
    #     model_average=paddle.optimizer.ModelAverage(
    #         average_window=0.5, max_average_window=10000), )
    optimizer = paddle.optimizer.Momentum(momentum=0, learning_rate=2e-2)

    trainer = paddle.trainer.SGD(cost=crf_cost,
                                 parameters=parameters,
                                 update_equation=optimizer,
                                 extra_layers=crf_dec)

    train_reader = paddle.batch(
        paddle.dataset.conll05.test(), batch_size=BATCH_SIZE)

    test_reader = paddle.batch(
        paddle.dataset.conll05.test(), batch_size=BATCH_SIZE)

    feeding = {
        'word_data': 0,
        'ctx_n2_data': 1,
        'ctx_n1_data': 2,
        'ctx_0_data': 3,
        'ctx_p1_data': 4,
        'ctx_p2_data': 5,
        'verb_data': 6,
        'mark_data': 7,
        'target': 8
    }

    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 1 == 0:
            # save parameters
                with open('params_pass_%d.tar' % event.pass_id, 'w') as f:
                    trainer.save_parameter_to_tar(f)
                exit(1)
                metrics = [sub.split(".")[1] for sub in event.metrics.keys()]
                metrics_val = event.metrics.values()
                print "Pass %d, Batch %d, Cost %f, %s, %s, %s" % (
                    event.pass_id, event.batch_id, event.cost, 
                    metrics[0] + ":" + str(metrics_val[0]), 
                    metrics[1] + ":" + str(metrics_val[1]), 
                    metrics[2] + ":" + str(metrics_val[2]))
        if isinstance(event, paddle.event.EndPass):
            result = trainer.test(reader=test_reader, feeding=feeding)
            print "\nTest with Pass %d, %s" % (event.pass_id, result.metrics)

    trainer.train(
        reader=train_reader,
        event_handler=event_handler,
        num_passes=1,
        feeding=feeding)

if __name__ == '__main__':
    main()
