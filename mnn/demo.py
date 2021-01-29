from __future__ import  print_function
import time
import argparse
import numpy as np
import MNN
from dataset import Dataset
nn = MNN.nn
F = MNN.expr

def train_func(net, train_dataloader, opt, num_classes):
    for epoch in range(10):
        t0 = time.time()
        net.train(True)
        print("epoch:", epoch)
        train_dataloader.reset()
        for i in range(train_dataloader.iter_number): # actually, in our full experiment, we only need 3K images using ILSVRC2012 training dataset
            example = train_dataloader.next()
            input_data = example[0]
            output_target = example[1]
            data = input_data[0] # which input, model may have more than one inputs
            label = output_target[0]  # also, model may have more than one outputs

            predict = net.forward(data)
            target = F.one_hot(label, num_classes, 1, 0)
            loss = nn.loss.cross_entropy(predict, target)
            opt.step(loss)

            if i % 10 == 0:
                print("train loss: ", loss.read())

        t1 = time.time()
        cost = t1 - t0
        print("Epoch %d " % epoch, "cost: %.3f s." % cost)
    F.save(net.parameters, "temp.nextword.snapshot")

def demo(model_file, filename):

    train_data = Dataset(filename, is_training=True, max_len=8)
    train_dataloader = MNN.data.DataLoader(train_data, batch_size=64, shuffle=False)
    net = nn.load_module_from_file(model_file, input_names=['X'], output_names=['Softmax'], for_training=True)

    # # turn net to quant-aware-training module
    # nn.compress.train_quant(net, quant_bits=8)

    opt = MNN.optim.ADAM(net, learning_rate=0.0025)
    num_classes = 3502
    train_func(net, train_dataloader, opt, num_classes)

if __name__ == "__main__":
    model_file = "nextword.snapshot.mnn"
    filename = "../data/tmp_word"
    demo(model_file, filename)