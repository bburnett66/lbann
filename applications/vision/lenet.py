import argparse
import lbann
import lbann.models
import data.mnist
import lbann.contrib.args
import lbann.contrib.launcher

# ----------------------------------
# Command-line arguments
# ----------------------------------

desc = ('Train LeNet on MNIST data using LBANN.')
parser = argparse.ArgumentParser(description=desc)
lbann.contrib.args.add_scheduler_arguments(parser)
parser.add_argument(
    '--job-name', action='store', default='lbann_lenet', type=str,
    help='scheduler job name (default: lbann_lenet)')
args = parser.parse_args()

# ----------------------------------
# Construct layer graph
# ----------------------------------

# Input data
images = lbann.Input(data_field='samples')
labels = lbann.Input(data_field='labels')

# LeNet
preds = lbann.models.LeNet(10)(images) #MNIST is 10 input labels
probs = lbann.Softmax(preds)
layers = list(lbann.traverse_layer_graph([images, labels]))

# Loss function and accuracy
loss = lbann.CrossEntropy(probs, labels)
acc = lbann.CategoricalAccuracy(probs, labels)

# ----------------------------------
# Setup experiment
# ----------------------------------

# Setup model
mini_batch_size = 64
num_epochs = 20
model = lbann.Model(num_epochs,
                    layers=layers,
                    objective_function=lbann.ObjectiveFunction([loss]),
                    metrics=[lbann.Metric(acc, name='accuracy', unit='%')],
                    callbacks=[lbann.CallbackPrintModelDescription(),
                               lbann.CallbackPrint(),
                               lbann.CallbackTimer()])

# Setup optimizer
opt = lbann.SGD(learn_rate=0.01, momentum=0.9)

# Setup data reader
data_reader = data.mnist.make_data_reader()

# Setup trainer
trainer = lbann.Trainer(mini_batch_size=mini_batch_size)

# ----------------------------------
# Run experiment
# ----------------------------------
kwargs = lbann.contrib.args.get_scheduler_kwargs(args)
lbann.contrib.launcher.run(trainer, model, data_reader, opt,
                           job_name=args.job_name,
                           **kwargs)
