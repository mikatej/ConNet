import os

weights = '2021-12-02 16_53_30.264649_train'
model = 'CSRNet'
mode = 'test'
batch = 1
use_gpu = 'True'
dataset = 'mall'

start = 0
save_step = 1
num_epochs = 5

for i in range(start + save_step, num_epochs + save_step, save_step):
    pretrained_model = '"{}/{}"'.format(weights, i)
    args = ('--mode {} --pretrained_model {} --model {} --use_gpu {} --dataset {}'
            )
    args = args.format(mode, pretrained_model, model, use_gpu, dataset)
    command = 'python main.py {}'.format(args)
    os.system(command)
