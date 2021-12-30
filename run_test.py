import os

train = False

# TRAIN
if train:
    model = 'MCNN'
    mode = 'train'
    lr = [(10 ** -x) for x in range(2, 9)]
    batch = 1
    use_gpu = 'True'
    dataset = 'micc'
    subcategory = 'all'
    num_epochs = 10

    for i in lr:
        args = ('--mode {} --model {} --use_gpu {} --dataset {} '
                '--dataset_subcategory {} --num_epochs {} --lr {} '
                )
        args = args.format(mode, model, use_gpu, dataset, subcategory, num_epochs, i)
        command = 'python main.py {}'.format(args)
        os.system(command)



# TEST / VAL
if not train:

    weights = ['MCNN micc all 2021-12-30 23_14_08.838658_train']

    # weights = 'MCNN micc flow 2021-12-16 17_35_46.014887_train'
    model = 'MCNN'
    mode = 'val'
    batch = 1
    use_gpu = 'True'
    dataset = 'micc'
    subcategory = 'all'

    start = 2045
    save_step = 5
    num_epochs = 2100

    for w in weights:
        for i in range(start + save_step, num_epochs + save_step, save_step):
            if i % 50 == 0:
                save_plots = True
            else:
                save_plots = False

            pretrained_model = '"{}/{}"'.format(w, i)
            args = ('--mode {} --pretrained_model {} --model {} --use_gpu {} --dataset {} '
                    '--dataset_subcategory {} --save_output_plots {}')
            args = args.format(mode, pretrained_model, model, use_gpu, dataset, subcategory, save_plots)
            command = 'python main.py {}'.format(args)
            os.system(command)
