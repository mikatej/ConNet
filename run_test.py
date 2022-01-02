import os

train = True

# TRAIN
if train:
    model = 'CSRNet'
    mode = 'train'
    lr = [(10 ** -x) for x in range(6, 9)]
    batch = 1
    use_gpu = 'True'
    dataset = 'micc'
    subcategory = 'all'
    num_epochs = 10
    augment = 'False'
    gsp = os.path.join('CSRNet_no_augment', 'with_base')

    for i in lr:
        args = ('--mode {} --model {} --use_gpu {} --dataset {} '
                '--dataset_subcategory {} --num_epochs {} --lr {} --augment {} '
                '--group_save_path {}'
                )
        args = args.format(mode, model, use_gpu, dataset, subcategory, num_epochs, i, augment, gsp)
        command = 'python main.py {}'.format(args)
        os.system(command)



# TEST / VAL
if not train:

    # weights = ['MCNN micc all 2021-12-31 01_29_54.042066_train']
    # weights = [
    #             'MCNN micc all 2021-12-31 18_08_14.801855_train',    # 1e-5 lr
    #             'MCNN micc all 2021-12-31 18_19_11.400743_train',
    #             'MCNN micc all 2021-12-31 18_30_09.286582_train',
    #             'MCNN micc all 2021-12-31 18_41_07.568654_train',
    #             'MCNN micc all 2021-12-31 17_20_12.055438_train',
    #             'MCNN micc all 2021-12-31 17_31_13.130339_train',
    #             'MCNN micc all 2021-12-31 17_42_14.488520_train',
    #             'MCNN micc all 2021-12-31 17_53_16.567051_train'
    #         ]   # 1e-8 lr

    # weights = 'MCNN micc flow 2021-12-16 17_35_46.014887_train'
    model = 'MCNN'
    mode = 'val'
    batch = 1
    use_gpu = 'True'
    dataset = 'micc'
    subcategory = 'all'

    start = 0
    save_step = 1
    num_epochs = 2
    gsp = 'CSRNet no_augment'
    gsp_full = os.path.join(gsp, 'no_base')

    for w_index, w in enumerate(weights):
        if w_index == 4:
            gsp_full = os.path.join(gsp, 'with_base')
        for i in range(start + save_step, num_epochs + save_step, save_step):
        # for i in [1, 10]:
            if i == 1 or i == 10:
                save_plots = True
            else:
                save_plots = False

            pretrained_model = '"{}/{}"'.format(w, i)
            args = ('--mode {} --pretrained_model {} --model {} --use_gpu {} --dataset {} '
                    '--dataset_subcategory {} --save_output_plots {} --group_save_path {}')
            args = args.format(mode, pretrained_model, model, use_gpu, dataset, subcategory, save_plots, gsp_full)
            command = 'python main.py {}'.format(args)
            os.system(command)
