
# Condensed Network (ConNet)

Condensed Network (ConNet) is a fast, efficient, and robust crowd counting model. It has two configurations: ConNet-04 with 447k parameters and ConNet-08 with 869k parameters.


## Training / Validating / Testing

To begin training, run main.py on the command line. If validating or testing, specify the `--mode` as shown in the options listed below (e.g. `python main.py --mode val`).

```
python main.py [OPTIONS] [--]
```

Further details can be specified through the options. Alternatively, these options can be directly modified on the `main.py` file. The next subsection lists down all other options that can be specified in this manner.

### Options

    --mode [STR]                        Mode of execution.
                                        Options: train, val, test, pred

#### Dataset Options
    --dataset [STR]                     Dataset to be used for training / testing.
                                        Options: micc, mall
    
    --dataset_subcategory [STR]         (If the chosen dataset is MICC) dataset sequence to be  
                                        used for training / testing. Default: all
                                        Options:  all, flow, groups, queue

#### Training Options 
    --lr [FLOAT]                        Learning rate to be used during training.
    --num_epochs [INT]                  Number of epochs to train.

    --model [STR]                       CNN model to use.
                                        Options: ConNet_04, ConNet_08, CSRNet, MCNN, MARUNet

    --pretrained_model [STR]            File path to pretrained model's weights (.pth or 
                                        .pth.tar file). Default: None

    --use_gpu [BOOLEAN]                 Toggles the use of GPU.
    --loss_log_step [INT]               Logs the loss every N epochs trained. Default: 1
    --model_save_step [INT]             Saves the weights every N epochs trained. Default: 1

#### Testing Options 

    --save_output_plots [BOOL]          Toggles the exporting of side-by-side comparison of
                                        groundtruth and output density maps during test/val

## Compressing

To compress an existing model, run main.py with the `--use_compress` option set to True. Failure to do so will run the program in training mode.

    python main.py --use_compress True [OPTIONS]

Similarly, these options can be directly modified on the `main.py` file.

#### Compression Options 

The compression technique to be used can be specified through the `--compression` option. Further options regarding technique-specific details can be seen in the following subsections.

    --compression [STR]                 Compression technique to use.
                                        Options: musco, skt 
    
#### MUSCO Options 

    --musco_layers_to_compress [STR]    Specific layers to be compressed, separated by comma.
                                        Ex: `frontend.0,frontend.1,frontend.2`. Default: None
                                        If None, all eligible layers for compression will be
                                        selected
    --musco_ft_every [INT]              Number of layers to compress at each compression step.
    --musco_iters [INT]                 Number of times to compress each layer.
    --musco_ft_epochs [INT]             Number of epochs to train during each fine-tuning step.
    --musco_ft_checkpoint [INT]         Checks validation MAE/MSE every N fine-tuning epochs.

#### SKT Options

    --skt_student_chkpt [STR]           File path to student model's weights to continue training
                                        (.pth.tar file). Default: None
    --skt_num_epochs [INT]              Number of epochs to train the student model. Default: 1000
    --skt_lamb_fsp [INT]                Weight of the summation of MSE loss between teacher and 
                                        student features. Default: 0.5
    --skt_lamb_cos [INT]                Weight of the summation of cross entropy loss between
                                        teacher and student features. Default: 0.5
    --skt_print_freq [INT]              Prints updates regarding the losses every N iterations of
                                        every epoch. Default: 200
    --skt_save_freq [INT]               Saves the weights every N epochs trained. Default: 0
                                        If N is 0, the weights will be saved every time the best
                                        MAE or best MSE improves.
