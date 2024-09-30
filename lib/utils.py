from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger
from ignite.engine import Engine,Events
from torchvision.utils import make_grid 
from ignite.contrib.handlers import tqdm_logger
import torch.nn as nn
import torch
from ignite.handlers.checkpoint import Checkpoint,DiskSaver
from torchvision import datasets
from torchvision import transforms
LOG_IMAGES_EVERY_N_EPOCH = 3

valid_ds = datasets.MNIST('data', train=False, download=False, transform=transforms.ToTensor())
valid_dl = torch.utils.data.DataLoader(valid_ds, batch_size=10, shuffle=False)

def attach_checkpoint_handler(engine: Engine, models: dict, optimizers: dict):
    checkpoint_handler = Checkpoint(
        to_save={**models, **optimizers, 'engine': engine},
        save_handler=DiskSaver(
            dirname='weights',
            create_dir=True,
            require_empty=False
        ),
        n_saved=2,
        filename_prefix='model',
        global_step_transform=lambda engine, event: engine.state.epoch
    )
    engine.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler)

def attach_ignite(
        trainer:Engine,
        model:nn.Module,
    ):
    
    tb_logger = TensorboardLogger(log_dir ='./vae_log')

    tqdm_train = tqdm_logger.ProgressBar().attach(trainer,output_transform=lambda x:x['loss'])

    
    tb_logger.attach_output_handler(
        engine=trainer,
        event_name=Events.EPOCH_COMPLETED,
        tag='train',
        output_transform=lambda x: x
    )

    @torch.no_grad
    def log_generated_images(engine, logger, model, epoch):
        
        batch = next(iter(valid_dl))
        input_images, _ = batch
        input_images = input_images.to('cuda')
        nn_input_images = input_images.view(-1, 784)

        # construct images
        constrcured_imgs ,_,_ = model(nn_input_images)
        constrcured_imgs= constrcured_imgs.view(-1,1,28,28)
        # Prepare the images to be logged
        input_grid = make_grid(input_images, normalize=True, value_range=(0, 1)).cpu()
        const_grid = make_grid(constrcured_imgs, normalize=True, value_range=(0, 1)).cpu()
        # Log the images
        logger.writer.add_image('input_images', input_grid, epoch)
        logger.writer.add_image('generated_images', const_grid, epoch)

        
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_images(engine):
        epoch = engine.state.epoch
        if epoch % LOG_IMAGES_EVERY_N_EPOCH ==0: # for time & disk efficiency  
            log_generated_images(engine, tb_logger, model,epoch)