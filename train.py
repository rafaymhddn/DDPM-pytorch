import torch
import yaml
import argparse
import os
import numpy as np
from tqdm import tqdm
import torchvision
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from model2 import Unet
from noise_schedular import LinearNoiseScheduler
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)


    diffusion_config = config['diffusion_params']
    model_config = config['model_params']
    train_config = config['train_params']

    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])
    
    data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5)),  # -1 to 1
    ])
    
    # Load MNIST dataset
    mnist_train = torchvision.datasets.MNIST(root='.', train=True, download=True, transform=data_transform)
    #mnist_test = torchvision.datasets.MNIST(root='.', train=False, download=True, transform=data_transform)

    mnist_loader = DataLoader(mnist_train, batch_size=train_config['batch_size'], shuffle=True, num_workers=4)
    #mnist_test_loader = DataLoader(mnist_test, batch_size=train_config['batch_size'], shuffle=True, num_workers=4)

    model = Unet(model_config).to(device)
    model.train()

    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])


    if os.path.exists(os.path.join(train_config['task_name'],train_config['ckpt_name'])):
        print('Loading checkpoint as found one')
        model.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                      train_config['ckpt_name']), map_location=device))

    num_epochs = train_config['num_epochs']
    optimizer = Adam(model.parameters(), lr=train_config['lr'])
    criterion = torch.nn.MSELoss(reduction='mean')

    # Run training
    for epoch_idx in range(num_epochs):
        losses = []
        for batch in tqdm(mnist_loader):
            
            
            images, _ = batch  # Unpack the batch tuple
            optimizer.zero_grad()
            im = images.float().to(device)  # Ensure data is moved to the right device
            
            # Sample random noise
            noise = torch.randn_like(im).to(device)
            
            # Sample timestep
            t = torch.randint(0, diffusion_config['num_timesteps'], (im.shape[0],)).to(device)
            
            # Add noise to images according to timestep
            noisy_im = scheduler.add_noise(im, noise, t)

            noise_pred = model(noisy_im, t)
            #print(noise_pred)

            loss = criterion(noise_pred, noise)
            losses.append(loss.item())
            
            with torch.autograd.detect_anomaly():
                loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  
            optimizer.step()
            print('Max/Min noise_pred:', noise_pred.max().item(), noise_pred.min().item())

            print(f'loss: {loss} | losses: {np.mean(losses)}')
        print('Finished epoch:{} | Loss : {:.4f}'.format(
            epoch_idx + 1,
            np.mean(losses),
        ))
        torch.save(model.state_dict(), os.path.join(train_config['task_name'],
                                                    train_config['ckpt_name']))
    
    print('Done Training ...')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ddpm training')
    parser.add_argument('--config', dest='config_path',
                        default='config.yaml', type=str)
    args = parser.parse_args()
    train(args)