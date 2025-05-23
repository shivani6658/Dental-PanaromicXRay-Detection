import torch

def select_device(device=''):
    """
    Selects the device to run inference on (CPU or CUDA)
    """
    device = str(device).strip().lower().replace('cuda:', '')
    if device == 'cpu' or not torch.cuda.is_available():
        selected_device = torch.device('cpu')
    else:
        selected_device = torch.device('cuda:0')
        print(f'Using CUDA device: {torch.cuda.get_device_name(0)}')

    return selected_device
