import torch
from misc.action_player import ActionPlayer
from PIL import Image
from torchvision import transforms

transforms = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor(),])


def load_image():
    image = Image.open("resources/collie.jpeg")
    # perform transforms and send to GPU
    image_tensor = transforms(image).cuda()
    return image_tensor


def load_random_on_gpu():
    # creates a tensor directly on a GPU (same size as ImageNet)
    torch.rand(469, 387, device=torch.device("cuda:0"))


def load_random_to_gpu():
    # creates a tensor directly on a GPU (same size as ImageNet)
    # check:
    #   https://towardsdatascience.com/7-tips-for-squeezing-maximum-performance-from-pytorch-ca4a40951259
    #   https://pytorch.org/docs/stable/notes/cuda.html
    torch.rand(469, 387).cuda()


def test_tensor_loading(create_tensor_fn, warmup_cycle=False, repeat=10, action_player=None):
    if action_player is None:
        action_player = ActionPlayer()
    # warmup cycle
    action_name = create_tensor_fn.__name__
    if warmup_cycle:
        for _ in range(30):
            torch.rand(256, 256, device=torch.device("cuda:0"))
        action_name = action_name + "_with_warmup"
    action_player.benchmark(action_name, create_tensor_fn, repeat)
