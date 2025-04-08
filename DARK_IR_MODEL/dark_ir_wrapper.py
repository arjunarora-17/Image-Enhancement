# darkir_wrapper.py

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import os
from archs import DarkIR  # Ensure this import is correct based on your project structure

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define model path
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'DarkIR_384.pt')

# Load model
model = DarkIR(
    img_channel=3, 
    width=32,
    middle_blk_num_enc=2,
    middle_blk_num_dec=2,
    enc_blk_nums=[1, 2, 3],
    dec_blk_nums=[3, 1, 1],
    dilations=[1, 4, 9],
    extra_depth_wise=True
)
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint['params'])
model.to(device).eval()

# Define transforms
pil_to_tensor = transforms.ToTensor()
tensor_to_pil = transforms.ToPILImage()

# Helper: Pad tensor to multiple of 8
def pad_tensor(tensor, multiple=8):
    _, _, H, W = tensor.shape
    pad_h = (multiple - H % multiple) % multiple
    pad_w = (multiple - W % multiple) % multiple
    return F.pad(tensor, (0, pad_w, 0, pad_h), value=0)

# Main enhancement function
def enhance_darkir_image(pil_img: Image.Image) -> Image.Image:
    tensor = pil_to_tensor(pil_img).unsqueeze(0).to(device)
    _, _, H, W = tensor.shape
    tensor = pad_tensor(tensor)

    with torch.no_grad():
        output = model(tensor, side_loss=False)

    output = torch.clamp(output, 0., 1.)
    output = output[:, :, :H, :W].squeeze(0)
    return tensor_to_pil(output)
