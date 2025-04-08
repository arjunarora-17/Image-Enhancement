from zero_dce import Trainer
from PIL import Image
import numpy as np

_model_path = './ZERO_DCE/pretrained-models/model200_dark_faces.pth'
_trainer = Trainer()
_trainer.build_model(pretrain_weights=_model_path)

def enhance_zero_dce_image(pil_image: Image.Image, resize_factor: float = 1.0) -> Image.Image:
    image_pil, enhanced_np = _trainer.infer_cpu(pil_image, image_resize_factor=resize_factor)
    enhanced_np = (enhanced_np * 255).astype(np.uint8)
    enhanced_np = enhanced_np[:, :, ::-1]
    return Image.fromarray(enhanced_np)
