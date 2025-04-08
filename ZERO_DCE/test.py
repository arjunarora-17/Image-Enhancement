import random
from glob import glob
from zero_dce import (
    Trainer, plot_result
)

trainer = Trainer()
trainer.build_model(pretrain_weights='./pretrained-models/model200_dark_faces.pth')

image_files = glob('./examples/*.png')
random.shuffle(image_files)
print(image_files)

for image_file in image_files[:5]:
    # Testing on resized images because I'm gareeb
    print("Image file - ", image_file)
    image, enhanced = trainer.infer_cpu(image_file, image_resize_factor=1)
    plot_result(image, enhanced)