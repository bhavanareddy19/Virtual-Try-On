import torchvision, pathlib

def save_grid(tensors, path, nrow=4):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torchvision.utils.save_image((tensors+1)/2., path, nrow=nrow)
