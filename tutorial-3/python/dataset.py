import synthetic_data_generator as sim_generator
from torch.utils.data import Dataset

class SimDataset(Dataset):
    def __init__(self, count, transform=None):
        self.input_images, self.target_masks = sim_generator.generate_random_data(192, 192, count=count)
        self.transform = transform

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        image = self.input_images[idx]
        mask = self.target_masks[idx]
        if self.transform:
            image = self.transform(image)

        return [image, mask]