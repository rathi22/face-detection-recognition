import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class FacesDataset(Dataset):
	def __init__(self, root_dir, size):
		self.root_dir = root_dir
		self.size = size

	def __len__(self):
		return 40*self.size

	def __getitem__(self, idx):
		if self.root_dir == "train/": 
			label = idx//self.size + 1
			num = idx%self.size + 1
		else:
			label = idx//self.size + 1
			num = idx%self.size + 11-self.size
		img = Image.open(self.root_dir+"s"+str(label)+"/"+str(num)+".png")
		trans = torchvision.transforms.ToTensor()
		sample = (trans(img),torch.tensor(label-1))
		return sample


def getFacesDataset(frac):
	train = FacesDataset(root_dir="train/",size=frac)
	test = FacesDataset(root_dir="test/",size=10-frac)
	return {'train': train, 'eval': test} 
