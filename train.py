import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torchvision # for data
import model
from matplotlib import pyplot as plt
from PIL import Image
import sys
from dataset import getFacesDataset

# hyper-parameters
epochs = 20
report_every = 16
conv = [1,32,64]
fc = [300,300]

n_classes = 40
size = 100
batch_size = 16

# return normalized dataset divided into two sets
model = model.cnn(size, conv, fc, n_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adagrad(model.parameters(), lr=0.01)

def train(model, optim, db):

	for epoch in range(1, epochs+1):

		train_loader = torch.utils.data.DataLoader(db['train'],batch_size=batch_size, shuffle=True)

		# Update (Train)
		model.train()
		for batch_idx, (data, target) in enumerate(train_loader):

			data, target = Variable(data), Variable(target)
			optimizer.zero_grad()
			output = model(data)
			loss = criterion(output,target)
			loss.backward()
			optimizer.step()

			if batch_idx % report_every == 0:
				print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
					epoch, batch_idx * len(data), len(train_loader.dataset),
					100. * batch_idx / len(train_loader), loss.item()))


		# Evaluate
		model.eval()
		test_loss = float(0)
		correct = 0
		test_loader = torch.utils.data.DataLoader(db['eval'], batch_size=batch_size, shuffle=True)
		for data, target in test_loader:
			with torch.no_grad():
				data= Variable(data)
			target = Variable(target)
			output = model(data)
			test_loss += criterion(output, target).item() # sum up batch loss
			pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
			correct += pred.eq(target.data.view_as(pred)).cpu().sum()

		# test_loss /= len(test_loader.dataset)
		accuracy = float(correct) / len(test_loader.dataset)

		print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.6f})\n'.format(
			test_loss, correct, len(test_loader.dataset),
			accuracy))			

def main():
	dataset = getFacesDataset(int(sys.argv[1]))
	train(model,optim,dataset)

if __name__ == '__main__':
	main()