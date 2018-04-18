# Basic
import sys
import os
import argparse
#import math
#from pathlib import Path

# Science
import numpy as np
import pandas as pd
from scipy.io import wavfile

# Pytorch
import torch
from torch import nn
from torch import optim
from torch import Tensor
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset



################################################################## ARGUMENTS

parser = argparse.ArgumentParser(description='Specify some hyper parametres:')


# Training
parser.add_argument("-b",   help="Batch size",             default=1,      type=int)
parser.add_argument("-l",   help="Learning rate",          default=0.001,  type=float) # 0.0001
parser.add_argument("-m",   help="Momentum",               default=0.9,    type=float)
parser.add_argument("-w",   help="Weight decay",           default=0.0005, type=float)
parser.add_argument("-e",   help="Max epochs",             default=5,      type=int)   # 50
parser.add_argument("-f",   help="Folds",                  default=10,     type=int)

# Hardware
parser.add_argument("-cpu", help="Do not use cuda",        action="store_true")
parser.add_argument('-gpus', help='Use multiple GPUs',     action='store_true')

# Data
parser.add_argument("-vp",  help="Validation percentage",  default=0.05,  type=float)
parser.add_argument('-nw',  help="Number of workers",      default=0,     type=int)   #4

# Model
parser.add_argument("-pt",  help="Load pre-trained model", action="store_true")
parser.add_argument("-cp",  help="Use check_point",        action="store_true")

args = parser.parse_args()

batch_size      = args.b
learning_rate   = args.l
momentum        = args.m
weight_decay    = args.w
num_epochs      = args.e
folds           = args.f
gpu             = not args.cpu
multiple_gpus   = args.gpus
val_percent     = args.vp
num_workers     = args.nw
check_point     = args.cp
trained_model   = args.pt


# Audio
sampling_rate  = 16000
audio_duration = 2
use_mfcc       = False
n_mfcc         = 20

audio_length = sampling_rate * audio_duration
if use_mfcc:
	dim = (n_mfcc, 1 + int(np.floor(audio_length/512)), 1)
else:
	dim = (audio_length, 1)





######################################################################### DATA

class Freesound(Dataset):

	def __init__(self, mode="train"):

		# setting directories for data
		data_root = "data" #../input

		self.mode = mode
		if self.mode is "train":
			self.data_dir = os.path.join(data_root, "audio_train")
			self.csv_file = pd.read_csv(os.path.join(data_root, "train.csv"))
		elif self.mode is "test":
			self.data_dir = os.path.join(data_root, "audio_test")
			self.csv_file = pd.read_csv(os.path.join(data_root, "sample_submission.csv"))

		# dict for mapping class names into indices. can be obtained by 
		# {cls_name:i for i, cls_name in enumerate(csv_file["label"].unique())}
		self.classes = {'Acoustic_guitar': 38, 'Applause': 37, 'Bark': 19, 'Bass_drum': 21, 'Burping_or_eructation': 28, 'Bus': 22, 'Cello': 4, 'Chime': 20, 'Clarinet': 7, 'Computer_keyboard': 8, 'Cough': 17, 'Cowbell': 33, 'Double_bass': 29, 'Drawer_open_or_close': 36, 'Electric_piano': 34, 'Fart': 14, 'Finger_snapping': 40, 'Fireworks': 31, 'Flute': 16, 'Glockenspiel': 3, 'Gong': 26, 'Gunshot_or_gunfire': 6, 'Harmonica': 25, 'Hi-hat': 0, 'Keys_jangling': 9, 'Knock': 5, 'Laughter': 12, 'Meow': 35, 'Microwave_oven': 27, 'Oboe': 15, 'Saxophone': 1, 'Scissors': 24, 'Shatter': 30, 'Snare_drum': 10, 'Squeak': 23, 'Tambourine': 32, 'Tearing': 13, 'Telephone': 18, 'Trumpet': 2, 'Violin_or_fiddle': 39, 'Writing': 11}

		self.transform = transforms.Compose([
			lambda x: x.astype(np.float32) / np.max(x),
			lambda x: np.expand_dims(x, axis=0),
			lambda x: Tensor(x)
		])

	def __len__(self):
		return self.csv_file.shape[0] 

	def __getitem__(self, idx):
		filename = self.csv_file["fname"][idx]

		rate, data = wavfile.read(os.path.join(self.data_dir, filename))
		#print("MAX: ",np.max(data),"    MIN: ",np.min(data))

		if self.transform is not None:
			data = self.transform(data)

		if self.mode is "train":
			label = self.classes[self.csv_file["label"][idx]]
			return data, label

		elif self.mode is "test":
			return data



def audio_norm(data):
	max_data = np.max(data)
	min_data = np.min(data)
	data = (data-min_data)/(max_data-min_data+1e-6)
	return data-0.5


############################################################################# MODEL

class CNN_1D(nn.Module):
	def __init__(self):
		super(CNN_1D, self).__init__()
		self.c1 = nn.Sequential(
			nn.Conv1d(1, 64, kernel_size=9, padding=4),
			nn.BatchNorm1d(64),
			nn.ReLU(),          # LeakyReLU
			nn.MaxPool1d(2))    # AvgPool1d
		self.c2 = nn.Sequential(
			nn.Conv1d(64, 128, kernel_size=9, padding=4),
			nn.BatchNorm1d(128),
			nn.ReLU(),
			nn.MaxPool1d(2))
		self.c3 = nn.Sequential(
			nn.Conv1d(128, 128, kernel_size=9, padding=4),
			nn.BatchNorm1d(128),
			nn.ReLU(),
			nn.MaxPool1d(2))
		self.c4 = nn.Sequential(
			nn.Conv1d(128, 41, kernel_size=9, padding=4),
			nn.MaxPool1d(2),
			nn.Softmax(dim=1))
		#self.gru = nn.GRU(128, 128, dropout=0.01)
		#self.fc = nn.Linear(128, 41)

	def forward(self, x):
		print(x.shape)
		out = self.c1(x)
		print(out.shape)
		out = self.c2(out)
		print(out.shape)
		out = self.c3(out)
		print(out.shape)
		out = self.c4(out)
		print(out.shape)
		#out = self.gru(out)
		#out = self.fc(out)
		return out


############################################################################# TRAIN


def train(model, dataloader):

	"""
	1D Conv model
	- Optimizer: Adam
	- Learning rate: 0.001
	- Loss: Binary crossentropy
	"""

	# Loss and Optimizer
	#criterion = nn.CrossEntropyLoss() 
	criterion = nn.BCELoss()
	optimizer = optim.Adam(model.parameters(), lr=learning_rate)

	# Training
	for epoch in range(num_epochs):
		for i, (data, label) in enumerate(dataloader):
			
			data = Variable(data)
			#if gpu: data = data.cuda()
			print(label.numpy()[0])

			# Forward
			optimizer.zero_grad()
			output = model(data)

			# Prepare target with outut shape
			target = torch.zeros(output.shape)
			target[:, 0] = label.numpy()[0]
			target = Variable(target)
			#if gpu: target = target.cuda()

			# Backward
			loss = criterion(output, target)
			loss.backward()
			optimizer.step()

			#if (i+1) % 100 == 0:
			print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))

		# Decaying Learning Rate
		#if (epoch+1) % 20 == 0:
		#	lr /= 3
		#	optimizer = torch.optim.Adam(model.parameters(), lr=lr) 



if __name__ == '__main__':

	dataset    = Freesound(mode="train")
	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
	# todo: multiprocessing, padding data

	model = CNN_1D()

	train(model, dataloader)


	a = torch.FloatTensor(5, 7)
	a.fill_(3.5)