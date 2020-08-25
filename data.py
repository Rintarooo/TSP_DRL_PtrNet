import torch
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

from env import Env_tsp
from config import Config, load_pkl, pkl_parser

class Generator(Dataset):
	def __init__(self, cfg, env):
		# if torch.cuda.is_available():
		# 	torch.cuda.synchronize()
		self.data_list = [env.get_nodes() for i in tqdm(range(cfg.n_samples), disable = False, desc = 'Generate input data')]

	def __getitem__(self, idx):
		return self.data_list[idx]

	def __len__(self):
		return len(self.data_list)

if __name__ == '__main__':
	cfg = load_pkl(pkl_parser().path)
	env = Env_tsp(cfg)
	dataset = Generator(cfg, env)

	data = next(iter(dataset))
	print(data.size())
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	dataloader = DataLoader(dataset, batch_size = cfg.batch, shuffle = True)
	for i, data in enumerate(dataloader):
		print(data.size())
		if i == 0:
			break