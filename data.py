import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

from env import Env_tsp
from config import Config, load_pkl, pkl_parser

class Generator(Dataset):
	def __init__(self, cfg, env):
		self.data = env.get_batch_nodes(cfg.n_samples)
		
	def __getitem__(self, idx):
		return self.data[idx]

	def __len__(self):
		return self.data.size(0)

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
