import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config, load_pkl, pkl_parser

class PtrNet2(nn.Module):
	def __init__(self, cfg):
		super().__init__()
		self.Embedding = nn.Linear(2, cfg.embed, bias = False)
		self.Encoder = nn.LSTM(input_size = cfg.embed, hidden_size = cfg.hidden, batch_first = True)
		self.Decoder = nn.LSTM(input_size = cfg.embed, hidden_size = cfg.hidden, batch_first = True)
		if torch.cuda.is_available():
			self.Vec = nn.Parameter(torch.cuda.FloatTensor(cfg.embed))
		else:
			self.Vec = nn.Parameter(torch.FloatTensor(cfg.embed))
		self.W_q = nn.Linear(cfg.hidden, cfg.hidden, bias = True)
		self.W_ref = nn.Conv1d(cfg.hidden, cfg.hidden, 1, 1)
		# self.dec_input = nn.Parameter(torch.FloatTensor(cfg.embed))
		self.final2FC = nn.Sequential(
					nn.Linear(cfg.hidden, cfg.hidden, bias = False),
					nn.ReLU(inplace = False),
					nn.Linear(cfg.hidden, 1, bias = False))
		self._initialize_weights(cfg.init_min, cfg.init_max)
		self.n_glimpse = cfg.n_glimpse
		self.n_process = cfg.n_process
	
	def _initialize_weights(self, init_min = -0.08, init_max = 0.08):
		for param in self.parameters():
			nn.init.uniform_(param.data, init_min, init_max)
			
	def forward(self, x, device):
		'''	x: (batch, city_t, 2)
			enc_h: (batch, city_t, embed)
			query(Decoder input): (batch, 1, embed)
			h: (1, batch, embed)
			return: pred_l: (batch)
		'''
		x = x.to(device)
		batch, city_t, xy = x.size()
		embed_enc_inputs = self.Embedding(x)
		embed = embed_enc_inputs.size(2)
		enc_h, (h, c) = self.Encoder(embed_enc_inputs, None)
		ref = enc_h
		# ~ query = h.permute(1,0,2).to(device)# query = self.dec_input.unsqueeze(0).repeat(batch,1).unsqueeze(1).to(device)
		query = h[-1]
		# ~ process_h, process_c = [torch.zeros((1, batch, embed), device = device) for _ in range(2)]
		for i in range(self.n_process):
			# ~ _, (process_h, process_c) = self.Decoder(query, (process_h, process_c))
			# ~ _, (h, c) = self.Decoder(query, (h, c))
			# ~ query = query.squeeze(1)
			for i in range(self.n_glimpse):
				query = self.glimpse(query, ref)
				# ~ query = query.unsqueeze(1)
		'''	
		- page 5/15 in paper
		critic model architecture detail is out there, "Critic’s architecture for TSP"
		- page 14/15 in paper
		glimpsing more than once with the same parameters 
		made the model less likely to learn and barely improved the results 
		
		query(batch,hidden)*FC(hidden,hidden)*FC(hidden,1) -> pred_l(batch,1) ->pred_l(batch)
		'''
		pred_l = self.final2FC(query).squeeze(-1).squeeze(-1)
		return pred_l 
	
	def glimpse(self, query, ref, infinity = 1e8):
		"""	Args: 
			query: the hidden state of the decoder at the current
			(batch, 128)
			ref: the set of hidden states from the encoder. 
			(batch, city_t, 128)
		"""
		u1 = self.W_q(query).unsqueeze(-1).repeat(1,1,ref.size(1))# u1: (batch, 128, city_t)
		u2 = self.W_ref(ref.permute(0,2,1))# u2: (batch, 128, city_t)
		V = self.Vec.unsqueeze(0).unsqueeze(0).repeat(ref.size(0), 1, 1)
		u = torch.bmm(V, torch.tanh(u1 + u2)).squeeze(1)
		# V: (batch, 1, 128) * u1+u2: (batch, 128, city_t) => u: (batch, 1, city_t) => (batch, city_t)
		a = F.softmax(u, dim = 1)
		d = torch.bmm(u2, a.unsqueeze(2)).squeeze(2)
		# u2: (batch, 128, city_t) * a: (batch, city_t, 1) => d: (batch, 128)
		return d

if __name__ == '__main__':
	cfg = load_pkl(pkl_parser().path)
	model = PtrNet2(cfg)
	inputs = torch.randn(3,20,2)
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')	
	model = model.to(device)
	pred_l = model(inputs, device)	
	print('pred_length:', pred_l.size(), pred_l)
	
	cnt = 0
	for i, k in model.state_dict().items():
		print(i, k.size(), torch.numel(k))
		cnt += torch.numel(k)	
	print('total parameters:', cnt)

	# pred_l.mean().backward()
	# print(model.W_q.weight.grad)
