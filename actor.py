import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config, load_pkl, pkl_parser
from env import Env_tsp

class Greedy(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, log_p):
		return torch.argmax(log_p, dim = 1).long()

class Categorical(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, log_p):
		return torch.multinomial(log_p.exp(), 1).long().squeeze(1)

# https://github.com/higgsfield/np-hard-deep-reinforcement-learning/blob/master/Neural%20Combinatorial%20Optimization.ipynb
class PtrNet1(nn.Module):
	def __init__(self, cfg):
		super().__init__()
		self.Embedding = nn.Linear(2, cfg.embed, bias = False)
		self.Encoder = nn.LSTM(input_size = cfg.embed, hidden_size = cfg.hidden, batch_first = True)
		self.Decoder = nn.LSTM(input_size = cfg.embed, hidden_size = cfg.hidden, batch_first = True)
		if torch.cuda.is_available():
			self.Vec = nn.Parameter(torch.cuda.FloatTensor(cfg.embed))
			self.Vec2 = nn.Parameter(torch.cuda.FloatTensor(cfg.embed))
		else:
			self.Vec = nn.Parameter(torch.FloatTensor(cfg.embed))
			self.Vec2 = nn.Parameter(torch.FloatTensor(cfg.embed))
		self.W_q = nn.Linear(cfg.hidden, cfg.hidden, bias = True)
		self.W_ref = nn.Conv1d(cfg.hidden, cfg.hidden, 1, 1)
		self.W_q2 = nn.Linear(cfg.hidden, cfg.hidden, bias = True)
		self.W_ref2 = nn.Conv1d(cfg.hidden, cfg.hidden, 1, 1)
		self.dec_input = nn.Parameter(torch.FloatTensor(cfg.embed))
		self._initialize_weights(cfg.init_min, cfg.init_max)
		self.clip_logits = cfg.clip_logits
		self.softmax_T = cfg.softmax_T
		self.n_glimpse = cfg.n_glimpse
		self.city_selecter = {'greedy': Greedy(), 'sampling': Categorical()}.get(cfg.decode_type, None)
	
	def _initialize_weights(self, init_min = -0.08, init_max = 0.08):
		for param in self.parameters():
			nn.init.uniform_(param.data, init_min, init_max)
		
	def forward(self, x, device):
		'''	x: (batch, city_t, 2)
			enc_h: (batch, city_t, embed)
			dec_input: (batch, 1, embed)
			h: (1, batch, embed)
			return: pi: (batch, city_t), ll: (batch)
		'''
		x = x.to(device)
		batch, city_t, _ = x.size()
		embed_enc_inputs = self.Embedding(x)
		embed = embed_enc_inputs.size(2)
		mask = torch.zeros((batch, city_t), device = device)
		enc_h, (h, c) = self.Encoder(embed_enc_inputs, None)
		ref = enc_h
		pi_list, log_ps = [], []
		dec_input = self.dec_input.unsqueeze(0).repeat(batch,1).unsqueeze(1).to(device)
		for i in range(city_t):
			_, (h, c) = self.Decoder(dec_input, (h, c))
			query = h.squeeze(0)
			for i in range(self.n_glimpse):
				query = self.glimpse(query, ref, mask)
			logits = self.pointer(query, ref, mask)	
			log_p = torch.log_softmax(logits, dim = -1)
			next_node = self.city_selecter(log_p)
			dec_input = torch.gather(input = embed_enc_inputs, dim = 1, index = next_node.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, embed))
			
			pi_list.append(next_node)
			log_ps.append(log_p)
			mask += torch.zeros((batch,city_t), device = device).scatter_(dim = 1, index = next_node.unsqueeze(1), value = 1)
			
		pi = torch.stack(pi_list, dim = 1)
		ll = self.get_log_likelihood(torch.stack(log_ps, 1), pi)
		return pi, ll 
	
	def glimpse(self, query, ref, mask, inf = 1e8):
		"""	-ref about torch.bmm, torch.matmul and so on
			https://qiita.com/tand826/items/9e1b6a4de785097fe6a5
			https://qiita.com/shinochin/items/aa420e50d847453cc296
			
				Args: 
			query: the hidden state of the decoder at the current
			(batch, 128)
			ref: the set of hidden states from the encoder. 
			(batch, city_t, 128)
			mask: model only points at cities that have yet to be visited, so prevent them from being reselected
			(batch, city_t)
		"""
		u1 = self.W_q(query).unsqueeze(-1).repeat(1,1,ref.size(1))# u1: (batch, 128, city_t)
		u2 = self.W_ref(ref.permute(0,2,1))# u2: (batch, 128, city_t)
		V = self.Vec.unsqueeze(0).unsqueeze(0).repeat(ref.size(0), 1, 1)
		u = torch.bmm(V, torch.tanh(u1 + u2)).squeeze(1)
		# V: (batch, 1, 128) * u1+u2: (batch, 128, city_t) => u: (batch, 1, city_t) => (batch, city_t)
		u = u - inf * mask
		a = F.softmax(u / self.softmax_T, dim = 1)
		d = torch.bmm(u2, a.unsqueeze(2)).squeeze(2)
		# u2: (batch, 128, city_t) * a: (batch, city_t, 1) => d: (batch, 128)
		return d

	def pointer(self, query, ref, mask, inf = 1e8):
		"""	Args: 
			query: the hidden state of the decoder at the current
			(batch, 128)
			ref: the set of hidden states from the encoder. 
			(batch, city_t, 128)
			mask: model only points at cities that have yet to be visited, so prevent them from being reselected
			(batch, city_t)
		"""
		u1 = self.W_q2(query).unsqueeze(-1).repeat(1,1,ref.size(1))# u1: (batch, 128, city_t)
		u2 = self.W_ref2(ref.permute(0,2,1))# u2: (batch, 128, city_t)
		V = self.Vec2.unsqueeze(0).unsqueeze(0).repeat(ref.size(0), 1, 1)
		u = torch.bmm(V, self.clip_logits * torch.tanh(u1 + u2)).squeeze(1)
		# V: (batch, 1, 128) * u1+u2: (batch, 128, city_t) => u: (batch, 1, city_t) => (batch, city_t)
		u = u - inf * mask
		return u

	def get_log_likelihood(self, _log_p, pi):
		"""	args:
			_log_p: (batch, city_t, city_t)
			pi: (batch, city_t), predicted tour
			return: (batch)
		"""
		log_p = torch.gather(input = _log_p, dim = 2, index = pi[:,:,None])
		return torch.sum(log_p.squeeze(-1), 1)
				
if __name__ == '__main__':
	cfg = load_pkl(pkl_parser().path)
	model = PtrNet1(cfg)
	inputs = torch.randn(3,20,2)	
	pi, ll = model(inputs, device = 'cpu')	
	print('pi:', pi.size(), pi)
	print('log_likelihood:', ll.size(), ll)

	cnt = 0
	for i, k in model.state_dict().items():
		print(i, k.size(), torch.numel(k))
		cnt += torch.numel(k)	
	print('total parameters:', cnt)

	# ll.mean().backward()
	# print(model.W_q.weight.grad)

	cfg.batch = 3
	env = Env_tsp(cfg)
	cost = env.stack_l(inputs, pi)
	print('cost:', cost.size(), cost)
	cost = env.stack_l_fast(inputs, pi)
	print('cost:', cost.size(), cost)
	