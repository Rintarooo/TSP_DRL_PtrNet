import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config, load_pkl, pkl_parser

class PtrNet1(nn.Module):
	def __init__(self, cfg):
		super().__init__()
		self.Embedding = nn.Linear(cfg.xy, cfg.embed, bias = False)
		self.Encoder = nn.LSTM(input_size = cfg.embed, hidden_size = cfg.hidden, batch_first = True)
		self.Decoder = nn.LSTM(input_size = cfg.embed, hidden_size = cfg.hidden, batch_first = True)
		self.Vec = nn.Linear(cfg.hidden, 1, bias = False)
		self.W_q = nn.Linear(cfg.hidden, cfg.hidden, bias = False)
		self.W_ref = nn.Linear(cfg.hidden, cfg.hidden, bias = False)
		self.CEL = nn.CrossEntropyLoss(reduction = 'none')
		'''
		This criterion combines "log_softmax" and "nll(negative log likelihood)_loss" in a single function
		'''
		self._initialize_weights(cfg.init_min, cfg.init_max)
		self.clip_logits = cfg.clip_logits
		self.softmax_T = cfg.softmax_T
	
	def _initialize_weights(self, init_min, init_max):
		for param in self.parameters():
			param.data.uniform_(init_min, init_max)
			
	def forward(self, x, device):
		x = x.to(device)
		batch, city_t, xy = x.size()
		embed_enc_inputs = self.Embedding(x)
		embed = embed_enc_inputs.size(2)
		already_played_action_mask = torch.zeros(batch, city_t).to(device)
		enc_h, (dec_h0, dec_c0) = self.Encoder(embed_enc_inputs, None)
		dec_state = (dec_h0, dec_c0)
		pred_tour_list, neg_log = [], 0
		dec_i1 = torch.rand(batch, 1, embed).to(device)
		for i in range(city_t):
			dec_h, dec_state = self.Decoder(dec_i1, dec_state)
			logits, probs, dec_i1 = self.pointing_mechanism(
								enc_h, dec_h, embed_enc_inputs, already_played_action_mask)
			next_city_index = torch.argmax(logits, dim=1)
			pred_tour_list.append(next_city_index)
			neg_log += self.CEL(input = logits, target = next_city_index)
			'''
			input(batch, class), target(batch);target value:0 ~ class-1)
			logits(batch,city_t) -> next_city_index(batch);value:0 ~ 20
			neg_log is for calculating log part of gradient policy equation 
			'''
			already_played_action_mask += torch.zeros(batch,city_t).to(device).scatter_(
							dim = 1, index = torch.unsqueeze(next_city_index, dim = 1),value = 1)
		
		pred_tour = torch.stack(pred_tour_list, dim = 1)
		'''
		a list of tensors -> pred_tour;tensor(batch,city_t)
		# ~ pred_tour = torch.LongTensor(pred_tour_list)#pred_tour = torch.tensor(pred_tour_list)
		# ~ for i in range(city_t):
			# ~ neg_log += self.CEL(input = logits, target = pred_tour[:,i])
		'''
		return pred_tour, neg_log 
	
	def pointing_mechanism(self, enc_h, dec_h, embed_enc_inputs, 
						already_played_action_mask, infinity = 1e7):
		'''
		-ref about torch.bmm, torch.matmul and so on
		https://qiita.com/tand826/items/9e1b6a4de785097fe6a5
		https://qiita.com/shinochin/items/aa420e50d847453cc296
		
		b:batch, c:city_t(time squence), e:embedding_size, h:hidden_size
		enc_h(bch)*W_ref(hh) = u1(bch)
		dec_h(b1h)*W_q(hh) = u2(b1h)
		tanh(bch)*Vec(h1) = u(bc1)
		u(bc1) -> u(bc)-already(bc) = u(bc)
		u(bc) -> a(bc)
		a(bc)*emb(bce) = d(be)
		d(be) -> d(b1e)
		'''
		u1 = self.W_ref(enc_h)
		u2 = self.W_q(dec_h)
		u = self.Vec(self.clip_logits * torch.tanh(u1 + u2))
		u = torch.squeeze(u, dim = 2) - infinity * already_played_action_mask
		'''
		mask: model only points at cities that have yet to be visited, so prevent them from being reselected
		'''
		a = F.softmax(u / self.softmax_T, dim = 1)
		d = torch.einsum('bc,bce->be', a, embed_enc_inputs)
		d = torch.unsqueeze(d, dim = 1)
		return u,a,d
				
if __name__ == '__main__':
	cfg = load_pkl(pkl_parser().path)
	model = PtrNet1(cfg)
	inputs = torch.randn(3,20,2)	
	pred_tour, neg_log = model(inputs, device = 'cpu')	
	print(pred_tour.size())
	print('pred_tour:', pred_tour)
	print(neg_log.size())
	print('neg_log:', neg_log)
