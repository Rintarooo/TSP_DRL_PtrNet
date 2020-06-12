import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config, load_pkl, pkl_parser

class PtrNet2(nn.Module):
	def __init__(self, cfg):
		super().__init__()
		self.Embedding = nn.Linear(cfg.xy, cfg.embed, bias = False)
		self.Encoder = nn.LSTM(input_size = cfg.embed, hidden_size = cfg.hidden, batch_first = True)
		self.LSTMprocess_block = nn.LSTM(input_size = cfg.embed, hidden_size = cfg.hidden, batch_first = True)
		self.Vec = nn.Linear(cfg.hidden, 1, bias = False)
		self.W_q = nn.Linear(cfg.hidden, cfg.hidden, bias = False)
		self.W_ref = nn.Linear(cfg.hidden, cfg.hidden, bias = False)
		self.CEL = nn.CrossEntropyLoss(reduction = 'none')
		self.final2FC = nn.Sequential(
					nn.Linear(cfg.hidden, cfg.hidden, bias = False),
					nn.ReLU(inplace = False),
					nn.Linear(cfg.hidden, 1, bias = False))
		self._initialize_weights(cfg.init_min, cfg.init_max)
	
	def _initialize_weights(self, init_min, init_max):
		for param in self.parameters():
			param.data.uniform_(init_min, init_max)
			
	def forward(self, x, device):
		x = x.to(device)
		batch, city_t, xy = x.size()
		embed_enc_inputs = self.Embedding(x)
		enc_h, (dec_h0, dec_c0) = self.Encoder(embed_enc_inputs, None)
		hidden = enc_h.size(2)
		dec_state = (dec_h0, dec_c0)
		dec_i1 = torch.rand(batch, 1, hidden).to(device)#hidden not embed
		for i in range(city_t):
			dec_h, dec_state = self.LSTMprocess_block(dec_i1, dec_state)
			dec_i1 = self.attending_mechanism(enc_h, dec_h)
		'''	
		- page 5/15 in paper
		critic model architecture detail is out there, "Critic’s architecture for TSP"
		- page 14/15 in paper
		glimpsing more than once with the same parameters 
		made the model less likely to learn and barely improved the results 
		
		dec_h(batch,1,hidden)*FC(hidden,hidden)*FC(hidden,1) -> pred_l(batch,1,1) ->pred_l(batch)
		'''
		pred_l = self.final2FC(dec_h).squeeze(1).squeeze(1)
		return pred_l 
	
	def attending_mechanism(self, enc_h, dec_h):
		'''
		b:batch, c:city_t(time squence), e:embedding_size, h:hidden_size
		enc_h(bch)*W_ref(hh) = u1(bch)
		dec_h(b1h)*W_q(hh) = u2(b1h)
		tanh(bch)*Vec(h1) = u(bc1)
		u(bc1) -> u(bc)
		u(bc) -> a(bc)
		a(bc)*enc_h(bch) = d(bh)     in contrast, actor model-> a(bc)*emb(bce) = d(be)
		d(bh) -> d(b1h)											d(be) -> d(b1e)
		'''
		u1 = self.W_ref(enc_h)
		u2 = self.W_q(dec_h)
		u = self.Vec(torch.tanh(u1 + u2))
		u = torch.squeeze(u, dim = 2)
		a = F.softmax(u, dim = 1)
		d = torch.einsum('bc,bch->bh', a, enc_h)
		d = torch.unsqueeze(d, dim = 1)
		return d
				
if __name__ == '__main__':
	cfg = load_pkl(pkl_parser().path)
	model = PtrNet2(cfg)
	inputs = torch.randn(3,20,2)
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')	
	model = model.to(device)
	pred_l = model(inputs, device)	
	print(pred_l.size())
	print('pred_length:', pred_l)
	
