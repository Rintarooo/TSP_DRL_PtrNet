import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
from datetime import datetime
from actor import PtrNet1 

def sampling(cfg, env, test_input):
	test_inputs = test_input.repeat(cfg.batch,1,1)
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	act_model = PtrNet1(cfg)
	if os.path.exists(cfg.act_model_path):	
		act_model.load_state_dict(torch.load(cfg.act_model_path, map_location = device))
	else:
		print('specify pretrained model path')
	act_model = act_model.to(device)
	pred_tours, _ = act_model(test_inputs, device)
	l_batch = env.stack_l_fast(test_inputs, pred_tours)
	index_lmin = torch.argmin(l_batch)
	best_tour = pred_tours[index_lmin]
	return best_tour

def active_search(cfg, env, test_input, log_path = None):
	'''
	active search updates model parameters even during inference on a single input
	test input:(city_t,xy)
	'''
	date = datetime.now().strftime('%m%d_%H_%M')
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	test_inputs = test_input.repeat(cfg.batch,1,1).to(device)
	random_tours = env.stack_random_tours().to(device)
	baseline = env.stack_l_fast(test_inputs, random_tours)
	l_min = baseline[0]
	
	act_model = PtrNet1(cfg)
	if os.path.exists(cfg.act_model_path):	
		act_model.load_state_dict(torch.load(cfg.act_model_path, map_location = device))
		
	if cfg.optim == 'Adam':
		act_optim = optim.Adam(act_model.parameters(), lr = cfg.lr)
	
	act_model = act_model.to(device)
	baseline = baseline.to(device)

	for i in tqdm(range(cfg.steps)):
		'''
		- page 6/15 in papar
		we randomly shuffle the input sequence before feeding it to our pointer network. 
		This increases the stochasticity of the sampling procedure and leads to large improvements in Active Search.
		'''
		test_inputs = test_inputs.to(device)
		shuffle_inputs = env.shuffle(test_inputs)
		pred_shuffle_tours, neg_log = act_model(shuffle_inputs, device)
		pred_tours = env.back_tours(pred_shuffle_tours, shuffle_inputs, test_inputs, device).to(device)
		
		l_batch = env.stack_l_fast(test_inputs, pred_tours)
		
		index_lmin = torch.argmin(l_batch)
		if torch.min(l_batch) != l_batch[index_lmin]:
			raise RuntimeError
		if l_batch[index_lmin] < l_min:
			best_tour = pred_tours[index_lmin]
			print('update best tour, min l(%1.3f -> %1.3f)'%(l_min,l_batch[index_lmin]))
			l_min = l_batch[index_lmin]
			
		adv = l_batch - baseline
		act_optim.zero_grad()
		act_loss = torch.mean(adv * neg_log)
		'''
		adv(batch) = l_batch(batch) - baseline(batch)
		mean(adv(batch) * neg_log(batch)) -> act_loss(scalar) 
		'''
		act_loss.backward()
		nn.utils.clip_grad_norm_(act_model.parameters(), max_norm = 1., norm_type = 2)
		act_optim.step()
		baseline = baseline*cfg.alpha + (1-cfg.alpha)*torch.mean(l_batch, dim = 0)
		print('step:%d/%d, actic loss:%1.3f'%(i, cfg.steps, act_loss.data))
		
		if cfg.islogger:
			if i % cfg.log_step == 0:
				if log_path is None:
					log_path = cfg.log_dir + 'active_search_%s.csv'%(date)#cfg.log_dir = ./Csv/
					with open(log_path, 'w') as f:
						f.write('step,actic loss,minimum distance\n')
				else:
					with open(log_path, 'a') as f:
						f.write('%d,%1.4f,%1.4f\n'%(i, act_loss, l_min))
	return best_tour
	
