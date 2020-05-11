import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from datetime import datetime
from actor import PtrNet1
from critic import PtrNet2

def train(cfg, env, log_path = None):
	date = datetime.now().strftime('%m%d_%H_%M')
	act_model = PtrNet1(cfg)
	if cfg.optim == 'Adam':
		act_optim = optim.Adam(act_model.parameters(), lr = cfg.lr)
	act_lr_scheduler = optim.lr_scheduler.StepLR(act_optim, 
					step_size=cfg.lr_decay_step, gamma=cfg.lr_decay)

	cri_model = PtrNet2()
	if cfg.optim == 'Adam':
		cri_optim = optim.Adam(cri_model.parameters(), lr = cfg.lr)
	cri_lr_scheduler = optim.lr_scheduler.StepLR(cri_optim, 
					step_size = cfg.lr_decay_step, gamma = cfg.lr_decay)
	cri_loss_func = nn.MSEloss()
	
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	act_model, cri_model = act_model.to(device), cri_model.to(device)
	
	for i in tqdm(range(cfg.steps)):
		inputs = env.stack_nodes()
		inputs = inputs.to(device)
		pred_tour, neg_log = act_model(inputs)
		real_l = env.stacl_l(inputs, pred_tour)
		pred_l = cri_model(inputs)
		cri_optim.zero_grad()
		cri_loss = cri_loss_func(pred_l, real_l)
		cri_loss.backward()
		nn.utils.clip_grad_norm_(cri_model.parameters(), max_norm = 1, norm_type = 2)
		'''
		calculate norm of gradient, then modify norm to be less than max_norm value
		'''
		cri_optim.step()
		cri_lr_scheduler.step()
		
		adv = real_l - pred_l
		act_optim.zero_grad()
		act_loss = torch.mean(adv * neg_log)
		act_loss.backward()
		nn.utils.clip_grad_norm_(act_model.parameters(), max_norm = 1, norm_type = 2)
		act_optim.step()
		act_lr_scheduler.step()
		
		if i % 10 == 0:
			print('step:%d, actic loss:%f\n'%(i, act_loss))
			
		if cfg.islogger:
			if i % cfg.log_step == 0:
				if log_path is None:
					log_path = cfg.log_dir + 'test_%s.csv'%(date)#cfg.log_dir = ./Csv/
					with open(log_path, 'w') as f:
						f.write('step,actic loss,critic loss,distance\n')
				else:
					with open(log_path, 'a') as f:
						f.write('%d,%f,%f,%f\n'%(i, act_loss, crit_loss,real_l))
						
				if cfg.issaver:		
					torch.save(act_model.state_dict(), cfg.model_dir + '%s_step%d_act.pt'%(date, i))#'cfg.model_dir = ./Pt/'
					torch.save(cri_model.state_dict(), cfg.model_dir + '%s_step%d_cri.pt'%(date, i))		
