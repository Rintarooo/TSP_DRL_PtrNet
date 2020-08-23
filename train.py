import torch
import torch.nn as nn
import torch.optim as optim
import sys
from tqdm import tqdm
from datetime import datetime

from actor import PtrNet1
from critic import PtrNet2
from env import Env_tsp
from config import Config, load_pkl, pkl_parser

torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.benchmark = True

def train_model(cfg, env, log_path = None):
	date = datetime.now().strftime('%m%d_%H_%M')
	act_model = PtrNet1(cfg)
	if cfg.optim == 'Adam':
		act_optim = optim.Adam(act_model.parameters(), lr = cfg.lr)
	if cfg.is_lr_decay:
		act_lr_scheduler = optim.lr_scheduler.StepLR(act_optim, 
						step_size=cfg.lr_decay_step, gamma=cfg.lr_decay)

	cri_model = PtrNet2(cfg)
	if cfg.optim == 'Adam':
		cri_optim = optim.Adam(cri_model.parameters(), lr = cfg.lr)
	if cfg.is_lr_decay:
		cri_lr_scheduler = optim.lr_scheduler.StepLR(cri_optim, 
					step_size = cfg.lr_decay_step, gamma = cfg.lr_decay)
	mse_loss = nn.MSELoss()
	
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	act_model, cri_model = act_model.to(device), cri_model.to(device)
	
	ave_act_loss, ave_cri_loss, ave_L = 0., 0., 0.
	for i in tqdm(range(cfg.steps)):
		inputs = env.stack_nodes()
		inputs = inputs.to(device)
		pred_tour, ll = act_model(inputs, device)
		real_l = env.stack_l(inputs, pred_tour)
		pred_l = cri_model(inputs, device)
		cri_loss = mse_loss(pred_l, real_l.detach())
		cri_optim.zero_grad()
		cri_loss.backward()
		nn.utils.clip_grad_norm_(cri_model.parameters(), max_norm = 1., norm_type = 2)
		cri_optim.step()
		cri_lr_scheduler.step()
		adv = real_l.detach() - pred_l.detach()
		act_loss = (adv * ll).mean()
		act_optim.zero_grad()
		act_loss.backward()
		nn.utils.clip_grad_norm_(act_model.parameters(), max_norm = 1., norm_type = 2)
		act_optim.step()
		act_lr_scheduler.step()

		ave_act_loss += act_loss.item()
		ave_cri_loss += cri_loss.item()
		ave_L += real_l.mean().item()
		
		if i % 10 == 0:
			print('step:%d, actic loss:%1.3f, critic loss:%1.3f, L:%1.3f\n'%(i, ave_act_loss/(i+1), ave_cri_loss/(i+1), ave_L/(i+1)))
		
		if i % cfg.log_step == 0:	
			if cfg.islogger:
				if log_path is None:
					log_path = cfg.log_dir + 'train_%s.csv'%(date)#cfg.log_dir = ./Csv/
					with open(log_path, 'w') as f:
						f.write('step,actic loss,critic loss,average distance\n')
				else:
					with open(log_path, 'a') as f:
						f.write('%d,%1.4f,%1.4f, %1.4f\n'%(i, ave_act_loss/(i+1), ave_cri_loss/(i+1), ave_L/(i+1)))
						
			if cfg.issaver:		
				torch.save(act_model.state_dict(), cfg.model_dir + '%s_step%d_act.pt'%(date, i))#'cfg.model_dir = ./Pt/'

# exponential moving average, not use critic model
def train_model_emv(cfg, env, log_path = None):
	date = datetime.now().strftime('%m%d_%H_%M')
	act_model = PtrNet1(cfg)
	if cfg.optim == 'Adam':
		act_optim = optim.Adam(act_model.parameters(), lr = cfg.lr)
	if cfg.is_lr_decay:
		act_lr_scheduler = optim.lr_scheduler.StepLR(act_optim, 
						step_size=cfg.lr_decay_step, gamma=cfg.lr_decay)

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	act_model = act_model.to(device)

	ave_act_loss, ave_L = 0., 0.
	for i in tqdm(range(cfg.steps)):
		inputs = env.stack_nodes()
		inputs = inputs.to(device)
		pred_tour, ll = act_model(inputs, device)
		real_l = env.stack_l(inputs, pred_tour)
		
		if i == 0:
			L = real_l.detach().mean()
		else:
			L = (L * 0.9) + (0.1 * real_l.detach().mean())

		adv = real_l.detach() - L
		act_loss = (adv * ll).mean()
		act_optim.zero_grad()
		act_loss.backward()
		nn.utils.clip_grad_norm_(act_model.parameters(), max_norm = 2., norm_type = 2)
		act_optim.step()
		act_lr_scheduler.step()

		ave_act_loss += act_loss.item()
		ave_L += real_l.mean().item()
		
		if i % 10 == 0:
			print('step:%d, actic loss:%1.3f, L:%1.3f\n'%(i, ave_act_loss/(i+1), ave_L/(i+1)))
		
		if i % cfg.log_step == 0:	
			if cfg.islogger:
				if log_path is None:
					log_path = cfg.log_dir + 'train_%s.csv'%(date)#cfg.log_dir = ./Csv/
					with open(log_path, 'w') as f:
						f.write('step,actic loss,average distance\n')
				else:
					with open(log_path, 'a') as f:
						f.write('%d,%1.4f, %1.4f\n'%(i, ave_act_loss/(i+1), ave_L/(i+1)))
						
			if cfg.issaver:		
				torch.save(act_model.state_dict(), cfg.model_dir + '%s_step%d_act.pt'%(date, i))#'cfg.model_dir = ./Pt/'


if __name__ == '__main__':
	cfg = load_pkl(pkl_parser().path)
	env = Env_tsp(cfg)

	if cfg.mode == 'train':
		train_model(cfg, env)
	if cfg.mode == 'train_emv':
		train_model_emv(cfg, env)
	else:
		sys.exit('train only, specify train pkl file')
				