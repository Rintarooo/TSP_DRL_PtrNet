import torch
from time import time
from env import Env_tsp
from config import Config, load_pkl, pkl_parser
from search1 import active_search
# ~ from search2 import sampling
	
def search(cfg, env):
	nodes = env.get_nodes(cfg.seed)
	random_tour = env.get_random_tour()
	env.show(nodes, random_tour)
	
	# ~ t1 = time()
	# ~ optimal_tour = env.get_optimal_tour(nodes)
	# ~ env.show(nodes, optimal_tour)
	# ~ t2 = time()
	# ~ print('optimal tour:%dmin %1.2fsec'%((t2-t1)//60, (t2-t1)%60))
	
	t1 = time()
	pred_tour = active_search(cfg, env, nodes)
	env.show(nodes, pred_tour)
	t2 = time()
	print('pointer tour:%dmin %1.2fsec'%((t2-t1)//60, (t2-t1)%60))
	
if __name__ == '__main__':
	cfg = load_pkl(pkl_parser().p)
	env = Env_tsp(cfg)
		
	# ~ inputs = env.stack_nodes()
	# ~ tours = env.stack_random_tours()
	# ~ l = env.stack_l(inputs, tours)
	
	# ~ nodes = env.get_nodes(cfg.seed)
	# ~ random_tour = env.get_random_tour()
	# ~ env.show(nodes)
	
	# ~ print(inputs[0])
	# ~ env.show(inputs[0], random_tour)
	# ~ inputs = env.shuffle_index(inputs)
	# ~ print(inputs[0])
	# ~ env.show(inputs[0], random_tour)
	
	if cfg.mode == 'train':
		train(cfg, env)
		
	elif cfg.mode == 'test':
		search(cfg, env)
