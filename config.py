
#### SELF PLAY
EPISODES = 2 # 30  # 나중에 늘리기 
MCTS_SIMS = 16 # 50
MEMORY_SIZE = 100 # 30000
TURNS_UNTIL_TAU0 = 10 # turn on which it starts playing deterministically
CPUCT = 1
EPSILON = 0.2
ALPHA = 0.8
THREADS = 8


BOARD_SIZE = 7


#### RETRAINING
BATCH_SIZE = 64 # 256
EPOCHS = 1
REG_CONST = 0.0001
LEARNING_RATE = 0.1
MOMENTUM = 0.9
TRAINING_LOOPS = 10
'''
HIDDEN_CNN_LAYERS = [
	{'filters':75, 'kernel_size': (4,4)}
	 , {'filters':75, 'kernel_size': (4,4)}
	 , {'filters':75, 'kernel_size': (4,4)}
	 , {'filters':75, 'kernel_size': (4,4)}
	 , {'filters':75, 'kernel_size': (4,4)}
	 , {'filters':75, 'kernel_size': (4,4)}
	]
'''

HIDDEN_CNN_LAYERS = [
	{'filters':75, 'kernel_size': (3,3)}
	 , {'filters':75, 'kernel_size': (3,3)}
	 , {'filters':75, 'kernel_size': (3,3)}
	 , {'filters':75, 'kernel_size': (3,3)}
	 , {'filters':75, 'kernel_size': (3,3)}
	 , {'filters':75, 'kernel_size': (3,3)}
	]

#### EVALUATION
EVAL_EPISODES = 2 #20
SCORING_THRESHOLD = 1.3
