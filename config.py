import numpy as np

class Config(object):
    def __init__(self,agent): 
        self.keys = {
                    0:'grade',
                    1:'terrain_type',
                    2:'height_friendly',
                    3:'finish_location',
                    4:'start_location',
                    5:'location',
                    6:'risk',
                    7:'intensity',
                    8:'complexity',
                    9:'intra_difficulty',
                    10:'style',
                    11:'techniques'
                }
        # For calculating the novelty feature
        self.importance_weight_dict = {
            0:0,
            1:0,
            2:0.1,
            3:0.1,
            4:0.1,
            5:0,
            6:0.1,
            7:0.1,
            8:0.1,
            9:0.1,
            10:0.1,
            11:0.2
        }
        self.importance_weights = np.array(list(self.importance_weights_dict.values()))
        # Example grades and techniques
        # 0:'blue',
        # 1:'pink',
        # 2:'purple',
        # 3:'green',
        # 4:'yellow',
        # 5:'orange',
        # 6:'red',
        # 7:'grey',
        # 8:'black',
        # 9:'white'
            # 0:"toe hooking",
            # 1:"heel hooking",
            # 2:"mantle",
            # 3:'gaston',
            # 4:'twist',
            # 5:'cross',
            # 6:'campus',
            # 7:'layback',
            # 8:'stemming',
            # 9:'lock off',
            # 10:'Dyno',
            # 11:'high step',
            # 12:'flagging',
            # 13:'Bumping'
        self.grade_technique_keys = np.array([
            [0,0,0,0,0,0,0,1,1,0,0,0,0,0],
            [0,0,0,0,0,0,0,1,1,0,0,0,0,0],
            [0,0,0,0,0,1,0,1,1,0,0,0,0,0],
            [0,1,0,0,0,1,0,1,1,0,0,0,0,0],
            [1,1,0,0,0,1,0,1,1,0,0,0,0,0],
            [1,1,0,0,1,1,0,1,1,0,0,0,0,0],
            [1,1,0,1,1,1,0,1,1,0,1,0,0,0],
            [1,1,1,1,1,1,0,1,1,0,0,1,0,0],
            [1,1,1,1,1,1,0,1,1,0,0,1,1,0],
            [1,1,1,1,1,1,1,1,1,1,1,1,1,0]
        ])
        # Example terrain types
                # 0:"roof",
                # 1:"overhung",
                # 2:"vertical",
                # 3:"slab"
        self.terrain_technique_keys = np.array([
            [1,1,0,1,1,1,1,1,0,0,0,0,1,0],
            [1,1,0,1,1,1,1,1,0,0,0,0,1,0],
            [1,1,1,1,1,1,1,1,1,1,0,0,1,0],
            [1,1,1,1,1,1,0,1,1,1,1,1,1,0],
        ])
        self.total_routes = 100
        self.num_reset_routes = 1
        self.episodes = 2000
        self.tmax = 100
        self.seed = 1234
        if agent == 'PPO':
            # PPO
            self.name = "PPO"
            self.gae_lambda=0.95
            self.num_agents=20
            self.batch_size=32
            self.gradient_clip=10
            self.SGD_epoch=10
            self.epsilon=0.2
            self.beta=0.01
            self.gamma=0.99
            self.lr = 1e-4
            self.L2 = 0.01
            self.checkpoint_path = 'model_weights/PPO.ckpt'
        elif agent == "ddpg":
            self.name = agent
            self.num_agents = 2
            self.QLR = 0.001
            self.ALR = 0.0001
            self.gamma = 0.99
            self.L2 = 0 # 0.1
            self.tau=0.01 # 0.001
            self.noise_decay=0.99995
            self.gae_lambda = 0.97
            self.clip_norm = 10
            # Buffer
            self.buffer_size = int(1e4)
            self.min_buffer_size = int(1e3)
            self.batch_size = 256
            # Priority Replay
            self.ALPHA = 0.6 # 0.7 or 0.6
            self.START_BETA = 0.5 # from 0.5-1
            self.END_BETA = 1
            # distributional
            self.N_atoms = 51
            self.v_min = -2
            self.v_max = 2
            self.delta_z = (self.v_min - self.v_max) / (self.N_atoms - 1)
            # RouteMuse
            self.action_low= 0 
            self.action_high=1.0
            self.winning_condition = 0.7
            # Training
            self.episodes = 2000
            self.tmax = 100
            self.print_every = 4
            self.SGD_epoch = 1
            self.checkpoint_path = 'model_weights/ddpg.ckpt'