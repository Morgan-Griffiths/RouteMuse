import numpy as np


class Config(object):
    def __init__(self, agent):
        self.keys = {
            0: 'grade',
            1: 'terrain_type',
            2: 'height_friendly',
            3: 'finish_location',
            4: 'start_location',
            5: 'locations',
            6: 'risk',
            7: 'intensity',
            8: 'complexity',
            9: 'intra_difficulty',
            10: 'style',
            11: 'teaches' # TODO turn into techniques
        }
        self.terrain_types = {'slab':0,'verticle':1,'overhung':2,'roof':3}
        # For calculating the novelty feature
        # 0 means not taken into account, 1 means its the only factor
        # 0.5 means its split 50/50 with the goal
        self.novelty_weights_dist = {
            0: 0,
            1: 0,
            2: 0.3,
            3: 0.3,
            4: 0.3,
            5: 0,
            6: 0.6,
            7: 0.6,
            8: 0.6,
            9: 0.3,
            10: 0.5,
            11: 0.8
        }
        self.novelty_weights = np.array(
            list(self.novelty_weights_dist.values()))
        self.goal_weights = 1 - \
            np.array(list(self.novelty_weights_dist.values()))
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
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
        ])
        # Example terrain types
        # 0:"roof",
        # 1:"overhung",
        # 2:"vertical",
        # 3:"slab"
        self.terrain_technique_keys = np.array([
            [1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0],
            [1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0],
            [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0],
        ])
        self.total_routes = 10
        self.num_reset_routes = 1
        self.episodes = 2000
        self.tmax = self.total_routes
        self.seed = 1234
        self.novelty_type = 'mean'
        self.field = 'grade'
        self.fields_to_ignore = ['Primary Hold Set','Finish Location','Start Location','Style','Length']
        self.goalfields_to_ignore = ['Primary Hold Set','Finish Location','Start Location','Style','Length','Location','Setter','Set Screwed','Stripped','Date','Notes']
        self.name = agent
        if agent == "PPO":
            # PPO
            self.name = "PPO"
            self.gae_lambda = 0.95
            self.num_agents = 1
            self.batch_size = 32
            self.gradient_clip = 10
            self.SGD_epoch = 10
            self.epsilon = 0.2
            self.beta = 0.01
            self.gamma = 0.99
            self.lr = 1e-4
            self.L2 = 0.01
            self.checkpoint_path = 'model_weights/PPO.ckpt'
            self.route_type = 'rl'

        elif agent == "ddpg":
            self.num_agents = 2
            self.QLR = 0.001
            self.ALR = 0.0001
            self.gamma = 0.99
            self.L2 = 0  # 0.1
            self.tau = 0.01  # 0.001
            self.noise_decay = 0.99995
            self.gae_lambda = 0.97
            self.clip_norm = 10
            # Buffer
            self.buffer_size = int(1e4)
            self.min_buffer_size = int(1e3)
            self.batch_size = 256
            # Priority Replay
            self.ALPHA = 0.6  # 0.7 or 0.6
            self.START_BETA = 0.5  # from 0.5-1
            self.END_BETA = 1
            # distributional
            self.N_atoms = 51
            self.v_min = -2
            self.v_max = 2
            self.delta_z = (self.v_min - self.v_max) / (self.N_atoms - 1)
            # RouteMuse
            self.action_low = 0
            self.action_high = 1.0
            self.winning_condition = 0.7
            # Training
            self.episodes = 2000
            self.tmax = 100
            self.print_every = 4
            self.SGD_epoch = 1
            self.checkpoint_path = 'model_weights/ddpg.ckpt'
            self.route_type = 'rl'

        elif agent == 'math':
            self.route_type = 'rl'
        else:
            raise ValueError("agent not understood. Must be 'math' or 'PPO'")

    def set_novelty_weights(self, new_weights):
        self.novelty_weights_dict = new_weights
        self.novelty_weights = np.array(
            list(self.novelty_weights_dict.values()))
        self.goal_weights = 1 - \
            np.array(list(self.novelty_weights_dist.values()))
