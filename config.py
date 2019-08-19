import numpy as np

class Config(object):
    def __init__(self): 
        self.keys = {
                    0:'style',
                    1:'techniques',
                    2:'height_friendly',
                    3:'finish_location',
                    4:'start_location',
                    5:'locations',
                    6:'risk',
                    7:'intensity',
                    8:'complexity',
                    9:'intra_difficulty',
                    10:'hold_sets',
                    11:'grade'
                }
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
        #     0:"toe hooking",
        #     1:"heel hooking",
        #     2:"mantle",
        #     3:'gaston',
        #     4:'twist',
        #     5:'cross',
        #     6:'campus',
        #     7:'layback',
        #     8:'stemming'
        self.grade_technique_keys = np.array([
            [0,0,0,0,0,0,0,1,1],
            [0,0,0,0,0,0,0,1,1],
            [0,0,0,0,0,1,0,1,1],
            [0,1,0,0,0,1,0,1,1],
            [1,1,0,0,0,1,0,1,1],
            [1,1,0,0,1,1,0,1,1],
            [1,1,0,1,1,1,0,1,1],
            [1,1,1,1,1,1,0,1,1],
            [1,1,1,1,1,1,1,1,1]
        ])
        # Example terrain types
                # 0:"roof",
                # 1:"overhung",
                # 2:"vertical",
                # 3:"slab"
        self.terrain_technique_keys = np.array([
            [1,1,0,1,1,1,1,1,0],
            [1,1,0,1,1,1,1,1,0],
            [1,1,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,0,1,1],
        ])
        self.total_routes = 100
        self.num_reset_routes = 1
        self.episodes = 2000
        self.tmax = 100
        self.seed = 1234
        self.checkpoint_path = 'model_weights/PPO.ckpt'
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