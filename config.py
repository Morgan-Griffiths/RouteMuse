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
        self.total_routes = 100
        self.num_reset_routes = 1
        self.episodes = 100
        self.tmax = 80
        self.seed = 1234
        # PPO
        self.gae_lambda=0.95
        self.num_agents=20
        self.batch_size=32
        self.gradient_clip=10
        self.SGD_epoch=10
        self.epsilon=0.2
        self.beta=0.01
        self.gamma=0.99