class Parameters(object):

    def from_config(self, config, experiment_name):
        self.episodes_num = config.getint(experiment_name, 'episodes_num')
        self.track_size = config.getint(experiment_name, 'track_size')
        self.epochs_num = config.getint(experiment_name, 'epochs_num')
        self.learning_rate = config.getfloat(experiment_name, 'learning_rate')
        self.easiness_factor = config.getfloat(experiment_name, 'easiness_factor')
        self.easiness_decay = config.getfloat(experiment_name, 'easiness_decay')
        self.use_ppo =  config.getboolean(experiment_name, 'use_ppo')
        self.eps_clip = config.getfloat(experiment_name, 'eps_clip')
        self.entropy_loss = config.getboolean(experiment_name, 'entropy_loss')
        self.entropy_coef = config.getfloat(experiment_name, 'entropy_coef')
        self.reward_mode = config.get(experiment_name, 'reward_mode')
        self.substract_baseline_reward = config.getboolean(experiment_name, 'substract_baseline_reward')
        self.root_mode = config.get(experiment_name, 'root_mode')
        self.gamma = config.getfloat(experiment_name, 'gamma')
        self.weight_decay = config.getfloat(experiment_name, 'weight_decay')

    def __iter__(self):
        return iter(self.__dict__.items())

    def to_dict(self):
        return dict(self)