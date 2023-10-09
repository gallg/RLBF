import os
import configparser
import json

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


class CustomConfigParser(configparser.ConfigParser):

    def getlist(self, section, option):
        return json.loads(self.get(section, option))


settings = CustomConfigParser(allow_no_value=True)
settings.read(os.path.join(ROOT_DIR, 'settings.conf'))

# real-time processing settings;
block_size = settings.getint('REAL_TIME', 'block_size')
rest_size = settings.getint('REAL_TIME', 'rest_size')
hrf_stimulus_onset = settings.getint('REAL_TIME', 'hrf_stimulus_onset')
repetition_time = settings.getfloat('REAL_TIME', 'tr')

# ------------------------------------------------

# reinforcement learning settings;
num_bins_per_observation = settings.getint('Q_LEARNING', 'n_bins')
q_table_noise_mean = settings.getfloat('Q_LEARNING', 'q_mean')
q_table_noise_sigma = settings.getfloat('Q_LEARNING', 'q_sigma')
kernel_size = settings.getint('Q_LEARNING', 'kernel_size')
kernel_sigma = settings.getfloat('Q_LEARNING', 'kernel_sigma')

learning_rate = settings.getfloat('Q_LEARNING', 'learning_rate')
temperature = settings.getfloat('Q_LEARNING', 'temperature')
min_temperature = settings.getfloat('Q_LEARNING', 'min_temperature')
max_temperature = settings.getfloat('Q_LEARNING', 'max_temperature')
reduce_temperature = settings.getboolean('Q_LEARNING', 'reduce_temperature')
decay_rate = settings.getfloat('Q_LEARNING', 'decay_rate')
