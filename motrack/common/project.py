"""
Set of path constants relative to project source (independent of project location on system).
"""
import os


ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SRC_PATH = os.path.join(ROOT_PATH, 'motrack')
CONFIGS_PATH = os.path.join(ROOT_PATH, 'configs')
OUTPUTS_PATH = os.path.join(ROOT_PATH, 'outputs')
PLAYGROUND_PATH = os.path.join(ROOT_PATH, 'playground')

ASSETS_PATH = '/media/home/motrack-outputs/'
MASTER_PATH = '/media/home/'

TRACKERS_CONFIG_PATH = os.path.join(CONFIGS_PATH, 'trackers')
DANCETRACK_TRACKERS_CONFIG_PATH = os.path.join(TRACKERS_CONFIG_PATH, 'dancetrack')
SPORTSMOT_TRACKERS_CONFIG_PATH = os.path.join(TRACKERS_CONFIG_PATH, 'sportsmot')
