from argparse import Namespace
import yaml
import numpy as np

def parse_config(parser, config_file, model_name):
    args = vars(parser.parse_args())
    args = {k: v for k, v in args.items() if v is not None}
    conf = yaml.load(open(config_file), Loader=yaml.FullLoader)
    opt = conf['common']
    opt.update(conf[model_name])
    opt.update(args)
    return Namespace(**opt)


def gaussian(x, a, mu, sigma):
    return a*np.exp(-(x-mu)**2/(2*sigma**2))