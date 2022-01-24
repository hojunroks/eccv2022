from argparse import Namespace
import yaml

def parse_config(parser, config_file, model_name):
    args = vars(parser.parse_args())
    args = {k: v for k, v in args.items() if v is not None}
    conf = yaml.load(open(config_file), Loader=yaml.FullLoader)
    opt = conf['common']
    opt.update(conf[model_name])
    opt.update(args)
    return Namespace(**opt)