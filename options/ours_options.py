from typing_extensions import Required
from .base_options import BaseOptions


class OursOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        #parser.add_argument('--model', type=str, default='galnet', help='saves results here.')
        parser.add_argument('--results_dir', type=str, default='./result/', help='saves results here.')
        parser.add_argument('--phase', type=str, default='ours', help='train, val, test')
        self.isTrain = False
        return parser

