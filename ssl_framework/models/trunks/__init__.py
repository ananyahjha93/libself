from ssl_framework.models.trunks.jigsaw_alexnet import JigsawAlexNetTrunk
from ssl_framework.models.trunks.resnet_50 import Resnet_50
from ssl_framework.models.trunks.resnet_18 import Resnet_18
from ssl_framework.models.trunks.utils import parse_out_keys_arg

TRUNKS = {"jigsaw_alexnet": JigsawAlexNetTrunk,
          "resnet_50": Resnet_50,
          "resnet_18": Resnet_18}