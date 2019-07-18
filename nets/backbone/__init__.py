from . import darknet
from . import shufflenet

backbone_fn = {
    "darknet_21": darknet.darknet21,
    "darknet_53": darknet.darknet53,
    "shufflenet_2": shufflenet.shufflenet2
}
