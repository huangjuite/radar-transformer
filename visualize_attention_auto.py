import torch
import random
import math
from dataset import RadarDataset
from utils import draw_dataset
import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from sensor_msgs.msg import LaserScan
import numpy as np
from pynput.keyboard import Key, Listener

from models import RadarTransformer

rospy.init_node('visualize_radar')

pub = rospy.Publisher('radar', MarkerArray, queue_size=1, latch=True)
publ = rospy.Publisher('laser', LaserScan, queue_size=1, latch=True)
pub_attention = rospy.Publisher(
    'attention', MarkerArray, queue_size=1, latch=True)
pub_anchor = rospy.Publisher('anchor', Marker, queue_size=1, latch=True)
pub_reconstruct = rospy.Publisher(
    'reconstruct', LaserScan, queue_size=1, latch=True)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print(device)

netG = RadarTransformer(
    features=7,
    embed_dim=64,
    nhead=8,
    encoder_layers=6,
    decoder_layers=6,
).to(device)

netG.load_state_dict(torch.load("model/transformer_cgan.pth"))
netG.eval()

# dataset = RadarDataset(remove_oulier=1.2)
dataset = RadarDataset()

r, l_t = None, None
idx_l = 0
layer_id = 0
encoder_attention = None
decoder_attention = None
ls = LaserScan()
ls.header.frame_id = 'map'
ls.header.stamp = rospy.Time.now()
ls.range_max = 100.0
ls.range_min = 0
ls.angle_max = 2.094395
ls.angle_min = -2.094395
ls.angle_increment = 0.017453

lsr = LaserScan()
lsr.header.frame_id = 'map'
lsr.header.stamp = rospy.Time.now()
lsr.range_max = 100.0
lsr.range_min = 0
lsr.angle_max = 2.094395
lsr.angle_min = -2.094395
lsr.angle_increment = 0.017453


def make_text(x, y, z, info, id):
    mk = Marker()
    mk.header.frame_id = 'map'
    mk.header.stamp = rospy.Time.now()
    mk.ns = 'info'
    mk.type = Marker.TEXT_VIEW_FACING
    mk.id = id
    mk.scale.z = 1
    mk.action = Marker.ADD
    mk.pose.position.x = x
    mk.pose.position.y = y
    mk.pose.position.z = z
    mk.pose.orientation.z = 1
    mk.color.a = 1
    mk.color.r = mk.color.g = mk.color.b = 1
    mk.text = info
    return mk


def make_attention(layer, idx, attention, radar, laser):
    at = attention[layer][idx]
    # print('attention: ', idx_l)
    # print('layer: ', layer)
    # print('--------------------')
    angle = lsr.angle_min+idx*lsr.angle_increment
    anchor_p = Point()
    anchor_p.x = laser[idx]*math.cos(angle)
    anchor_p.y = laser[idx]*math.sin(angle)
    anchor_p.z = 0
    anchor_mk = Marker()
    anchor_mk.header.frame_id = 'map'
    anchor_mk.header.stamp = rospy.Time.now()
    anchor_mk.ns = 'anchor'
    anchor_mk.type = Marker.SPHERE
    anchor_mk.scale.x = 0.2
    anchor_mk.scale.y = 0.2
    anchor_mk.scale.z = 0.2
    anchor_mk.action = Marker.ADD
    anchor_mk.pose.position.x = laser[idx]*math.cos(angle)
    anchor_mk.pose.position.y = laser[idx]*math.sin(angle)
    anchor_mk.pose.position.z = 0
    anchor_mk.pose.orientation.x = 0
    anchor_mk.pose.orientation.y = 0
    anchor_mk.pose.orientation.z = 0
    anchor_mk.pose.orientation.w = 1
    anchor_mk.color.a = 1
    anchor_mk.color.r = 1
    anchor_mk.color.g = 1
    anchor_mk.color.b = 1
    pub_anchor.publish(anchor_mk)

    lines = MarkerArray()
    for i, p in enumerate(r):
        p = p[3:6]
        line = Marker()
        line.header.frame_id = 'map'
        line.header.stamp = rospy.Time.now()
        line.ns = 'line'
        line.id = i
        line.scale.x = 0.05
        line.type = Marker.LINE_STRIP
        line.points.append(anchor_p)
        r_p = Point()
        r_p.x = p[0]
        r_p.y = p[1]
        r_p.z = p[2]
        line.points.append(r_p)
        line.color.r = 1
        line.color.g = 1
        line.color.b = 1
        line.color.a = min(at[i]*10, 1)
        lines.markers.append(line)

    t1_mk = make_text(8, 8, 5, 'idx:%d\nlayer:%d' % (idx_l, layer), 0)
    t2_mk = make_text(8, -8, 5, 'idx:%d\nlayer:%d' % (idx_l, layer), 1)

    lines.markers.append(t1_mk)
    lines.markers.append(t2_mk)
    pub_attention.publish(lines)


def make_marker(p, ns, i, id, c, min_v, max_v,  text=False):
    mk = Marker()
    mk.header.frame_id = 'map'
    mk.header.stamp = rospy.Time.now()
    mk.ns = ns
    mk.id = i
    scale = 0.1

    if text:
        mk.type = Marker.TEXT_VIEW_FACING
        mk.text = ns + '-' + str(id)
        scale = 0.05
    else:
        mk.type = Marker.SPHERE
        mk.scale.x = scale
        mk.scale.y = scale

    mk.scale.z = scale

    mk.action = Marker.ADD

    mk.pose.position.x = p[0]
    mk.pose.position.y = p[1]
    mk.pose.position.z = p[2] if not text else p[2]+0.2

    mk.pose.orientation.x = 0
    mk.pose.orientation.y = 0
    mk.pose.orientation.z = 0
    mk.pose.orientation.w = 1

    mk.color.a = 1
    mk.color.r = 1 - (c-min_v)/(max_v-min_v)
    mk.color.g = 1 - (c-min_v)/(max_v-min_v)
    mk.color.b = (c-min_v)/(max_v-min_v)

    return mk


indx = random.randint(0, len(dataset)-1)

# junction
# indx = 110379

# hall
# indx = 35762
# indx = 39714

# parkin
indx = 43030

print('indx: ', indx)
l, r = dataset[indx]

r_t = torch.Tensor(r).to(device)
r_t = torch.unsqueeze(r_t, dim=1)
l_t, encoder_attention, decoder_attention = netG(
    r_t, None, attention_map=True)
l_t = l_t.detach().cpu().numpy()
encoder_attention = encoder_attention.detach().cpu().numpy()[0]
decoder_attention = decoder_attention.detach().cpu().numpy()[0]

min_v = np.min(r[:, 6])
max_v = np.max(r[:, 6])

mks = MarkerArray()

for i, radar in enumerate(r):
    ns = str(int(radar[1]))
    id = str(int(radar[2]))
    v = radar[6]
    p = radar[3:6]
    mk = make_marker(p, ns, 2*i+1, id, c=v, min_v=min_v, max_v=max_v)
    tk = make_marker(p, ns, 2*i, id, c=v, min_v=min_v,
                     max_v=max_v, text=True)
    mks.markers.append(mk)
    mks.markers.append(tk)
pub.publish(mks)

ls.ranges = l.tolist()
publ.publish(ls)

lsr.ranges = l_t.tolist()
pub_reconstruct.publish(lsr)

layer_id = 0
idx_l = 0


def update(e):
    global layer_id
    global idx_l
    make_attention(layer_id, idx_l, decoder_attention, r, l_t)
    layer_id = (layer_id+1) % 6
    if layer_id == 0:
        idx_l = (idx_l+1) % 240
        if idx_l == 0:
            exit()


timer = rospy.Timer(rospy.Duration(0.25), update)

rospy.spin()
