import torch
import random
from dataset import RadarDataset, RadarDatasetClass
from utils import draw_dataset
import rospy
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import LaserScan
import numpy as np



dataset = RadarDataset()
indx = 2872

# dataset = RadarDatasetClass(scene='parking')
# indx = 14547

# dataset = RadarDatasetClass(scene='outdoor')
# indx = 6893

# indx = random.randint(0, len(dataset)-1)

l, r = dataset[indx]
print('sample: ', indx)

# draw_dataset(l, r)
rospy.init_node('visualize_radar')

pub = rospy.Publisher('radar', MarkerArray, queue_size=1, latch=True)
publ = rospy.Publisher('laser', LaserScan, queue_size=1, latch=True)

min_v = np.min(r[:, 6])
max_v = np.max(r[:, 6])


def make_marker(p, ns, i, id, c, text=False):
    mk = Marker()
    mk.header.frame_id = 'map'
    mk.header.stamp = rospy.Time.now()
    mk.ns = ns
    mk.id = i
    scale = 0.2

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

    # mk.color.r = 1
    # mk.color.g = 1
    # mk.color.b = 0

    return mk


rospy.loginfo('update')
mks = MarkerArray()

for i, radar in enumerate(r):
    ns = str(int(radar[1]))
    id = str(int(radar[2]))
    v = radar[6]
    p = radar[3:6]
    mk = make_marker(p, ns, 2*i+1, id, c=v)
    tk = make_marker(p, ns, 2*i, id, c=v, text=True)
    mks.markers.append(mk)
    # mks.markers.append(tk)
pub.publish(mks)

ls = LaserScan()
ls.header.frame_id = 'map'
ls.header.stamp = rospy.Time.now()
ls.range_max = 100.0
ls.range_min = 0
ls.angle_max = 2.094395
ls.angle_min = -2.094395
ls.angle_increment = 0.017453
ls.ranges = l.tolist()

publ.publish(ls)


rospy.spin()
