import numpy as np
from scipy.interpolate import UnivariateSpline
import math


class LaneProcessing():
    """Takes in LaneNet's raw output vectors and process then into left to right ordered lanes"""

    def __init__(self,full_lane_pts,x_size,y_size):
        self.full_lane_pts = full_lane_pts
        self.input_image_x = x_size
        self.input_image_y = y_size
        self._full_lanes_transformation()
        self._ordering_lanes()

    def _ordering_lanes(self):
        max_y_pts = []
        min_y_pts = []
        min_y_VAL = []

        for lane in self.full_lane_pts:
            max_y_pts.append(lane[np.argmax(lane[:,1])])
            min_y_pts.append(lane[np.argmin(lane[:,1])])
            min_y_VAL.append(lane[np.argmin(lane[:,1]),1])

        max_y_pts = np.array(max_y_pts)
        min_y_pts = np.array(min_y_pts)
        maxmin_y_val = max(min_y_VAL)

        slopes = (max_y_pts[:,1]-min_y_pts[:,1])/(max_y_pts[:,0]-min_y_pts[:,0])
        intercepts = min_y_pts[:,1] - slopes*min_y_pts[:,0]
        order_x_values = (maxmin_y_val - intercepts)/slopes

        lane_ordering = []
        for i in range(len(order_x_values)):
            lane_ordering.append((order_x_values[i],self.full_lane_pts[i]))
        dtype = [("x_values",float),("lane_pts",list)]
        lane_ordering = np.array(lane_ordering,dtype=dtype)
        lane_ordering = np.sort(lane_ordering, order="x_values")

        ordered_lane_pts = []
        for _,lane_pts in lane_ordering:
            ordered_lane_pts.append(lane_pts)

        self.full_lane_pts = ordered_lane_pts


    def _full_lanes_transformation(self):
        for i in range(len(self.full_lane_pts)):
            self.full_lane_pts[i] = ([0,self.input_image_y] - self.full_lane_pts[i]) * [-1,1]
            idx = np.argsort(self.full_lane_pts[i], axis=0)
            self.full_lane_pts[i][:,0] = np.take_along_axis(self.full_lane_pts[i][:,0], idx[:,1], axis=0)
            self.full_lane_pts[i][:,1] = np.take_along_axis(self.full_lane_pts[i][:,1], idx[:,1], axis=0)



    def get_full_lane_pts(self):
        return self.full_lane_pts



class DualLanesToTrajectory():
    """Takes two x by 2 shaped lane line point vector arrays and outputs estimated centerlines."""

    def __init__(self,lane_left_pts,lane_right_pts,N_centerpts=10):
        self.lane_pts = [lane_left_pts,lane_right_pts]
        self.tot_dist = []
        self.seg_cum_dists = []
        self.seg_vectors = []
        self.matching_pts = []
        self.centerpoints = []

        self.N_centerpts = N_centerpts
        self.result_status = True

        self._update_centerpoints()


    def _pre_processing_lane_pts(self):
        for i in range(len(self.lane_pts)):
            lane_side_pts = self.lane_pts[i]
            if lane_side_pts.size <= 1:
                self.result_status = False
                return None
            else:
                self.result_status = True

            if lane_side_pts[0,1] >= 0:
                extrapo_vector = lane_side_pts[1] - lane_side_pts[0]
                new_pt_x = lane_side_pts[0,0] + (lane_side_pts[0,1] * -1.0 * extrapo_vector[0] / extrapo_vector[1])
                self.lane_pts[i] = np.insert(lane_side_pts, 0, np.array([new_pt_x, 0]), axis=0)
            else:
                self.lane_pts[i] = lane_side_pts


    def _cal_segment_param(self):
        for lane_side_pts in self.lane_pts:
            total_distance = 0
            segment_cumulative_distances = []
            segment_vectors = []
            for i in range(len(lane_side_pts)):
                if i==0: continue
                seg_dist = math.sqrt((lane_side_pts[i][0] - lane_side_pts[i-1][0])**2 + (lane_side_pts[i][1] - lane_side_pts[i-1][1])**2)
                segment_cumulative_distances.append(total_distance)
                segment_vectors.append(lane_side_pts[i]-lane_side_pts[i-1])
                total_distance += seg_dist
            self.tot_dist.append(total_distance)
            self.seg_cum_dists.append(np.array(segment_cumulative_distances))
            self.seg_vectors.append(np.array(segment_vectors))


    def _cal_centerpts_pairs(self):
        for i in range(len(self.lane_pts)):
            total_distance = self.tot_dist[i]
            segment_cumulative_distances = self.seg_cum_dists[i]
            segment_vectors = self.seg_vectors[i]
            lane_pts = self.lane_pts[i]
            wedge_dists = np.linspace(1, total_distance, self.N_centerpts, endpoint=True)
            starting_pts_index = np.searchsorted(segment_cumulative_distances,wedge_dists, side="left") - 1
            lacking_dists = wedge_dists-np.take(segment_cumulative_distances, starting_pts_index)
            matching_pts = []
            i = 0
            for idx in starting_pts_index:
                theta = math.atan(segment_vectors[idx][1]/segment_vectors[idx][0] if segment_vectors[idx][0] != 0 else 0)
                x,y = 0,0
                if theta > 0:
                    x = lacking_dists[i] * math.cos(theta) + lane_pts[idx][0]
                    y = lacking_dists[i] * math.sin(theta) + lane_pts[idx][1]
                else:
                    x = -1 * lacking_dists[i] * math.cos(theta) + lane_pts[idx][0]
                    y = -1 * lacking_dists[i] * math.sin(theta) + lane_pts[idx][1]
                matching_pts.append([x,y])
                i += 1

            self.matching_pts.append(np.array(matching_pts))


    def _cal_centerpts(self):
        left_pts = self.matching_pts[0]
        right_pts = self.matching_pts[1]
        self.centerpoints = (left_pts[:,0]+right_pts[:,0])/2 , (left_pts[:,1]+right_pts[:,1])/2


    def update_input(self,lane_left_pts,lane_right_pts):
        self.lane_pts = [np.array(lane_left_pts),np.array(lane_right_pts)]
        self._update_centerpoints()


    def _update_centerpoints(self):
        self._pre_processing_lane_pts()
        if self.result_status: 
            self._cal_segment_param()
            self._cal_centerpts_pairs()
            self._cal_centerpts()


    def get_centerpoints(self):
        return self.centerpoints


    def get_matching_points(self):
        return self.matching_pts


    def get_spline(self):
        x_center, y_center = self.centerpoints
        return UnivariateSpline(y_center, x_center)
