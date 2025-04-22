#pragma once

#include "STrack.h"
#include <vector>


struct Rect {
	float x;
	float y;
	float width;
	float height;
};


struct Object
{
    Rect rect;
    int label;
    float prob;
	std::vector<float> extra;
};

class BYTETracker
{
public:
	BYTETracker(float track_thresh = 0.5, float high_thresh = 0.6, float match_thresh = 0.8, int frame_rate = 30, int track_buffer = 30);
	~BYTETracker();

	uint32_t update(const float* const box, const uint32_t nbox, float* out, const uint32_t n_extra = 0);
	vector<STrack> update(const vector<Object>& objects);

	int get_tracked_tracks_count() { return tracked_stracks.size(); }
	int get_lost_tracks_count() { return lost_stracks.size(); }
	int get_removed_tracks_count() { return removed_stracks.size(); }

	void clear_buffer();
	void clear_lost();

private:
	vector<STrack*> joint_stracks(vector<STrack*> &tlista, vector<STrack> &tlistb);
	vector<STrack> joint_stracks(vector<STrack> &tlista, vector<STrack> &tlistb);

	vector<STrack> sub_stracks(vector<STrack> &tlista, vector<STrack> &tlistb);
	void remove_duplicate_stracks(vector<STrack> &resa, vector<STrack> &resb, vector<STrack> &stracksa, vector<STrack> &stracksb);

	void linear_assignment(vector<vector<float> > &cost_matrix, int cost_matrix_size, int cost_matrix_size_size, float thresh,
		vector<vector<int> > &matches, vector<int> &unmatched_a, vector<int> &unmatched_b);
	vector<vector<float> > iou_distance(vector<STrack*> &atracks, vector<STrack> &btracks, int &dist_size, int &dist_size_size);
	vector<vector<float> > iou_distance(vector<STrack> &atracks, vector<STrack> &btracks);
	vector<vector<float> > ious(vector<vector<float> > &atlbrs, vector<vector<float> > &btlbrs);

	double lapjv(const vector<vector<float> > &cost, vector<int> &rowsol, vector<int> &colsol,
		bool extend_cost = false, float cost_limit = LONG_MAX, bool return_cost = true);

private:

	float track_thresh;
	float high_thresh;
	float match_thresh;
	int frame_id;
	int max_time_lost;

	vector<STrack> tracked_stracks;
	vector<STrack> lost_stracks;
	vector<STrack> removed_stracks;
	byte_kalman::KalmanFilter kalman_filter;
};
