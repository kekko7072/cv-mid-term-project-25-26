// ALBERTO SALESE

#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace midterm {

struct TrackEvidence {
    cv::Point2f first_point;
    cv::Point2f last_point;
    int observed_steps = 0;
    int foreground_steps = 0;
    float accumulated_residual = 0.0f;
    float accumulated_flow = 0.0f;
};

struct MotionAnalysis {
    std::vector<TrackEvidence> observed_tracks;
    std::vector<TrackEvidence> moving_tracks;
};

struct MotionConfig {
    int max_corners = 400;
    double quality_level = 0.01;
    double min_distance = 7.0;
    int block_size = 7;
    cv::Size window_size = cv::Size(21, 21);
    int pyramid_levels = 3;
    double min_flow_magnitude = 2.0;
    double residual_threshold = 2.0;
    int min_track_observations = 1;
    int min_foreground_hits = 1;
};

class MotionExtractor {
  public:
    MotionExtractor(MotionConfig config = MotionConfig());

    MotionAnalysis extractMotionEvidence(const std::vector<cv::Mat>& frames,
                                         std::string* debug_note = nullptr) const;

  private:
    MotionConfig config_;
};

}  // namespace midterm
