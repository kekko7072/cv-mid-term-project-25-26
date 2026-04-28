// Nordin Mohamed Mohamed Shafiq Elbastwisi

#include "localization.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>

namespace {
// standard deviation and mean will be used to reject outliers in the pointsToBoundingBox function, so we compute them here as helper functions
double computeMean(const std::vector<float>& values) {
  if (values.empty()) {
    return 0.0;
  }

  // We use the full point set here because the outlier filter is based on a global
  // center/spread estimate, not on a running or per-axis window.
  const double sum = std::accumulate(values.begin(), values.end(), 0.0);
  return sum / static_cast<double>(values.size());
}

double computeStdDev(const std::vector<float>& values, const double mean) {
  if (values.size() < 2) {
    return 0.0;
  }

  double variance = 0.0;
  for (const float value : values) {
    const double diff = static_cast<double>(value) - mean;
    variance += diff * diff;
  }

  variance /= static_cast<double>(values.size());
  return std::sqrt(variance);
}

int clampToRange(const int value, const int low, const int high) {
  return std::max(low, std::min(value, high));
}

} // namespace

std::vector<cv::Point2f> rejectOutliers3Sigma(const std::vector<cv::Point2f>& points, std::size_t minRequired) {
  if (points.size() < minRequired || points.size() < 2) {
    return points;
  }

  std::vector<float> xs;
  std::vector<float> ys;
  xs.reserve(points.size());
  ys.reserve(points.size());

  for (const cv::Point2f& point : points) {
    xs.push_back(point.x);
    ys.push_back(point.y);
  }

  const double meanX = computeMean(xs);
  const double meanY = computeMean(ys);
  const double stdX = computeStdDev(xs, meanX);
  const double stdY = computeStdDev(ys, meanY);

  if (stdX <= 0.0 || stdY <= 0.0) {
    return points;
  }

  std::vector<cv::Point2f> filtered;
  filtered.reserve(points.size());

  for (const cv::Point2f& point : points) {
    const bool inX = std::abs(static_cast<double>(point.x) - meanX) <= (3.0 * stdX);
    const bool inY = std::abs(static_cast<double>(point.y) - meanY) <= (3.0 * stdY);
    if (inX && inY) {
      filtered.push_back(point);
    }
  }

  if (filtered.size() < minRequired) {
    // If filtering removes too many points, fall back to the raw set so the
    // pipeline still produces a box instead of failing on sparse sequences.
    return points;
  }

  return filtered;
}

BoundingBox pointsToBoundingBox(const std::vector<cv::Point2f>& points, const int imageWidth, const int imageHeight) {
  BoundingBox box;

  if (points.empty() || imageWidth <= 0 || imageHeight <= 0) {
    return box;
  }

  float minX = points.front().x;
  float maxX = points.front().x;
  float minY = points.front().y;
  float maxY = points.front().y;

  for (const cv::Point2f& point : points) {
    minX = std::min(minX, point.x);
    maxX = std::max(maxX, point.x);
    minY = std::min(minY, point.y);
    maxY = std::max(maxY, point.y);
  }

  // Clamp to the image bounds before converting to integers, otherwise a
  // slightly noisy feature can produce coordinates outside the frame.
  box.xMin = clampToRange(static_cast<int>(std::lround(minX)), 0, imageWidth - 1);
  box.xMax = clampToRange(static_cast<int>(std::lround(maxX)), 0, imageWidth - 1);
  box.yMin = clampToRange(static_cast<int>(std::lround(minY)), 0, imageHeight - 1);
  box.yMax = clampToRange(static_cast<int>(std::lround(maxY)), 0, imageHeight - 1);

  if (!box.isValid()) {
    return BoundingBox{};
  }

  return box;
}

bool writeBoundingBoxFile(const std::string& path, const BoundingBox& box) {
  std::ofstream out(path);
  if (!out.is_open()) {
    return false;
  }

  out << box.xMin << ' ' << box.yMin << ' ' << box.xMax << ' ' << box.yMax << '\n';
  return true;
}

bool writePointsFile(const std::string& path, const std::vector<cv::Point2f>& points) {
  std::ofstream out(path);
  if (!out.is_open()) {
    return false;
  }

  for (const cv::Point2f& point : points) {
    out << point.x << ' ' << point.y << '\n';
  }
  return true;
}

