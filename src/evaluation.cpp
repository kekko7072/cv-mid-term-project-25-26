// Nordin Mohamed Mohamed Shafiq Elbastwisi

#include "evaluation.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <map>
#include <numeric>
#include <sstream>

namespace {

double computeArea(const BoundingBox& box) {
  if (!box.isValid()) {
    return 0.0;
  }

  // Boxes are stored as inclusive pixel coordinates, so the side lengths must
  // include the endpoint pixel (+1) when converting to area.
  const int width = box.xMax - box.xMin + 1;
  const int height = box.yMax - box.yMin + 1;
  if (width <= 0 || height <= 0) {
    return 0.0;
  }

  return static_cast<double>(width) * static_cast<double>(height);
}

std::vector<std::string> splitCsvLine(const std::string& line) {
  std::vector<std::string> parts;
  std::stringstream ss(line);
  std::string token;
  while (std::getline(ss, token, ',')) {
    parts.push_back(token);
  }
  return parts;
}

} // namespace

double computeIoU(const BoundingBox& prediction, const BoundingBox& label) {
  if (!prediction.isValid() || !label.isValid()) {
    return 0.0; // Invalid boxes have IoU of 0
  }

  const int interXMin = std::max(prediction.xMin, label.xMin);
  const int interYMin = std::max(prediction.yMin, label.yMin);
  const int interXMax = std::min(prediction.xMax, label.xMax);
  const int interYMax = std::min(prediction.yMax, label.yMax);

  BoundingBox inter{interXMin, interYMin, interXMax, interYMax};
  const double intersection = computeArea(inter);

  const double predictionArea = computeArea(prediction);
  const double labelArea = computeArea(label);
  const double uni = predictionArea + labelArea - intersection;

  if (uni <= 0.0) {
    return 0.0; // edge case: avoid division by zero, treat as no overlap
  }

  return intersection / uni;
}

bool readBoundingBoxFile(const std::string& path, BoundingBox& box) {
  std::ifstream in(path);
  if (!in.is_open()) {
    return false;
  }

  if (!(in >> box.xMin >> box.yMin >> box.xMax >> box.yMax)) {
    return false;
  }

  return true;
}

bool appendSequenceMetricCsv(const std::string& csvPath, const SequenceMetric& metric) {
  bool writeHeader = false;
  {
    std::ifstream check(csvPath);
    // Append mode needs a header only for the first write, so we probe the file
    // once instead of rewriting the whole CSV every time.
    writeHeader = !check.good() || check.peek() == std::ifstream::traits_type::eof();
  }

  std::ofstream out(csvPath, std::ios::app);
  if (!out.is_open()) {
    return false;
  }

  if (writeHeader) {
    out << "category,sequence_id,pred_xmin,pred_ymin,pred_xmax,pred_ymax,label_xmin,label_ymin,label_xmax,label_ymax,iou\n";
  }

  out << metric.category << ','
    << metric.sequenceId << ','
    << metric.prediction.xMin << ','
    << metric.prediction.yMin << ','
    << metric.prediction.xMax << ','
    << metric.prediction.yMax << ','
    << metric.label.xMin << ','
    << metric.label.yMin << ','
    << metric.label.xMax << ','
    << metric.label.yMax << ','
    << std::fixed << std::setprecision(6) << metric.iou << '\n';

  return true;
}

std::vector<SequenceMetric> readSequenceMetricsCsv(const std::string& csvPath) {
  std::vector<SequenceMetric> metrics;
  std::ifstream in(csvPath);
  if (!in.is_open()) {
    return metrics;
  }

  std::string line;
  bool isFirst = true;
  while (std::getline(in, line)) {
    if (line.empty()) {
      continue;
    }

    if (isFirst) {
      isFirst = false;
      if (line.rfind("category,", 0) == 0) {
        // Skip the header row when rebuilding metrics from disk.
        continue;
      }
    }

    std::vector<std::string> parts = splitCsvLine(line);
    if (parts.size() != 11) {
      continue;
    }

    SequenceMetric metric;
    try {
      metric.category = parts[0];
      metric.sequenceId = parts[1];
      metric.prediction.xMin = std::stoi(parts[2]);
      metric.prediction.yMin = std::stoi(parts[3]);
      metric.prediction.xMax = std::stoi(parts[4]);
      metric.prediction.yMax = std::stoi(parts[5]);
      metric.label.xMin = std::stoi(parts[6]);
      metric.label.yMin = std::stoi(parts[7]);
      metric.label.xMax = std::stoi(parts[8]);
      metric.label.yMax = std::stoi(parts[9]);
      metric.iou = std::stod(parts[10]);
    } catch (...) {
      continue;
    }

    metrics.push_back(metric);
  }

  return metrics;
}

std::vector<CategorySummary> summarizeByCategory(const std::vector<SequenceMetric>& metrics) {
  std::map<std::string, std::vector<double>> byCategory;
  for (const SequenceMetric& metric : metrics) {
    byCategory[metric.category].push_back(metric.iou);
  }

  std::vector<CategorySummary> summaries;
  summaries.reserve(byCategory.size());

  for (auto& item : byCategory) {
    std::vector<double>& values = item.second;
    if (values.empty()) {
      continue;
    }

    std::sort(values.begin(), values.end());

    CategorySummary summary;
    summary.category = item.first;
    summary.count = values.size();
    summary.minIoU = values.front();
    summary.maxIoU = values.back();

    const double sum = std::accumulate(values.begin(), values.end(), 0.0);
    summary.meanIoU = sum / static_cast<double>(values.size());

    const std::size_t middle = values.size() / 2;
    if (values.size() % 2 == 0) {
      summary.medianIoU = (values[middle - 1] + values[middle]) / 2.0;
    } else {
      summary.medianIoU = values[middle];
    }

    double variance = 0.0;
    for (const double value : values) {
      const double diff = value - summary.meanIoU;
      variance += diff * diff;
    }
    variance /= static_cast<double>(values.size());
    summary.stdIoU = std::sqrt(variance);

    summaries.push_back(summary);
  }

  return summaries;
}

bool writeCategorySummaryCsv(const std::string& csvPath, const std::vector<CategorySummary>& summaries) {
  std::ofstream out(csvPath);
  if (!out.is_open()) {
    return false;
  }

  out << "category,n_sequences,mean_iou,median_iou,std_iou,min_iou,max_iou\n";
  for (const CategorySummary& summary : summaries) {
    out << summary.category << ','
      << summary.count << ','
      << std::fixed << std::setprecision(6)
      << summary.meanIoU << ','
      << summary.medianIoU << ','
      << summary.stdIoU << ','
      << summary.minIoU << ','
      << summary.maxIoU << '\n';
  }

  return true;
}

