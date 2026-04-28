// Nordin Mohamed Mohamed Shafiq Elbastwisi
#pragma once

#include "localization.h"

#include <string>
#include <vector>

struct SequenceMetric {
    std::string category;
    std::string sequenceId;
    BoundingBox prediction;
    BoundingBox label;
    double iou = 0.0;
};

struct CategorySummary {
    std::string category;
    std::size_t count = 0;
    double meanIoU = 0.0;
    double medianIoU = 0.0;
    double stdIoU = 0.0;
    double minIoU = 0.0;
    double maxIoU = 0.0;
};

double computeIoU(const BoundingBox& prediction, const BoundingBox& label);

bool readBoundingBoxFile(const std::string& path, BoundingBox& box);

bool appendSequenceMetricCsv(const std::string& csvPath, const SequenceMetric& metric);

std::vector<SequenceMetric> readSequenceMetricsCsv(const std::string& csvPath);

std::vector<CategorySummary> summarizeByCategory(const std::vector<SequenceMetric>& metrics);

bool writeCategorySummaryCsv(const std::string& csvPath, const std::vector<CategorySummary>& summaries);
