#pragma once

#include <opencv2/core/types.hpp>

#include <string>
#include <vector>

struct BoundingBox {
    int xMin = 0;
    int yMin = 0;
    int xMax = 0;
    int yMax = 0;

    bool isValid() const {
        return xMin <= xMax && yMin <= yMax;
    }
};

std::vector<cv::Point2f> rejectOutliers3Sigma(const std::vector<cv::Point2f>& points, std::size_t minRequired = 4);

BoundingBox pointsToBoundingBox(const std::vector<cv::Point2f>& points, int imageWidth, int imageHeight);

bool writeBoundingBoxFile(const std::string& path, const BoundingBox& box);

bool writePointsFile(const std::string& path, const std::vector<cv::Point2f>& points);
