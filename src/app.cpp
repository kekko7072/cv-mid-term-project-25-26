#include "app.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "evaluation.h"
#include "localization.h"
#include "motion_extraction.h"

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

namespace midterm {
namespace {

bool isImageFile(const fs::path& path) {
    string ext = path.extension().string();
    transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp";
}

vector<fs::path> collectFiles(const fs::path& folder, const string& ext = "") {
    vector<fs::path> files;
    if (!fs::exists(folder)) return files;
    for (const auto& entry : fs::directory_iterator(folder)) {
        if (!entry.is_regular_file()) continue;
        if (ext.empty() ? isImageFile(entry.path()) : entry.path().extension() == ext) files.push_back(entry.path());
    }
    sort(files.begin(), files.end());
    return files;
}

vector<Mat> loadFrames(const vector<fs::path>& files) {
    vector<Mat> frames;
    for (const auto& file : files) {
        Mat img = imread(file.string());
        if (img.empty()) return {};
        frames.push_back(img);
    }
    return frames;
}

vector<Point2f> trackPoints(const vector<TrackEvidence>& tracks) {
    vector<Point2f> points;
    for (const auto& track : tracks) points.push_back(track.first_point);
    return points;
}

BoundingBox bestLabel(const fs::path& folder, const BoundingBox& prediction, bool& found) {
    found = false;
    double bestIou = -1.0;
    BoundingBox best;
    if (!fs::exists(folder)) return best;
    for (const auto& entry : fs::recursive_directory_iterator(folder)) {
        if (!entry.is_regular_file() || entry.path().extension() != ".txt") continue;
        BoundingBox label;
        if (!readBoundingBoxFile(entry.path().string(), label)) continue;
        double iou = computeIoU(prediction, label);
        if (!found || iou > bestIou) {
            found = true;
            bestIou = iou;
            best = label;
        }
    }
    return best;
}

bool runSequence(const fs::path& imageFolder, const fs::path& labelFolder, const fs::path& outputFolder,
                 const string& category, const string& id, const string& metricsCsv) {
    vector<fs::path> files = collectFiles(imageFolder);
    if (files.size() < 2) return false;

    vector<Mat> frames = loadFrames(files);
    if (frames.empty()) return false;

    MotionExtractor extractor;
    string note;
    MotionAnalysis motion = extractor.extractMotionEvidence(frames, &note);

    vector<Point2f> points = trackPoints(motion.moving_tracks);
    if (points.size() < 4) points = trackPoints(motion.observed_tracks);
    points = rejectOutliers3Sigma(points, 4);

    BoundingBox prediction;
    if (points.empty()) prediction = {0, 0, frames[0].cols - 1, frames[0].rows - 1};
    else prediction = pointsToBoundingBox(points, frames[0].cols, frames[0].rows);

    fs::create_directories(outputFolder);
    writeBoundingBoxFile((outputFolder / "0000.txt").string(), prediction);
    writePointsFile((outputFolder / "moving_points.txt").string(), points);

    Mat image = frames[0].clone();
    rectangle(image, Point(prediction.xMin, prediction.yMin), Point(prediction.xMax, prediction.yMax), Scalar(0, 255, 0), 2);
    imwrite((outputFolder / "0000.png").string(), image);

    bool found = false;
    BoundingBox label = bestLabel(labelFolder, prediction, found);
    if (found) {
        SequenceMetric metric{category, id, prediction, label, computeIoU(prediction, label)};
        appendSequenceMetricCsv(metricsCsv, metric);
    }

    cout << category << ": observed=" << motion.observed_tracks.size()
         << " moving=" << motion.moving_tracks.size()
         << " note=" << note << '\n';
    return true;
}

int runDataset(const fs::path& rawRoot, const fs::path& labelRoot, const fs::path& outputRoot) {
    fs::path resultsRoot = outputRoot / "results";
    fs::path metricsRoot = outputRoot / "metrics";
    fs::create_directories(resultsRoot);
    fs::create_directories(metricsRoot);

    string metricsCsv = (metricsRoot / "sequence_metrics.csv").string();
    fs::remove(metricsCsv);
    fs::remove(metricsRoot / "category_summary.csv");
    fs::remove(metricsRoot / "summary.txt");

    int processed = 0;
    for (const auto& entry : fs::directory_iterator(rawRoot)) {
        if (!entry.is_directory()) continue;
        string category = entry.path().filename().string();
        if (runSequence(entry.path(), labelRoot / category, resultsRoot / category, category, category, metricsCsv)) processed++;
    }

    vector<SequenceMetric> metrics = readSequenceMetricsCsv(metricsCsv);
    vector<CategorySummary> summaries = summarizeByCategory(metrics);
    writeCategorySummaryCsv((metricsRoot / "category_summary.csv").string(), summaries);

    double miou = 0.0;
    for (const auto& summary : summaries) miou += summary.meanIoU;
    if (!summaries.empty()) miou /= static_cast<double>(summaries.size());

    int truePositives = 0;
    for (const auto& metric : metrics) if (metric.iou > 0.5) truePositives++;

    ofstream out(metricsRoot / "summary.txt");
    if (out) {
        out << "processed_sequences: " << processed << '\n';
        out << "evaluated_sequences: " << metrics.size() << '\n';
        out << "mean_iou: " << miou << '\n';
        out << "true_positives: " << truePositives << '\n';
        out << "detection_accuracy: " << (metrics.empty() ? 0.0 : static_cast<double>(truePositives) / metrics.size()) << '\n';
    }

    cout << "Processed " << processed << " sequences\n";
    cout << "mIoU: " << miou << '\n';
    cout << "Detection accuracy: " << (metrics.empty() ? 0.0 : static_cast<double>(truePositives) / metrics.size()) << '\n';
    return processed > 0 ? 0 : -1;
}

}  // namespace

int runApplication(int argc, char** argv) {
    if (argc == 1) return runDataset("data/raw", "data/labels", "output");
    if (argc == 3) return runSequence(argv[1], "", argv[2], fs::path(argv[1]).filename().string(), "single", "") ? 0 : -1;
    if (argc == 4) return runDataset(argv[1], argv[2], argv[3]);

    cout << "Usage: " << argv[0] << " [raw_folder labels_folder output_folder]\n";
    cout << "   or: " << argv[0] << " <input_folder> <output_folder>\n";
    return -1;
}

}  // namespace midterm
