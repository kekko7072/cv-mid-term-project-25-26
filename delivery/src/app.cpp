// Francesco Vezzani

#include "app.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
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

vector<fs::path> collectImageFiles(const fs::path& folder) {
    vector<fs::path> files;
    if (!fs::exists(folder)) return files;

    // Keep the file list sorted so the sequence is processed in frame order.
    for (const auto& entry : fs::directory_iterator(folder)) {
        if (!entry.is_regular_file()) continue;
        if (isImageFile(entry.path())) files.push_back(entry.path());
    }

    sort(files.begin(), files.end());
    return files;
}

vector<Mat> loadFrames(const fs::path& folder) {
    vector<Mat> frames;
    for (const auto& file : collectImageFiles(folder)) {
        Mat img = imread(file.string());
        if (img.empty()) return {};
        frames.push_back(img);
    }
    return frames;
}

vector<Point2f> extractTrackPoints(const vector<TrackEvidence>& tracks) {
    vector<Point2f> points;
    for (const auto& track : tracks) points.push_back(track.first_point);
    return points;
}

vector<Point2f> selectPoints(const MotionAnalysis& motion) {
    // Prefer the moving tracks, but fall back to all observed tracks when the
    // motion evidence is too sparse to build a stable box.
    vector<Point2f> points = extractTrackPoints(motion.moving_tracks);
    if (points.size() < 4) points = extractTrackPoints(motion.observed_tracks);
    return rejectOutliers3Sigma(points, 4);
}

BoundingBox makePrediction(const vector<Point2f>& points, const Mat& frame) {
    if (points.empty()) return {0, 0, frame.cols - 1, frame.rows - 1};
    return pointsToBoundingBox(points, frame.cols, frame.rows);
}

void saveSequenceOutputs(const fs::path& outputFolder, const Mat& frame,
                         const BoundingBox& prediction, const vector<Point2f>& points) {
    fs::create_directories(outputFolder);
    writeBoundingBoxFile((outputFolder / "0000.txt").string(), prediction);
    writePointsFile((outputFolder / "moving_points.txt").string(), points);

    Mat image = frame.clone();
    rectangle(image, Point(prediction.xMin, prediction.yMin), Point(prediction.xMax, prediction.yMax), Scalar(0, 255, 0), 2);
    imwrite((outputFolder / "0000.png").string(), image);
}

optional<BoundingBox> findBestLabel(const fs::path& folder, const BoundingBox& prediction) {
    if (folder.empty() || !fs::exists(folder)) return nullopt;

    double bestIou = -1.0;
    optional<BoundingBox> best;
    for (const auto& entry : fs::recursive_directory_iterator(folder)) {
        if (!entry.is_regular_file() || entry.path().extension() != ".txt") continue;

        BoundingBox label;
        if (!readBoundingBoxFile(entry.path().string(), label)) continue;

        double iou = computeIoU(prediction, label);
        if (!best || iou > bestIou) {
            bestIou = iou;
            best = label;
        }
    }

    return best;
}

double meanIoU(const vector<CategorySummary>& summaries) {
    double total = 0.0;
    for (const auto& summary : summaries) total += summary.meanIoU;
    return summaries.empty() ? 0.0 : total / static_cast<double>(summaries.size());
}

int countTruePositives(const vector<SequenceMetric>& metrics) {
    return static_cast<int>(count_if(metrics.begin(), metrics.end(),
                                     [](const SequenceMetric& metric) { return metric.iou > 0.5; }));
}

void writeDatasetSummary(const fs::path& summaryPath, int processed,
                         const vector<SequenceMetric>& metrics, double miou, int truePositives) {
    ofstream out(summaryPath);
    if (!out) return;

    out << "processed_sequences: " << processed << '\n';
    out << "evaluated_sequences: " << metrics.size() << '\n';
    out << "mean_iou: " << miou << '\n';
    out << "true_positives: " << truePositives << '\n';
    out << "detection_accuracy: "
        << (metrics.empty() ? 0.0 : static_cast<double>(truePositives) / metrics.size()) << '\n';
}

struct DatasetArguments {
    fs::path rawRoot;
    fs::path labelRoot;
    fs::path outputRoot;
};

bool runSequence(const fs::path& imageFolder, const fs::path& labelFolder, const fs::path& outputFolder,
                 const string& category, const string& id, const string& metricsCsv) {
    vector<Mat> frames = loadFrames(imageFolder);
    if (frames.size() < 2) return false;

    MotionExtractor extractor;
    string note;
    MotionAnalysis motion = extractor.extractMotionEvidence(frames, &note);

    vector<Point2f> points = selectPoints(motion);
    BoundingBox prediction = makePrediction(points, frames.front());
    saveSequenceOutputs(outputFolder, frames.front(), prediction, points);

    if (optional<BoundingBox> label = findBestLabel(labelFolder, prediction)) {
        appendSequenceMetricCsv(metricsCsv, {category, id, prediction, *label, computeIoU(prediction, *label)});
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

    // Regenerate the summary files from scratch on each dataset run.
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

    double miou = meanIoU(summaries);
    int truePositives = countTruePositives(metrics);
    writeDatasetSummary(metricsRoot / "summary.txt", processed, metrics, miou, truePositives);

    cout << "Processed " << processed << " sequences\n";
    cout << "mIoU: " << miou << '\n';
    cout << "Detection accuracy: " << (metrics.empty() ? 0.0 : static_cast<double>(truePositives) / metrics.size()) << '\n';
    return processed > 0 ? 0 : -1;
}

void printUsage(const char* program) {
    cout << "Usage: " << program << '\n';
    cout << "   or: " << program << " <input_folder> <output_folder>\n";
    cout << "   or: " << program << " <raw_folder> <labels_folder> <output_folder>\n";
    cout << "   or: " << program << " --raw <raw_folder> [--labels <labels_folder>] --output <output_folder>\n";
}

bool parseDatasetFlags(int argc, char** argv, DatasetArguments& args, string& error) {
    // Only the dataset mode uses flags; single-sequence mode stays positional.
    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "--raw" || arg == "--labels" || arg == "--output") {
            if (i + 1 >= argc) {
                error = "Missing value for " + arg;
                return false;
            }

            fs::path value = argv[++i];
            if (arg == "--raw") args.rawRoot = value;
            else if (arg == "--labels") args.labelRoot = value;
            else args.outputRoot = value;
            continue;
        }

        error = "Unknown argument: " + arg;
        return false;
    }

    if (args.rawRoot.empty()) {
        error = "Missing required argument: --raw";
        return false;
    }
    if (args.outputRoot.empty()) {
        error = "Missing required argument: --output";
        return false;
    }

    if (args.labelRoot.empty()) {
        // If the raw folder lives next to a labels folder, pick it up automatically.
        fs::path siblingLabels = args.rawRoot.parent_path() / "labels";
        if (!siblingLabels.empty() && fs::exists(siblingLabels)) args.labelRoot = siblingLabels;
    }

    return true;
}

}  // namespace

int runApplication(int argc, char** argv) {
    // No arguments means "run the repository defaults".
    if (argc == 1) return runDataset("data/raw", "data/labels", "output");

    string firstArg = argv[1];
    if (firstArg == "-h" || firstArg == "--help") {
        printUsage(argv[0]);
        return 0;
    }

    if (!firstArg.empty() && firstArg[0] == '-') {
        DatasetArguments args;
        string error;

        if (!parseDatasetFlags(argc, argv, args, error)) {
            if (!error.empty()) cout << error << '\n';
            printUsage(argv[0]);
            return -1;
        }

        return runDataset(args.rawRoot, args.labelRoot, args.outputRoot);
    }

    if (argc == 3) {
        return runSequence(argv[1], "", argv[2], fs::path(argv[1]).filename().string(), "single", "") ? 0 : -1;
    }
    if (argc == 4) return runDataset(argv[1], argv[2], argv[3]);

    printUsage(argv[0]);
    return -1;
}

}  // namespace midterm
