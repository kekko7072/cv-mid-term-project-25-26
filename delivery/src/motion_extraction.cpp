// ALBERTO SALESE

#include "motion_extraction.h"

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <filesystem>
#include <vector>
#include <algorithm>
#include <fstream>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

//soglia punto si muove o no
const float MOTION_THRESHOLD = 2.0;


//carico sequenza immagini
void loadImages(const string& folder, vector<Mat>& frames, vector<Mat>& gray)
{
    vector<string> files;

    //prendo tutti i file dentro la cartella
    for (auto& entry : fs::directory_iterator(folder))
        files.push_back(entry.path().string());
    for (auto& f : files)
    {
        Mat img = imread(f); //leggo immagine

        if (img.empty()) continue;

        frames.push_back(img); //salvo immagine originale

        Mat g;
        cvtColor(img, g, COLOR_BGR2GRAY); //converto in grayscale
        gray.push_back(g);
    }
}

//feature con SIFT
void siftFeatures(Mat& img, vector<KeyPoint>& kp, Mat& des)
{
    Ptr<SIFT> sift = SIFT::create(); //creo oggetto SIFT
    sift->detectAndCompute(img, noArray(), kp, des); //detectAndCompute: trova keypoints e calcola descrittori
}

//matching
vector<DMatch> matchFeatures(Mat& d1, Mat& d2)
{
    //matcher basato su distanza
    BFMatcher matcher(NORM_L2);

    vector<vector<DMatch>> knn;

    //per ogni punto trova i 2 match migliori
    matcher.knnMatch(d1, d2, knn, 2);

    vector<DMatch> good;

    for (auto& m : knn)
    {
        //controllo che ci siano almeno 2 match
        if (m.size() < 2) continue;
        //tengo solo match buoni
        if (m[0].distance < 0.75 * m[1].distance)
            good.push_back(m[0]);
    }
    return good;
}

//calcolo movimento
void processSequence(const string& folder, const string& outFolder)
{
    (void)outFolder;

    vector<Mat> frames, gray;

    //carico sequenza immagini
    loadImages(folder, frames, gray);

    vector<Point2f> allPoints;   //tutti i punti
    vector<float> allMotion;     //movimento di ogni punto

    vector<KeyPoint> kp1, kp2;
    Mat des1, des2;

    //estraggo feature dal primo frame
    siftFeatures(gray[0], kp1, des1);

    //ciclo su tutti i frame successivi
    for (size_t i = 1; i < gray.size(); i++)
    {
        //estraggo feature del frame corrente
        siftFeatures(gray[i], kp2, des2);

        //se non ho descrittori salto
        if (des1.empty() || des2.empty())
            continue;

        //faccio matching tra frame precedente e corrente
        vector<DMatch> matches = matchFeatures(des1, des2);

        for (auto& m : matches)
        {
            //punto nel frame precedente
            Point2f p1 = kp1[m.queryIdx].pt;

            //punto nel frame corrente
            Point2f p2 = kp2[m.trainIdx].pt;

            //distanza
            float dist = norm(p2 - p1);

            allPoints.push_back(p1);
            allMotion.push_back(dist);
        }
        kp1 = kp2;
        des1 = des2;
    }

    vector<Point2f> moving;
    vector<Point2f> staticPts;

    //separo punti in movimento e statici
    for (int i = 0; i < allPoints.size(); i++)
    {
        if (allMotion[i] > MOTION_THRESHOLD)
            moving.push_back(allPoints[i]);   //oggetto in movimento
        else
            staticPts.push_back(allPoints[i]); 
    }

}

namespace midterm {

MotionExtractor::MotionExtractor(MotionConfig config) : config_(config) {}

MotionAnalysis MotionExtractor::extractMotionEvidence(const vector<Mat>& frames,
                                                      string* debug_note) const
{
    MotionAnalysis result;

    if (debug_note) *debug_note = "";

    if (frames.size() < 2)
    {
        if (debug_note) *debug_note = "not_enough_frames";
        return result;
    }

    vector<Mat> gray;

    for (auto& frame : frames)
    {
        if (frame.empty()) continue;

        Mat g;
        if (frame.channels() == 1) g = frame.clone();
        else cvtColor(frame, g, COLOR_BGR2GRAY);
        gray.push_back(g);
    }

    if (gray.size() < 2)
    {
        if (debug_note) *debug_note = "not_enough_frames";
        return result;
    }

    vector<KeyPoint> kp1, kp2;
    Mat des1, des2;

    siftFeatures(gray[0], kp1, des1);

    for (size_t i = 1; i < gray.size(); i++)
    {
        siftFeatures(gray[i], kp2, des2);

        if (des1.empty() || des2.empty())
        {
            continue;
        }

        vector<DMatch> matches = matchFeatures(des1, des2);

        for (auto& m : matches)
        {
            Point2f p1 = kp1[m.queryIdx].pt;
            Point2f p2 = kp2[m.trainIdx].pt;
            float dist = norm(p2 - p1);

            TrackEvidence track;
            track.first_point = p1;
            track.last_point = p2;
            track.observed_steps = 1;
            track.accumulated_flow = dist;
            result.observed_tracks.push_back(track);

            if (dist > config_.min_flow_magnitude)
            {
                track.foreground_steps = 1;
                result.moving_tracks.push_back(track);
            }
        }
    }

    if (debug_note)
    {
        if (result.observed_tracks.empty()) *debug_note = "no_valid_tracks";
        else if (result.moving_tracks.empty()) *debug_note = "no_moving_tracks";
        else *debug_note = "simple_sift_matching";
    }

    return result;
}

}  // namespace midterm
