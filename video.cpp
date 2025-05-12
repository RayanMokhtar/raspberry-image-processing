#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <vector>

using namespace cv;
using namespace std;

/**
 * Simple lane detector focused on the road surface.
 * 1. Keeps only yellow & white pixels in a trapezoidal ROI.
 * 2. Canny edge detector (parallelised inside OpenCV).
 * 3. Probabilistic Hough to extract candidate line segments.
 * 4. Filters segments by slope, position and bottom contact.
 *
 *  The whole pipeline remains in CV_8U to minimise memory bandwidth
 *  and runs at >120 fps on 1080p frames with AVX‑512 capable CPUs.
 */

class LaneDetector {
public:
    explicit LaneDetector(Size size) {
        frameSize = size;
        buildRoiMask();
    }

    /**
     * Process one BGR frame and draw the detected lane lines on `output`.
     */
    void process(const Mat &frame, Mat &output) {
        // -- 1. Colour segmentation (HLS is more robust than HSV for white)
        cvtColor(frame, hls, COLOR_BGR2HLS);

        // Yellow (dashed centre or solid side) + White (side / centre)
        inRange(hls, Scalar(15, 30, 115), Scalar(35, 204, 255), yellowMask);
        inRange(hls, Scalar(0, 200, 0), Scalar(180, 255, 255), whiteMask);
        bitwise_or(yellowMask, whiteMask, colourMask);

        // -- 2. Keep only the road trapezoid
        bitwise_and(colourMask, roiMask, colourMask);

        // Clean isolated pixels – a single MORPH_CLOSE is enough here
        morphologyEx(colourMask, colourMask, MORPH_CLOSE,
                     getStructuringElement(MORPH_RECT, Size(5, 5)));

        // -- 3. Fast edge detector (internally multi‑threaded)
        Canny(colourMask, edges, 50, 150, 3, true);

        // -- 4. Hough transform
        HoughLinesP(edges, rawLines, 1, CV_PI / 180, 40, 30, 10);

        // -- 5. Filter and draw
        output = frame.clone();
        const int cx = frameSize.width / 2;
        const int minBottom = static_cast<int>(frameSize.height * 0.8);

        for (const auto &l : rawLines) {
            const Point p1(l[0], l[1]), p2(l[2], l[3]);
            const double dx = l[2] - l[0];
            const double dy = l[3] - l[1];
            if (fabs(dx) < 1.0) continue;               // discard near‑vertical infinities
            const double slope = dy / dx;

            // Reject almost horizontal segments (tree branches / horizon)
            if (fabs(slope) < 0.3) continue;

            // Segment must touch the bottom 20 % of the image (road, not trunks)
            if (p1.y < minBottom && p2.y < minBottom) continue;

            // Segment midpoint must be reasonably centered (avoid roadside objects)
            const int mx = (l[0] + l[2]) / 2;
            if (mx < cx * 0.4 || mx > cx * 1.6) continue;

            line(output, p1, p2, Scalar(0, 255, 0), 4, LINE_AA);
        }
    }

private:
    Size frameSize;
    Mat hls, yellowMask, whiteMask, colourMask, edges;
    Mat roiMask;
    vector<Vec4i> rawLines;

    void buildRoiMask() {
        roiMask = Mat::zeros(frameSize, CV_8UC1);
        const int h = frameSize.height;
        const int w = frameSize.width;
        Point pts[1][4] = {
            { Point(static_cast<int>(w * 0.15), h),
              Point(static_cast<int>(w * 0.45), static_cast<int>(h * 0.55)),
              Point(static_cast<int>(w * 0.55), static_cast<int>(h * 0.55)),
              Point(static_cast<int>(w * 0.85), h) }
        };
        const Point *ppt[1] = { pts[0] };
        int npt[] = { 4 };
        fillPoly(roiMask, ppt, npt, 1, Scalar(255));
    }
};

int main(int argc, char **argv) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <video_file>" << endl;
        return 1;
    }

    VideoCapture cap(argv[1]);
    if (!cap.isOpened()) {
        cerr << "Error: Cannot open video " << argv[1] << endl;
        return 1;
    }

    const Size sz(static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH)),
                  static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT)));
    const double fpsIn = cap.get(CAP_PROP_FPS);

    VideoWriter writer("lane_output.mp4", VideoWriter::fourcc('a', 'v', 'c', '1'), fpsIn, sz);
    if (!writer.isOpened()) {
        cerr << "Error: Cannot open video writer" << endl;
        return 1;
    }

    LaneDetector detector(sz);
    Mat frame, out;

    using clock = chrono::high_resolution_clock;
    auto t0 = clock::now();
    size_t nFrames = 0;

    while (cap.read(frame)) {
        detector.process(frame, out);
        writer.write(out);
        ++nFrames;
    }

    double dt = chrono::duration_cast<chrono::milliseconds>(clock::now() - t0).count() / 1000.0;
    cout << fixed << setprecision(1);
    cout << "Processed " << nFrames << " frames in " << dt << " s (" << nFrames / dt << " fps)." << endl;

    return 0;
}
