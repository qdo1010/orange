#include "kernel.cuh"
#include "opencv2/opencv.hpp"
#include "utils.h"
#include "yolov8_det.h"
#include <string> // for std::stoi

const std::vector<std::string> CLASS_NAMES = {"rat"};
const std::vector<std::vector<unsigned int>> COLORS = {{255, 0, 255}};

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr,
                "Usage: %s [engine_path] [video_path] [gpu_id] [nogui] [max_frames] [bgr|rgb]\n"
                "  nogui      : print detections to stdout, no window\n"
                "  max_frames : stop after this many frames (0 = all)\n"
                "  bgr|rgb    : channel order fed to the engine (default bgr = old detect\n"
                "               engines; use rgb for Ultralytics OBB engines, as lime does)\n",
                argv[0]);
        return -1;
    }
    const bool headless = (argc >= 5 && std::string(argv[4]) == "nogui");
    const long max_frames = (argc >= 6) ? std::stol(argv[5]) : 0;
    const bool feed_rgb = (argc >= 7 && std::string(argv[6]) == "rgb");
    long frame_idx = 0;

    int device_id = std::stoi(argv[3]);
    // cuda:0
    cudaSetDevice(device_id);

    const std::string engine_file_path{argv[1]};
    const std::string input_video{argv[2]};

    cv::VideoCapture cap(input_video);
    if (!cap.isOpened()) {
        printf("Cannot open %s\n", input_video.c_str());
        return -1;
    }

    // Get video frame width and height
    int camera_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int camera_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    std::cout << "Video Width: " << camera_width << std::endl;
    std::cout << "Video Height: " << camera_height << std::endl;

    unsigned int skeleton[8] = {0, 1, 1, 2, 2, 3, 3, 0}; // box

    float *d_points;
    unsigned int *d_skeleton;
    unsigned char *d_frame;
    // get input size from the video
    cv::Mat image;

    printf("YOLO initialization...\n");
    int frame_size = camera_width * camera_height * 3;
    CHECK(cudaMalloc((void **)&d_frame, frame_size));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    NppStreamContext npp_ctx = make_npp_stream_context(device_id, stream);
    YOLOv8 *yolov8 = new YOLOv8(engine_file_path, camera_width, camera_height,
                                stream, d_frame, npp_ctx);
    yolov8->make_pipe(false);

    cudaMalloc((void **)&d_points, sizeof(float) * 8);
    cudaMalloc((void **)&d_skeleton, sizeof(unsigned int) * 8);
    CHECK(cudaMemcpy(d_skeleton, skeleton, sizeof(unsigned int) * 8,
                     cudaMemcpyHostToDevice));

    std::vector<Bbox> objs;
    unsigned char *frame_draw =
        (unsigned char *)malloc(frame_size * sizeof(unsigned char));

    cv::Mat view;
    cv::Mat final_view;
    while (cap.read(image)) {

        auto start = std::chrono::high_resolution_clock::now();
        if (feed_rgb) cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
        CHECK(cudaMemcpy(d_frame, (uint8_t *)image.data, frame_size,
                         cudaMemcpyHostToDevice));
        cudaDeviceSynchronize();
        auto stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = stop - start;
        std::cout << "copy frame from cpu to gpu:  " << elapsed.count() << " ms"
                  << std::endl;

        start = std::chrono::high_resolution_clock::now();

        if (yolov8->graph_captured) {
            // nvtxRangePush("graph");
            CHECK(cudaGraphLaunch(yolov8->inference_graph_exec, stream));
            CHECK(cudaStreamSynchronize(stream));
            // nvtxRangePop();
        } else {
            yolov8->preprocess_gpu();
            yolov8->infer(); // it sync gpu with cpu here
        }
        yolov8->postprocess(objs);
        if (headless) {
            // One line per detection: frame label conf then either
            // obb cx cy w h theta_deg  or  box x y w h.
            for (const auto &o : objs) {
                if (o.has_obb)
                    printf("det frame=%ld label=%d conf=%.3f obb cx=%.1f cy=%.1f w=%.1f h=%.1f theta=%.1f\n",
                           frame_idx, o.label, o.prob, o.cx, o.cy, o.rw, o.rh, o.theta_deg);
                else
                    printf("det frame=%ld label=%d conf=%.3f box x=%.1f y=%.1f w=%.1f h=%.1f\n",
                           frame_idx, o.label, o.prob, o.rect.x, o.rect.y, o.rect.width, o.rect.height);
            }
            if (objs.empty()) printf("det frame=%ld none\n", frame_idx);
        }
        yolov8->copy_keypoints_gpu(d_points, objs);
        cudaDeviceSynchronize();
        stop = std::chrono::high_resolution_clock::now();
        elapsed = stop - start;
        std::cout << "yolo pre/infer/post time:  " << elapsed.count() << " ms"
                  << std::endl;

        gpu_draw_rat_pose(d_frame, camera_width, camera_height, d_points,
                          d_skeleton, yolov8->stream, 3);
        // copy frame back for opencv visualization
        cudaMemcpy2D(frame_draw, camera_width * 3, d_frame, camera_width * 3,
                     camera_width * 3, camera_height, cudaMemcpyDeviceToHost);
        view = cv::Mat(camera_width * camera_height * 3, 1, CV_8U, frame_draw)
                   .reshape(3, camera_height);
        // yolov8->draw_objects(image, view, objs, CLASS_NAMES, COLORS);
        float r = 0.5;
        int output_w = std::round(camera_width * r);
        int output_h = std::round(camera_height * r);
        cv::resize(view, final_view, cv::Size(output_w, output_h));

        frame_idx++;
        if (max_frames > 0 && frame_idx >= max_frames) break;
        if (headless) continue;
        cv::imshow(engine_file_path.c_str(), final_view);
        if (cv::waitKey(10) == 'q') {
            break;
        }
    }
}
