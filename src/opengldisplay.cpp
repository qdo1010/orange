#include "video_capture.h"
#if defined(__GNUC__)
#include <unistd.h>
#endif
#include "global.h"
#include "kernel.cuh"
#include "opengldisplay.h"
#include "utils.h"
#include "obj_generated.h"
#include <cuda_runtime_api.h>
#include <nvToolsExt.h>
#include <stdio.h>
#include <string.h>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <iostream>

// Mean intensity (0-255) of the debayered RGBA frame inside a detection box.
// Used to reject dim reflections (~150) vs bright real objects (~210+).
static float box_mean_brightness(const unsigned char *d_rgba, int W, int H,
                                 const cv::Rect_<float> &rect) {
    int bx = std::max(0, (int)rect.x);
    int by = std::max(0, (int)rect.y);
    int bw = std::min(W - bx, (int)rect.width);
    int bh = std::min(H - by, (int)rect.height);
    if (bw <= 0 || bh <= 0)
        return 0.0f;
    std::vector<unsigned char> buf((size_t)bw * bh * 4);
    cudaMemcpy2D(buf.data(), (size_t)bw * 4,
                 d_rgba + ((size_t)by * W + bx) * 4, (size_t)W * 4,
                 (size_t)bw * 4, bh, cudaMemcpyDeviceToHost);
    double sum = 0;
    size_t n = (size_t)bw * bh;
    for (size_t i = 0; i < n; i++)
        sum += buf[i * 4] + buf[i * 4 + 1] + buf[i * 4 + 2];
    return (float)(sum / (3.0 * n));
}

// Detect the arena (a large static grey rectangle) in a grayscale frame.
// Returns its boundary as a convex polygon. The arena must dominate the frame
// (>= 10% area) to be accepted, which rejects smaller bright regions such as
// reflections.
static bool detect_arena(const cv::Mat &gray, std::vector<cv::Point> &out_poly) {
    cv::Mat blur, bw;
    cv::GaussianBlur(gray, blur, cv::Size(7, 7), 0);
    // Otsu separates the bright grey arena floor from the darker surround.
    cv::threshold(blur, bw, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    cv::Mat k = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(15, 15));
    cv::morphologyEx(bw, bw, cv::MORPH_CLOSE, k);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(bw, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty())
        return false;

    int best = -1;
    double best_area = 0;
    for (size_t i = 0; i < contours.size(); i++) {
        double a = cv::contourArea(contours[i]);
        if (a > best_area) {
            best_area = a;
            best = (int)i;
        }
    }
    if (best < 0)
        return false;
    if (best_area / (double)(gray.cols * gray.rows) < 0.10)
        return false; // arena should dominate the frame

    std::vector<cv::Point> hull;
    cv::convexHull(contours[best], hull);
    std::vector<cv::Point> approx;
    double peri = cv::arcLength(hull, true);
    cv::approxPolyDP(hull, approx, 0.02 * peri, true);
    out_poly = (approx.size() >= 4) ? approx : hull;
    return true;
}

COpenGLDisplay::COpenGLDisplay(const char *name, CameraParams *camera_params,
                               CameraEachSelect *camera_select,
                               unsigned char *display_buffer,
                               INDIGOSignalBuilder *indigo_signal_builder)
    : CThreadWorker(name), camera_params(camera_params),
      camera_select(camera_select), display_buffer(display_buffer),
      indigo_signal_builder(indigo_signal_builder) {
    input_image_size.width = camera_params->width;
    input_image_size.height = camera_params->height;
    input_image_roi.x = 0;
    input_image_roi.y = 0;
    input_image_roi.width = camera_params->width;
    input_image_roi.height = camera_params->height;

    output_image_size.width =
        int(camera_params->width / camera_select->downsample);
    output_image_size.height =
        int(camera_params->height / camera_select->downsample);

    output_image_roi.x = 0;
    output_image_roi.y = 0;
    output_image_roi.width = output_image_size.width;
    output_image_roi.height = output_image_size.height;

    memset(workerEntries, 0, sizeof(workerEntries));
    workerEntriesFreeQueueCount = WORK_ENTRIES_MAX;
    for (int i = 0; i < workerEntriesFreeQueueCount; i++) {
        workerEntriesFreeQueue[i] = &workerEntries[i];
    }
    
    if (camera_select->detect_mode == Detect2D_GLThread &&
        camera_select->yolo_model.empty()) {
        std::cerr << "COpenGLDisplay: Detect2D_GLThread selected but YOLO model "
                     "path is empty for camera "
                  << camera_params->camera_serial << std::endl;
    }

    if (camera_select->enable_obb &&
        camera_select->detect_mode != Detect2D_GLThread) {
        std::cerr
            << "COpenGLDisplay: OBB is enabled for camera "
            << camera_params->camera_serial
            << " but stream overlay path only feeds YOLO boxes in Detect2D_GLThread mode."
            << std::endl;
    }

    // Initialize OBB detector if enabled
    if (camera_select->enable_obb) {
        if (camera_select->obb_csv_path.empty()) {
            std::cerr
                << "COpenGLDisplay: OBB is enabled but obb_csv_path is empty for camera "
                << camera_params->camera_serial << std::endl;
            return;
        }

        OBBDetectorParams obb_params;
        std::vector<std::string> csv_paths = {camera_select->obb_csv_path};
        obb_detector = new OBBDetector(camera_params, csv_paths, obb_params);
        obb_detector->set_target_class(DET_SIDECYL);

        if (!obb_detector->initialize()) {
            std::cerr << "COpenGLDisplay: Failed to initialize OBB detector with CSV "
                      << camera_select->obb_csv_path << " for camera "
                      << camera_params->camera_serial << std::endl;
            delete obb_detector;
            obb_detector = nullptr;
        } else {
            std::cout << "COpenGLDisplay: OBB detector initialized for camera "
                      << camera_params->camera_serial << std::endl;
            obb_detector->start();
        }
    }
}

COpenGLDisplay::~COpenGLDisplay() {
    cudaFree(frame_original.d_orig);
    cudaFree(debayer.d_debayer);
    if (camera_select->detect_mode == Detect2D_GLThread) {
        delete yolov8;
    }
    
    if (obb_detector) {
        obb_detector->stop();
        delete obb_detector;
        obb_detector = nullptr;
    }
    
    if (d_obb_points) {
        cudaFree(d_obb_points);
        d_obb_points = nullptr;
    }

    if (d_box_points) {
        cudaFree(d_box_points);
        d_box_points = nullptr;
    }
}

void COpenGLDisplay::ThreadRunning() {
    ck(cudaSetDevice(camera_params->gpu_id));

    if (camera_select->downsample != 1) {
        ck(cudaMalloc((void **)&d_resize,
                      output_image_size.width * output_image_size.height * 4));
    }

    // innitialization
    initalize_gpu_frame(&frame_original, camera_params);
    initialize_gpu_debayer(&debayer, camera_params, 4);
    initialize_cpu_frame(&frame_cpu, camera_params);

    ck(cudaMalloc((void **)&d_convert,
                  camera_params->width * camera_params->height * 3));

    unsigned int skeleton[8] = {0, 1, 1, 2, 2, 3, 3, 0}; // box
    NppStreamContext npp_ctx =
        make_npp_stream_context(camera_params->gpu_id, 0);
    const bool yolo_glthread_enabled =
        (camera_select->detect_mode == Detect2D_GLThread);
    if (yolo_glthread_enabled) {
        printf("YOLO initialization...\n");

        const std::string engine_file_path = camera_select->yolo_model;
        yolov8 = new YOLOv8(engine_file_path, camera_params->width,
                            camera_params->height, 0, d_convert, npp_ctx);
        yolov8->make_pipe(false);

        cudaMalloc((void **)&d_points, sizeof(float) * 8);
        cudaMalloc((void **)&d_skeleton, sizeof(unsigned int) * 8);
        CHECK(cudaMemcpy(d_skeleton, skeleton, sizeof(unsigned int) * 8,
                         cudaMemcpyHostToDevice));
        // Buffer for axis-aligned boxes (Mouse / VertCyl). One box at a time.
        cudaMalloc((void **)&d_box_points, sizeof(float) * 8);
    }
    
    // Allocate OBB GPU resources if needed.
    // OBB refinement DISABLED: cbot uses the CURRICULUM angle (not the detected angle),
    // so the 2-stage OBB fitting is unnecessary. Its worker thread also races the main
    // loop (get_latest_detections vs the worker) which can delay/drop detections for the
    // whole frame -- including the mouse. Force off so ALL classes are sent as raw YOLO
    // axis-aligned boxes straight through.
    const bool obb_overlay_enabled = false;
    if (camera_select->enable_obb && obb_detector && !yolo_glthread_enabled) {
        std::cerr << "COpenGLDisplay: OBB detector is running but will not receive "
                     "YOLO boxes until detect mode is Detect2D_GLThread."
                  << std::endl;
    }
    if (obb_overlay_enabled) {
        cudaMalloc((void **)&d_obb_points, sizeof(float) * 8 * 10); // Support up to 10 OBBs
    }

    // If the config supplies an explicit arena polygon (normalized x,y pairs),
    // use it directly and skip live auto-detection. The arena is static, so a
    // fixed polygon is the most robust option.
    if (!camera_select->arena_polygon.empty()) {
        arena_poly.clear();
        const auto &ap = camera_select->arena_polygon;
        for (size_t i = 0; i + 1 < ap.size(); i += 2) {
            arena_poly.emplace_back(
                (int)(ap[i] * camera_params->width + 0.5f),
                (int)(ap[i + 1] * camera_params->height + 0.5f));
        }
        if (arena_poly.size() >= 3) {
            arena_detected = true;
            std::cout << "Arena polygon from config for "
                      << camera_params->camera_serial << " ("
                      << arena_poly.size() << " pts)" << std::endl;
        }
    }

    std::vector<Bbox> objs;
    std::vector<Bbox> objs_last_frame;
    std::vector<OBB> obb_detections;  // Add OBB results

    using clock = std::chrono::steady_clock;
    
    // Static variables for OBB lock-in system tracking
    static std::vector<OBB> last_locked_detections;
    static bool last_detections_sent = false;
    
    // Static variables for persistent slot tracking
    static int persistent_slot_assignments[2] = {-1, -1};  // Object IDs assigned to each slot
    static bool slots_initialized = false;
    
    // OBB Background Building

    int frameCount = 0;
    auto lastFPSUpdate = clock::now();
    auto lastDetectionStatsUpdate = clock::now();
    uint64_t yolo_frame_counter = 0;
    uint64_t yolo_nonempty_counter = 0;
    uint64_t obb_nonempty_counter = 0;
    size_t last_yolo_obj_count = 0;
    size_t last_obb_obj_count = 0;

    while (IsMachineOn()) {
        auto frameStart = clock::now();
        std::chrono::duration<double, std::milli> targetFrameDuration(
            1000.0 / streaming_target_fps.load());
        void *f = GetObjectFromQueueIn();
        if (f) {
            WORKER_ENTRY entry = *(WORKER_ENTRY *)f;
            PutObjectToQueueOut(f);

            // nvtxRangePush("display_gl_copy_debayer");
            // copy frame from cpu to gpu
            CHECK(cudaMemcpy2D(frame_original.d_orig, camera_params->width,
                               entry.imagePtr, camera_params->width,
                               camera_params->width, camera_params->height,
                               cudaMemcpyHostToDevice));

            if (camera_params->color) {
                debayer_frame_gpu(camera_params, &frame_original, &debayer);
            } else {
                duplicate_channel_gpu(camera_params, &frame_original, &debayer);
            }
            // nvtxRangePop();

            if (yolo_glthread_enabled) {
                // One-time arena detection: copy a debayered frame to the CPU
                // and locate the static grey arena. Retries each frame until it
                // succeeds (e.g. once the stream produces a real image).
                if (!arena_detected) {
                    cv::Mat rgba(camera_params->height, camera_params->width,
                                 CV_8UC4);
                    CHECK(cudaMemcpy(rgba.data, debayer.d_debayer,
                                     (size_t)camera_params->width *
                                         camera_params->height * 4,
                                     cudaMemcpyDeviceToHost));
                    cv::Mat gray;
                    cv::cvtColor(rgba, gray, cv::COLOR_RGBA2GRAY);
                    if (detect_arena(gray, arena_poly)) {
                        arena_detected = true;
                        std::cout << "Arena detected for "
                                  << camera_params->camera_serial << " ("
                                  << arena_poly.size() << " pts)" << std::endl;
                    }
                }

                // Match yolo_offline input layout (OpenCV frames are BGR).
                rgba2bgr_convert(d_convert, debayer.d_debayer,
                                 camera_params->width, camera_params->height,
                                 0);

                if (yolov8->graph_captured) {
                    // nvtxRangePush("graph");
                    CHECK(cudaGraphLaunch(yolov8->inference_graph_exec, 0));
                    CHECK(cudaStreamSynchronize(0));
                    // nvtxRangePop();
                } else {
                    yolov8->preprocess_gpu();
                    yolov8->infer(); // it sync gpu with cpu here
                }

                yolov8->postprocess(objs);

                // Reject false positives (reflections, off-arena, low score):
                //   1) confidence gate, 2) arena ROI, 3) box brightness.
                if (!objs.empty()) {
                    static uint64_t gate_dbg = 0;
                    std::vector<Bbox> kept;
                    kept.reserve(objs.size());
                    for (const auto &b : objs) {
                        bool ok_conf =
                            (camera_select->min_confidence <= 0.0f) ||
                            (b.prob >= camera_select->min_confidence);
                        // Arena ROI gate is a rectangle tuned to suppress
                        // reflections off the old square TABLE. It is OFF by
                        // default because on a CIRCLE arena the rectangle clips
                        // ~75% of the circular edge and rejects rim mice. Set
                        // "use_arena_gate": true in the config for square-table rigs.
                        bool ok_arena = true;
                        if (camera_select->use_arena_gate && arena_detected &&
                            !arena_poly.empty()) {
                            cv::Point2f ctr(b.rect.x + b.rect.width * 0.5f,
                                            b.rect.y + b.rect.height * 0.5f);
                            ok_arena =
                                cv::pointPolygonTest(arena_poly, ctr, false) >= 0;
                        }
                        float bright = -1.0f;
                        bool ok_bright = true;
                        // Brightness gate suppresses bright-cylinder false
                        // positives -- but the mouse is a DARK object, so applying
                        // it to Mouse would always reject the animal. Skip Mouse.
                        if (camera_select->min_brightness > 0.0f &&
                            b.label != DET_MOUSE) {
                            bright = box_mean_brightness(
                                debayer.d_debayer, camera_params->width,
                                camera_params->height, b.rect);
                            ok_bright = bright >= camera_select->min_brightness;
                        }
                        if (gate_dbg < 40) {
                            std::cout << "  gate[" << camera_params->camera_serial
                                      << "] label=" << b.label
                                      << " conf=" << b.prob
                                      << " bright=" << bright
                                      << " arena=" << ok_arena
                                      << " -> "
                                      << ((ok_conf && ok_arena && ok_bright)
                                              ? "KEEP"
                                              : "REJECT")
                                      << std::endl;
                            gate_dbg++;
                        }
                        if (ok_conf && ok_arena && ok_bright)
                            kept.push_back(b);
                    }
                    objs.swap(kept);
                }
                yolo_frame_counter++;
                last_yolo_obj_count = objs.size();
                if (!objs.empty()) {
                    yolo_nonempty_counter++;
                    // Print a few early detections to validate runtime coordinates.
                    if (yolo_nonempty_counter <= 5) {
                        const auto &b0 = objs[0];
                        std::cout << "YOLO sample [" << camera_params->camera_serial
                                  << "]: n=" << objs.size() << " first=("
                                  << b0.rect.x << "," << b0.rect.y << ","
                                  << b0.rect.width << "," << b0.rect.height
                                  << ") conf=" << b0.prob
                                  << " label=" << b0.label << std::endl;
                    }
                }
                if (objs.size() > 0) {

                    // std::cout << objs[0].rect.x << ", " << objs[0].rect.y <<
                    // std::endl; f32 bbox_center_x = objs[0].rect.x +
                    // objs[0].rect.width / 2.0; std::cout << bbox_center_x <<
                    // std::endl; if (objs[0].rect.x < 2260.41 && objs[0].rect.x
                    // < objs_last_frame[0].rect.x) { if (objs[0].rect.x <
                    // 2500.0 && objs[0].rect.x > 2100.0) {
                    if (objs[0].rect.x < 2600.0 &&
                        objs[0].rect.x > 2100.0) { // trigger earlier
                        // std::cout << "trigger ball drop" << std::endl;
                        if (indigo_signal_builder->indigo_connection != NULL) {
                            send_indigo_message(
                                indigo_signal_builder->server,
                                indigo_signal_builder->builder,
                                indigo_signal_builder->indigo_connection,
                                FetchGame::SignalType_INDIGO_TRIAL_TRIGGER);
                        }
                    }
                    objs_last_frame.push_back(objs[0]);
                } else {
                    objs_last_frame.clear();
                }
            }
            
            // Class-aware detection split:
            //   SideCyl (1)             -> 2-stage OBB routine below.
            //   Mouse (0) + VertCyl (2) -> plain axis-aligned box, drawn AFTER
            //                              the OBB block so the overlay graphics
            //                              don't pollute the frame the OBB CV
            //                              refinement reads.
            // SideCyl is treated as plain when OBB refinement is unavailable.
            std::vector<Bbox> sidecyl_objs;
            std::vector<Bbox> plain_objs;
            std::vector<OBB> sidecyl_obbs;  // oriented SideCyl results for msg
            if (yolo_glthread_enabled) {
                for (const auto &b : objs) {
                    if (b.label == DET_SIDECYL && obb_overlay_enabled)
                        sidecyl_objs.push_back(b);
                    else
                        plain_objs.push_back(b);
                }
            }

            // OBB Detection (SideCyl only)
            if (obb_overlay_enabled) {
                std::vector<OBB> obb_detections;

                if (sidecyl_objs.empty()) {
                    // No YOLO detection this frame — no OBBs. Keep the worker
                    // in sync (so it won't return stale data on the next
                    // frame that does have a detection) but don't consult it
                    // now; get_latest_detections() races the worker and
                    // would return the previous frame's box.
                    obb_detector->set_yolo_boxes(sidecyl_objs);
                } else if (yolov8 && yolov8->has_mask_protos() &&
                           !sidecyl_objs[0].mask_coeffs.empty()) {
                    // Seg-mask path: compute initial OBBs, hand to worker
                    // for iterative edge optimization.
                    auto seg_obbs = obb_detector->refine_from_seg_masks(
                        sidecyl_objs,
                        yolov8->get_mask_protos(),
                        yolov8->get_mask_proto_h(),
                        yolov8->get_mask_proto_w(),
                        yolov8->get_mask_num_protos(),
                        yolov8->pparam,
                        DET_SIDECYL);
                    obb_detector->set_seg_obbs(seg_obbs);
                    obb_detector->notify_frame_ready(debayer.d_debayer, 0);
                    obb_detections = obb_detector->get_latest_detections();
                } else {
                    // Fallback: two-stage CV-based refinement
                    obb_detector->set_yolo_boxes(sidecyl_objs);
                    obb_detector->notify_frame_ready(debayer.d_debayer, 0);
                    obb_detections = obb_detector->get_latest_detections();
                }
                last_obb_obj_count = obb_detections.size();
                if (!obb_detections.empty()) {
                    obb_nonempty_counter++;
                }
                
                if (obb_detections.size() > 0) {
                    // Check if detections changed significantly
                    bool detections_changed = (last_locked_detections.size() != obb_detections.size());
                    if (!detections_changed) {
                        for (size_t i = 0; i < obb_detections.size() && i < last_locked_detections.size(); i++) {
                            auto cur = obb_detector->obb_to_xywhr(obb_detections[i]);
                            auto prev = obb_detector->obb_to_xywhr(last_locked_detections[i]);
                            if (std::hypot(cur.x - prev.x, cur.y - prev.y) > 50.0f) {
                                detections_changed = true;
                                break;
                            }
                        }
                    }
                    
                    if (detections_changed || !last_detections_sent) {
                        last_locked_detections = obb_detections;
                        last_detections_sent = true;
                        
                        if (detections_changed) {
                            slots_initialized = false;
                            persistent_slot_assignments[0] = -1;
                            persistent_slot_assignments[1] = -1;
                        }
                    }
                    
                    // Draw OBBs on video stream
                    const auto& stable_detections = last_locked_detections;
                    for (size_t i = 0; i < stable_detections.size() && i < 10; i++) {
                        const OBB& obb = stable_detections[i];
                        float obb_points[8] = {
                            obb.x1, obb.y1, obb.x2, obb.y2,
                            obb.x3, obb.y3, obb.x4, obb.y4
                        };
                        CHECK(cudaMemcpyAsync(d_obb_points + i * 8, obb_points,
                                             sizeof(float) * 8, cudaMemcpyHostToDevice, 0));
                        gpu_draw_obb(debayer.d_debayer, camera_params->width,
                                    camera_params->height, d_obb_points + i * 8,
                                    obb.class_id, 0, 255, 0, 0);
                    }
                    
                } else {
                    last_locked_detections.clear();
                    last_detections_sent = false;
                }
                // Oriented SideCyl results to fold into the outgoing message.
                sidecyl_obbs = last_locked_detections;
            }

            // Draw axis-aligned boxes for Mouse / VertCyl (and SideCyl when OBB
            // is disabled). Done after the OBB block so these overlay lines are
            // not present in the frame the OBB CV refinement reads.
            for (const auto &b : plain_objs) {
                const auto &r = b.rect;
                float corners[8] = {
                    r.x,           r.y,
                    r.x + r.width, r.y,
                    r.x + r.width, r.y + r.height,
                    r.x,           r.y + r.height};
                CHECK(cudaMemcpyAsync(d_box_points, corners, sizeof(float) * 8,
                                      cudaMemcpyHostToDevice, 0));
                gpu_draw_box(debayer.d_debayer, camera_params->width,
                             camera_params->height, d_box_points, b.label, 0);
            }

            // Send all detected objects to cbot as a flexible list. Each entry
            // carries its class label (0=Mouse,1=SideCyl,2=VertCyl) and image
            // center; SideCyls carry their oriented angle, others theta=0.
            // cbot assigns left/right per class and renders in MuJoCo.
            if (yolo_glthread_enabled && indigo_signal_builder->indigo_connection) {
                flatbuffers::FlatBufferBuilder *fb = indigo_signal_builder->builder;
                fb->Clear();
                std::vector<::flatbuffers::Offset<Obj::obb>> fb_objs;
                // Mouse / VertCyl (and SideCyl when OBB off): axis-aligned.
                for (const auto &b : plain_objs) {
                    float cx = b.rect.x + b.rect.width * 0.5f;
                    float cy = b.rect.y + b.rect.height * 0.5f;
                    fb_objs.push_back(Obj::Createobb(*fb, cx, cy, b.rect.width,
                                                     b.rect.height, 0.0f,
                                                     (float)b.label));
                }
                // SideCyl: oriented box from the OBB stage.
                for (const auto &obb : sidecyl_obbs) {
                    auto x = obb_detector->obb_to_xywhr(obb);
                    fb_objs.push_back(Obj::Createobb(*fb, x.x, x.y, x.w, x.h,
                                                     x.r, (float)DET_SIDECYL));
                }
                auto vec = fb->CreateVector(fb_objs);
                auto msg = Obj::Createobj_msg(*fb, vec);
                fb->Finish(msg);
                send_cbot_obj_pos2d(indigo_signal_builder->server, fb,
                                    indigo_signal_builder->indigo_connection);
            }
            // nvtxRangePush("display_gl_copy_to_interop_buffer");
            if (camera_select->downsample != 1) {
                const NppStatus npp_result = nppiResize_8u_C4R(
                    debayer.d_debayer, camera_params->width * sizeof(uchar4),
                    input_image_size, input_image_roi, (Npp8u *)d_resize,
                    output_image_size.width * sizeof(uchar4), output_image_size,
                    output_image_roi, NPPI_INTER_SUPER);
                if (npp_result != NPP_SUCCESS) {
                    std::cerr << "Error executing resize in display -- code: "
                              << npp_result << std::endl;
                }
                CHECK(cudaMemcpy2D(
                    display_buffer, output_image_size.width * 4, d_resize,
                    output_image_size.width * 4, output_image_size.width * 4,
                    output_image_size.height, cudaMemcpyDeviceToDevice));

            } else {
                CHECK(cudaMemcpy2D(
                    display_buffer, output_image_size.width * 4,
                    debayer.d_debayer, output_image_size.width * 4,
                    output_image_size.width * 4, output_image_size.height,
                    cudaMemcpyDeviceToDevice));
            }
            // nvtxRangePop();
            cudaDeviceSynchronize();
        }
        // Count frame for FPS
        frameCount++;
        auto now = clock::now();
        std::chrono::duration<double> timeSinceLastFPSUpdate =
            now - lastFPSUpdate;
        if (timeSinceLastFPSUpdate.count() >= 1.0) {
            streaming_fps.store(frameCount / timeSinceLastFPSUpdate.count());
            frameCount = 0;
            lastFPSUpdate = now;
        }
        std::chrono::duration<double> timeSinceLastDetStatsUpdate =
            now - lastDetectionStatsUpdate;
        if (timeSinceLastDetStatsUpdate.count() >= 1.0 && yolo_glthread_enabled) {
            std::cout << "Detection stats [" << camera_params->camera_serial
                      << "]: yolo_nonempty=" << yolo_nonempty_counter << "/"
                      << yolo_frame_counter << " last_yolo_n=" << last_yolo_obj_count;
            if (obb_overlay_enabled) {
                std::cout << " obb_nonempty=" << obb_nonempty_counter << "/"
                          << yolo_frame_counter << " last_obb_n="
                          << last_obb_obj_count;
            }
            std::cout << std::endl;
            lastDetectionStatsUpdate = now;
        }
        // Frame duration (GPU time included)
        std::chrono::duration<double, std::milli> frameDuration =
            now - frameStart;
        if (frameDuration < targetFrameDuration) {
            std::this_thread::sleep_for(targetFrameDuration - frameDuration);
        }
    }
}

bool COpenGLDisplay::PushToDisplay(void *imagePtr, size_t bufferSize, int width,
                                   int height, int pixelFormat,
                                   unsigned long long timestamp,
                                   unsigned long long frame_id) {
    WORKER_ENTRY *entriesOut[WORK_ENTRIES_MAX]; // entris got out from saver
                                                // thread, their frames should
                                                // be returned to driver queue.
    int entriesOutCount = WORK_ENTRIES_MAX;
    GetObjectsFromQueueOut((void **)entriesOut, &entriesOutCount);
    if (entriesOutCount) { // return the frames to driver, and put entries back
                           // to frameSaveEntriesFreeQueue
        // printf("++++++++++++++++++++++++ %s %s %d get WORKER_ENTRY from out
        // entriesOutCount: %d\n", __FILE__, __FUNCTION__, __LINE__,
        // entriesOutCount);
        for (int j = 0; j < entriesOutCount; j++) {
            workerEntriesFreeQueue[workerEntriesFreeQueueCount] = entriesOut[j];
            workerEntriesFreeQueueCount++;
        }
    }

    // get the free entry if there is one and put in to QueueIn, otherwise
    // EVT_CameraQueueFrame.
    if (workerEntriesFreeQueueCount) {
        // printf("++++++++++++++++++++++++ %s %s %d put WORKER_ENTRY to in
        // workerEntriesFreeQueueCount: %d\n", __FILE__, __FUNCTION__, __LINE__,
        // workerEntriesFreeQueueCount);
        WORKER_ENTRY *entry =
            workerEntriesFreeQueue[workerEntriesFreeQueueCount - 1];
        workerEntriesFreeQueueCount--;
        entry->imagePtr = imagePtr;
        entry->bufferSize = bufferSize;
        entry->width = width;
        entry->height = height;
        entry->pixelFormat = pixelFormat;
        entry->timestamp = timestamp;
        entry->frame_id = frame_id;
        PutObjectToQueueIn(entry);
        return true;
    }
    return false;
}
