#include "project.h"
#include "network_base.h"
#include "imgui.h"
#include "global.h"
#include <chrono>

void create_enet_thread(EnetContext* server, ConnectedServer* my_servers, INDIGOSignalBuilder* indigo_signal_builder, bool* quit_enet, bool* cbot_trigger_stop_recording)
{
    auto last_pose_send = std::chrono::steady_clock::now();
    uint64_t last_pose_seq = 0;
    while(!(*quit_enet)) {
        // forward the latest yolo detections to indigo (~5 Hz, channel 1)
        auto now = std::chrono::steady_clock::now();
        if (indigo_signal_builder->indigo_connection != nullptr &&
            std::chrono::duration_cast<std::chrono::milliseconds>(now - last_pose_send).count() >= 200) {

            DetectedPose ball, mouse;
            uint64_t seq;
            {
                const std::lock_guard<std::mutex> lock(g_detected_poses.mtx);
                ball = g_detected_poses.ball;
                mouse = g_detected_poses.mouse;
                seq = g_detected_poses.seq;
            }
            if (seq != last_pose_seq && (ball.valid || mouse.valid)) {
                send_indigo_ball_pose(indigo_signal_builder->indigo_connection,
                                      ball.valid ? ball.x : -1000.0f,
                                      ball.valid ? ball.y : -1000.0f,
                                      ball.prob,
                                      mouse.valid ? mouse.x : -1000.0f,
                                      mouse.valid ? mouse.y : -1000.0f,
                                      mouse.prob);
                last_pose_seq = seq;
            }
            last_pose_send = now;
        }

        service_network(server, ImGui::GetIO().DeltaTime, [&](const ENetEvent& evnt)
        {
            switch (evnt.type)
            {
            case ENET_EVENT_TYPE_CONNECT:
                printf ("A new client connected from %x:%u.\n", evnt.peer -> address.host, evnt.peer -> address.port);
                break;

            case ENET_EVENT_TYPE_RECEIVE:
                {
                    uint8_t* buffer_pointer = evnt.packet->data;
                    auto server_control = FetchGame::GetServer(buffer_pointer);
                    
                    if (server_control->signal_type() == FetchGame::SignalType_ClientBringup) {
                        for (int i = 0; i < 2; i++) {
                            if (my_servers[i].peer == evnt.peer) {
                                auto server_name = server_control->server_mesg()->server_name()->c_str();
                                auto server_num_cameras = server_control->server_mesg()->num_cameras();
                                auto server_state = server_control->server_state();
                                my_servers[i].num_cameras = server_num_cameras;
                                my_servers[i].server_state = server_state;
                            }                           
                        }                        
                    } else if (server_control->signal_type() == FetchGame::SignalType_INDIGO) {
                        indigo_signal_builder->indigo_connection = evnt.peer;
                    } else if (server_control->signal_type() == FetchGame::SignalType_CalibrationPoseReached) {
                        std::cout << "From Indigo: Calibration pose reached." << std::endl;
                        calib_state = CalibPoseReached;
                    } else if (server_control->signal_type() == FetchGame::SignalType_CalibrationDone) {
                        std::cout << "From Indigo: Calibration done." << std::endl;
                        calib_state = CalibIdle;
                    } else if (server_control->signal_type() == FetchGame::SignalType_INDIGO_STOP_RECORD) {
                        std::cout << "From Indigo: <Stop Recording> trigger received." << std::endl;
                        *cbot_trigger_stop_recording = true;
                    }

                    else {
                        for (int i = 0; i < 2; i++) {
                            if (my_servers[i].peer == evnt.peer) {
                                auto server_state = server_control->server_state();
                                my_servers[i].server_state = server_state;
                            }
                        }
                    }
                    enet_packet_destroy(evnt.packet);
                }
                break;

            case ENET_EVENT_TYPE_DISCONNECT:
                printf("- Client %d has disconnected.\n", evnt.peer->incomingPeerID);
                break;
            }
        });
        usleep(10);
    }
}
