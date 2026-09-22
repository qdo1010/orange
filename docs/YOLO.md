## Deploy Realtime YOLO models

Adapted from [YOLOv8-TensorRT](https://github.com/triple-Mu/YOLOv8-TensorRT), we deploy realtime YOLO models via [Nividia TensorRT](https://developer.nvidia.com/tensorrt). 

### Preparation
You will need a `mp4` recording to create the training data for your YOLO model. 

Label and create a YOLOv8 compatible dataset using frames from your recording. 

### Compiling the model
Clone the repo: 
```
cd ~/src/
git clone https://github.com/triple-Mu/YOLOv8-TensorRT
cd YOLOv8-TensorRT
```

Create a venv and activate it:
```
python3 -m venv .venv
```

Now open the folder where you previously downloaded the YOLO dataset. 

Source the venv from earlier and install ultralytics:
```
source ~/src/YOLOv8-TensorRT/.venv/bin/activate
pip install ultralytics
```

Now we want to train the YOLO model. Download your preferred model size from the [yolov8 huggingface](https://huggingface.co/Ultralytics/YOLOv8/tree/main). Note that larger model sizes will result in higher latency in predictions. We can then train our model on our custom dataset: 
```
yolo task=detect mode=train model=yolov8m.pt data=data.yaml epochs=100 imgsz=640
```

This should give you a `.pt` file in `runs/detect/train/weights`. We want to use `best.pt`. Copy this file into the cloned repo from earlier: 

```
cd runs/detect/train/weights
cp ./best.pt ~/src/YOLOv8-TensorRT/best.pt
cd ~/src/YOLOv8-TensorRT/
```

Now we're ready to compile into an engine file. Install the requirements:
```
pip install -r requirements.txt
pip install tensorrt==10.11.0.33
```

Now we convert the `.pt` to `.onnx`:
```
python3 export-det.py \
--weights best.pt \
--iou-thres 0.65 \
--conf-thres 0.25 \
--topk 100 \
--opset 11 \
--sim \
--input-shape 1 3 640 640 \
--device cuda:0
```
Change the arguments to fit your use case. IOU treshold and confidence threshold can be adjusted to each setup. Topk sets the maximum amount of allowed bounding boxes to be drawn. 

Finally, use `trtexec` to compile `.onnx` to `.engine`:
```
~/nvidia/TensorRT/bin/trtexec --onnx=best.onnx --saveEngine=best.engine --device=0 --fp16
```

When running `trtexec`, the device used to compile the engine will be shown, as well as a list of devices available. Take note of which device your video output is connected to. You can change which device is used for compilation by changing the flag. 

We can move this `.engine` file into the default directory that orange looks at:
```
cp ./best.engine ~/orange_data/detect/best.engine
```

Finally, in the configuration file for the cameras, located in `~/orange_data/config/local/<CONFIG_NAME>`, update all cameras' `gpu_id` in the json files for the cameras you want to use the model with to match the id of the GPU that you used for compilation. 
## Choosing between the detect engine and the native OBB engine

Each camera can hold two engines and a dropdown decides which one runs.

The camera configs the GUI actually reads are `~/orange_data/config/network/<rig>/<serial>.json`
and `~/orange_data/config/local/<rig>/<serial>.json` (both folder trees are listed in the
config picker). `lime/config/` in the repo is only a sample. If `yolo_obb` is absent from a
camera config, it falls back to the built-in default `kDefaultYoloOBBEngine` in
`src/video_capture.h`, so the "OBB (new)" entry still works on older configs:


| Config key | GUI field (Camera Property) | Meaning |
|---|---|---|
| `"yolo"` | `YOLO` | Original axis-aligned **detect** engine (unchanged behaviour, `theta = 0` in the cbot message). |
| `"yolo_obb"` | `YOLO OBB` | Native **oriented-box** engine trained with `TrainYOLO/scripts/train_obb_cyl.py` and exported with `pt_to_engine.py --task obb`. |
| `"yolo_net"` | `YOLO net` dropdown (also next to the detect-mode combo in the camera table) | `"detect"` (default) or `"obb"`. Which engine `Detect2D_GLThread` loads. Changing it takes effect the next time detection is started. |
| `"yolo_obb_label_map"` | – | Remaps the OBB net's class ids to cbot labels (0=Mouse, 1=SideCyl, 2=VertCyl). The small-cylinder dataset is `0=vert_cyl, 1=side_cyl`, so the default is `[2, 1]`. Ids outside the map are dropped. |

Example:

```json
"yolo": "/home/ratan/yolo_model/obb.engine",
"yolo_obb": "/home/ratan/yolo_model/small_cyl_obb.engine",
"yolo_net": "obb",
"yolo_obb_label_map": [2, 1]
```

How the OBB engine is handled (`src/yolov8_det.cpp::postprocess`): the engine's
output tensor is `[1, max_det, 7] = cx cy w h conf cls angle_rad` in letterboxed
input pixels. It is un-letterboxed, the long axis is put first (`rw >= rh`), and
the angle is stored on `Bbox` as `theta_deg` in `[0, 180)`: direction of the long
axis in image coordinates (x right, y down, measured from +x toward +y). `Bbox::rect`
holds the axis-aligned bounding rect of the rotated box so the existing confidence,
arena and brightness gates work unchanged. The rotated box is drawn on the stream
and sent to cbot as `obb(cx, cy, rw, rh, theta_deg, label)`. The input size is read
from the engine, so a 1280x1280 OBB engine and a 640x640 detect engine coexist.

Offline check of any engine on a video, without a window:

```bash
LD_LIBRARY_PATH=~/nvidia/TensorRT/lib ./targets/yolo_offline <engine> <video.mp4> <gpu_id> nogui 100
# prints: det frame=N label=L conf=C obb cx= cy= w= h= theta=   (or "box x= y= w= h=")
```

**Channel order.** The old detect engine is fed BGR (unchanged). The OBB engine is fed
**RGB** (`rgba2rgb_convert`), because Ultralytics trains and exports on RGB; on these
sepia-tinted frames feeding BGR to the OBB model halves its recall. `yolo_offline` takes
`bgr|rgb` as its 6th argument for the same reason.
