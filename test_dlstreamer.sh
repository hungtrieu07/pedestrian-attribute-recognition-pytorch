# gst-launch-1.0 filesrc location=vehicle.mp4 ! decodebin ! videoconvert ! \
#                 gvadetect model=openvino_models/yolov9-retrained-human-vehicle/best_int8_openvino_model/best.xml model-proc=openvino_models/yolov9-retrained-human-vehicle/best_int8_openvino_model/model-proc.json device=CPU inference-interval=1 ! queue ! \
#                 gvaclassify model=openvino_models/reid/reid_model.xml model-proc=openvino_models/reid/reid_model.json device=CPU inference-interval=1 ! queue ! \
#                 gvawatermark ! videoconvert ! autovideosink

# gst-launch-1.0 filesrc location=vehicle.mp4 ! decodebin ! videoconvert ! \
#                 gvadetect model=openvino_models/yolov9-retrained-human-vehicle/best_int8_openvino_model/best.xml model-proc=openvino_models/yolov9-retrained-human-vehicle/best_int8_openvino_model/model-proc.json device=CPU inference-interval=1 ! queue ! \
#                 gvawatermark ! videoconvert ! autovideosink

#video_path_list
# /home/hungtrieu07/dev/video_data/1740730604383.mp4
# /home/hungtrieu07/dev/video_data/1740730890898.mp4
# /home/hungtrieu07/dev/video_data/1740730950413.mp4
# /home/hungtrieu07/dev/video_data/1740731094984.mp4
# /home/hungtrieu07/dev/video_data/1740731190026.mp4
# /home/hungtrieu07/dev/video_data/test_video.avi

# GST_DEBUG=4 gst-launch-1.0 filesrc location=/home/hungtrieu07/dev/video_data/1740731094984.mp4 ! decodebin ! videoconvert ! \
#     gvadetect model=ov_exported_model/person_vehicle/FP16.xml device=CPU inference-interval=1 ! queue ! \
#     gvaclassify model=onnx_models/openvino_model/model.xml object-class="person" model-proc=ov_exported_model/person-attr/model-proc.json \
#                 device=CPU inference-interval=1 ! queue ! \
#     gvametaconvert format=json ! gvawatermark ! videoconvert ! autovideosink sync=false

GST_DEBUG=4 gst-launch-1.0 filesrc location=/home/hungtrieu07/dev/video_data/1740731094984.mp4 ! decodebin ! videoconvert ! \
    gvadetect model=ov_exported_model/person_vehicle/FP16.xml device=CPU ! \
    gvawatermark ! \
    gvaclassify model=ov_exported_model/person-attr/model.xml model-proc=ov_exported_model/person-attr/model-proc.json device=CPU ! \
    gvametaconvert format=json ! gvawatermark ! videoconvert ! autovideosink sync=false

