from super_gradients.common.object_names import Models
from super_gradients.training import models

model = models.get(Models.YOLO_NAS_POSE_S, pretrained_weights="coco_pose")
export_result = model.export("yolo_nas_pose_s.onnx")

import cv2
import numpy as np
from super_gradients.training.utils.media.image import load_image
import onnxruntime

image = load_image("https://deci-pretrained-models.s3.amazonaws.com/sample_images/beatles-abbeyroad.jpg")
image = cv2.resize(image, (export_result.input_image_shape[1], export_result.input_image_shape[0]))
image_bchw = np.transpose(np.expand_dims(image, 0), (0, 3, 1, 2))

session = onnxruntime.InferenceSession(export_result.output,
                                       providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
inputs = [o.name for o in session.get_inputs()]
outputs = [o.name for o in session.get_outputs()]
result = session.run(outputs, {inputs[0]: image_bchw})

# result[0].shape, result[1].shape, result[2].shape, result[3].shape


def iterate_over_batch_predictions(predictions, batch_size):
    num_detections, batch_boxes, batch_scores, batch_joints = predictions
    for image_index in range(batch_size):
        num_detection_in_image = num_detections[image_index, 0]

        pred_scores = batch_scores[image_index, :num_detection_in_image]
        pred_boxes = batch_boxes[image_index, :num_detection_in_image]
        pred_joints = batch_joints[image_index, :num_detection_in_image].reshape((len(pred_scores), -1, 3))

        yield image_index, pred_boxes, pred_scores, pred_joints


from super_gradients.training.utils.visualization.pose_estimation import PoseVisualization
import matplotlib.pyplot as plt


def show_predictions_from_batch_format(image, predictions):
    # In this tutorial we are using batch size of 1, therefore we are getting only first element of the predictions
    image_index, pred_boxes, pred_scores, pred_joints = next(iter(iterate_over_batch_predictions(predictions, 1)))

    image = PoseVisualization.draw_poses(
        image=image, poses=pred_joints, scores=pred_scores, boxes=pred_boxes,
        edge_links=None, edge_colors=None, keypoint_colors=None, is_crowd=None
    )

    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.tight_layout()
    plt.show()


show_predictions_from_batch_format(image, result)
