import argparse
import multiprocessing
import time

import cv2
import numpy as np
import onnxruntime as ort
import torch
import torchvision.transforms as standard_transforms
from scipy.ndimage import gaussian_filter
from scipy.spatial import KDTree

from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords
from utils.plots import plot_only_box
from utils.torch_utils import time_synchronized


def gaussian_filter_density(img_shape, points):
    '''
    Generates a density map using k-nearest neighbors for sigma calculation.

    Args:
    - img: Input image from OpenCV (can be grayscale or color).
    - points: A list of pedestrian annotations as [[col,row], [col,row], ...].

    Returns:
    - density: A density map of the same shape as the input image but with one channel.
    '''

    density = np.zeros(img_shape, dtype=np.float32)
    gt_count = len(points)

    if gt_count == 0:
        return density

    # Build KDTree
    leafsize = 2048
    tree = KDTree(points, leafsize=leafsize)

    # Query KDTree for distances and indices of nearest neighbors
    distances, _ = tree.query(points, k=4)  # Get distances to the 4 nearest neighbors

    print('Generating density map...')
    for i, pt in enumerate(points):
        pt2d = np.zeros(img_shape, dtype=np.float32)
        x, y = int(pt[0]), int(pt[1])  # col, row format

        if 0 <= y < img_shape[0] and 0 <= x < img_shape[1]:
            pt2d[y, x] = 1.0  # Place the point on the density map

            if gt_count > 1:  # More than 1 point
                # Sum distances to the 3 nearest neighbors (skip self at distances[i][0])
                sigma = np.sum(distances[i][1:4]) * 0.1
            else:  # Single point case
                sigma = np.average(np.array(img_shape)) / 4.0  # Use average image dimension

            # Apply Gaussian filter
            density += gaussian_filter(pt2d, sigma, mode='constant')

    print('Density map generation complete.')
    return density


def process_frame(samples, session):
    ort_inputs = {'images': samples.numpy()}
    output = session.run(['output'], ort_inputs)[0]

    return torch.Tensor(output)

def postprocess_output(output, frame_orig, frame):
    # Apply NMS
    output = non_max_suppression(
        output,
        opt.conf_thres,
        opt.iou_thres,
        classes=opt.classes,
        agnostic=opt.agnostic_nms,
    )

    for i, det in enumerate(output):  # detections per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(frame.shape[2:], det[:, :4], frame_orig.shape).round()
            return det

def visualize(det, frame):

    # Print results
    for c in det[:, -1].unique():
        n = (det[:, -1] == c).sum()  # detections per class

    # Write results
    # Process the detected objects and draw bounding boxes and labels on the image
    for *xyxy, conf, cls in reversed(det):  # Loop through detections
        # Draw bounding box
        plot_only_box(xyxy, frame, color=(255, 255, 255), line_thickness=2)

    if torch.is_tensor(n):
        prediction = n.item()
    else:
        prediction = n
    cv2.putText(frame, 'Head count=' + str(prediction), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame



def main(args):
    providers = ['CPUExecutionProvider']

    sess_options = ort.SessionOptions()
    # Set graph optimization
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # Use OpenMP optimizations.
    sess_options.intra_op_num_threads = multiprocessing.cpu_count()


    session = ort.InferenceSession(args.weights, providers=providers, sess_options=sess_options)
    b, c, h, w = session.get_inputs()[0].shape

    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        #standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    cap = cv2.VideoCapture(args.source)

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            t1 = time_synchronized()
            frame_ = letterbox(frame, (h,w), scaleup=False, auto=False)[0]

            # Convert
            frame_ = transform(frame_).unsqueeze(0)
            output = process_frame(frame_, session)
            tp  = time_synchronized()

            det = postprocess_output(output, frame, frame_)
            t2 = time_synchronized()

            frame = visualize(det, frame)
            # Stream results
            cv2.imshow("output", frame)
            key = cv2.waitKey(1)  # 1 millisecond
            if key == ord('q'):
                break
            print(f'Inference: ({tp - t1:.3f}s) nms: ({t2 - tp:.3f}s) total: ({t2 - t1:.3f}s)')



            # # Generate the density map
            # density_map = gaussian_filter_density((h, w), points)
            #
            # # Visualize the density map
            # normalized_density = cv2.normalize(density_map, None, 0, 255, cv2.NORM_MINMAX)
            # normalized_density = normalized_density.astype(np.uint8)
            #
            # # Apply a colormap for better visualization
            # colored_density = cv2.applyColorMap(normalized_density, cv2.COLORMAP_JET)
            #
            # # Display the image
            # cv2.imshow("Density Map", colored_density)
            cv2.waitKey(1)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights", nargs="+", type=str, default="yolo-crowd.onnx", help="model.pt path(s)"
    )
    parser.add_argument(
        "--source", type=str, default="/home/nicola/Software/YOLO-CROWD/data/MOT16-03.mp4", help="source"
    )  # file/folder, 0 for webcam
    parser.add_argument(
        "--conf-thres", type=float, default=0.25, help="object confidence threshold"
    )
    parser.add_argument(
        "--iou-thres", type=float, default=0.45, help="IOU threshold for NMS"
    )
    parser.add_argument(
        "--device", default="cpu", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        type=int,
        help="filter by class: --class 0, or --class 0 2 3",
    )
    parser.add_argument(
        "--agnostic-nms", action="store_true", help="class-agnostic NMS"
    )

    opt = parser.parse_args()
    print(opt)

    main(opt)
