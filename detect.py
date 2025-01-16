import argparse
import time

import cv2
import torch
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import (check_img_size,
                           check_requirements,
                           non_max_suppression, scale_coords, set_logging)
from utils.plots import plot_only_box
from utils.torch_utils import model_info, select_device, time_synchronized


def resize_numpy_image(img, expected_height, expected_width, device):
    """
    Resize a NumPy image to the expected height and width if necessary,
    and return the image as a PyTorch tensor.
    """
    # Check if the image is in (H, W, C) format
    if img.shape[0] == 3:  # If it's in (C, H, W), convert to (H, W, C)
        img = img.transpose(1, 2, 0)

    # Check if resizing is necessary
    if img.shape[0] != expected_height or img.shape[1] != expected_width:
        print(
                f"Resizing image from {img.shape} to ({expected_height}, {expected_width})"
        )
        img_resized_np = cv2.resize(img, (expected_width, expected_height))
    else:
        img_resized_np = img  # Use original image if resizing is not needed

    # Convert the resized NumPy array back to a PyTorch tensor (C, H, W) and move to the specified device
    img_resized = (
            torch.from_numpy(img_resized_np).permute(2, 0, 1).unsqueeze(0).to(device)
    )

    return img_resized


def main(opt, save_img=False):
    source, weights, imgsz = (
            opt.source,
            opt.weights,
            opt.img_size,
    )

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != "cpu"  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max()) * 2  # model stride
    imgsz = [check_img_size(x, s=stride) for x in opt.img_size]  # verify imgsz are gs-multiples
    if half:
        model.half()  # to FP16

    model_info(model, img_size=imgsz)

    # Set Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, "module") else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != "cpu":
        model(
                torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters()))
        )  # run once
    t0 = time.time()

    for path, img, im0s, vid_cap in dataset:
        # # Check the image size
        # print("Image size:", img.shape)
        # print("im0s size:", im0s.shape)
        # # Resize the image if necessary
        # img = resize_numpy_image(img, imgsz, imgsz, device)

        img = torch.from_numpy(img).to(device)

        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        tp = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(
                pred,
                opt.conf_thres,
                opt.iou_thres,
                classes=opt.classes,
                agnostic=opt.agnostic_nms,
        )
        t2 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path, "", im0s, getattr(dataset, "frame", 0)

            s += "%gx%g " % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                # Process the detected objects and draw bounding boxes and labels on the image
                for *xyxy, conf, cls in reversed(det):  # Loop through detections
                    # Draw bounding box
                    plot_only_box(xyxy, im0, color=(255, 255, 255), line_thickness=2)

            # Print time (inference + NMS)
            print(f'{s} inference: ({tp - t1:.3f}s) nms: ({t2 - tp:.3f}s) total: ({t2 - t1:.3f}s)')
            # print('%.3f fps' % (1.0/(t2 - t1)))

            if torch.is_tensor(n):
                prediction = n.item()
            else:
                prediction = n
            cv2.putText(im0, 'Head count=' + str(prediction), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Stream results
            cv2.imshow(str(p), im0)
            key = cv2.waitKey(1)  # 1 millisecond
            if key == ord('q'):
                break

    print(f"Done. ({time.time() - t0:.3f}s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--weights", nargs="+", type=str, default="weights/yolo-crowd.pt", help="model.pt path(s)"
    )
    parser.add_argument(
            "--source", type=str, default="/home/nicola/Software/YOLO-CROWD/data/MOT16-03.mp4", help="source"
    )  # file/folder, 0 for webcam
    parser.add_argument('--img-size', nargs='+', type=int, default=[480, 640], help='[train, test] image sizes')
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
    parser.add_argument("--augment", action="store_true", help="augmented inference")

    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=("pycocotools", "thop"))

    with torch.no_grad():
        main(opt)
