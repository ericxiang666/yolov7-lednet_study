#yolov7
import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

# lednet
from distutils.log import error
import os
from turtle import down
import glob
from torchvision.transforms.functional import normalize
from basicsr.utils import imwrite, img2tensor, tensor2img, scandir
from basicsr.utils.download_util import load_file_from_url
import torch.nn.functional as F

from basicsr.utils.registry import ARCH_REGISTRY


def calculate_brightness(image_path):
    # 讀取圖片
    img = cv2.imread(image_path)

    # 將圖片轉換為灰階圖片
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 計算灰階圖片的平均亮度
    brightness = int(round(cv2.mean(gray)[0]))

    print(f'圖片亮度 : {brightness}')
    # print('直接辨識')

    return brightness

def process_image_based_on_brightness(image_path):
    brightness = calculate_brightness(image_path)
    
    # 根據亮度值執行對應處理
    if brightness < 30:  # 自訂亮度閾值，根據實際情況調整
        # 亮度較暗，執行操作
        print('OK') #確認上面亮度分析完畢，進到下一動
        lednet_inference()
        detect()
        # detect()

    else:
        # 亮度亮，執行另一個操作
        detect()

def check_image_size(x, down_factor):
    _, _, h, w = x.size()
    mod_pad_h = (down_factor - h % down_factor) % down_factor
    mod_pad_w = (down_factor - w % down_factor) % down_factor
    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    return x

def lednet_inference():
    # ------------------------ input & output ------------------------
    if opt.source.endswith('/'):  # solve when path ends with /
        opt.source = opt.source[:-1]
    if opt.result_path.endswith('/'):  # solve when path ends with /
        opt.result_path = opt.result_path[:-1]
    result_root = f'{opt.result_path}/{os.path.basename(opt.source)}'

    # ------------------ set up LEDNet network -------------------
    down_factor = 8 # check_image_size
    net = ARCH_REGISTRY.get('LEDNet')(channels=[32, 64, 128, 128], connection=False).to(device)
    
    # ckpt_path = 'weights/lednet.pth'
    assert opt.model in ['lednet', 'lednet_retrain', 'lednetgan'], ('model name should be [lednet] or [lednetgan]')
    ckpt_path = 'weights/lednet.pth'
    checkpoint = torch.load(ckpt_path)['params']
    net.load_state_dict(checkpoint)
    net.eval()

    # -------------------- start to processing ---------------------
    # scan all the jpg and png images
    img_paths = sorted(list(scandir(opt.source, suffix=('jpg', 'png'), recursive=True, full_path=True)))
    # img_paths = opt.source

    for img_path in img_paths:
        # img_name = os.path.basename(img_path)
        img_name = img_path.replace(opt.source+'/', '')
        print(f'Processing: {img_name}')
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        # prepare data
        img_t = img2tensor(img / 255., bgr2rgb=True, float32=True)

        # without [-1,1] normalization in lednet model (paper version) 
        if not opt.model == 'lednet':
            normalize(img_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        img_t = img_t.unsqueeze(0).to(device)

        # lednet inference
        with torch.no_grad():
            # check_image_size
            H, W = img_t.shape[2:]
            img_t = check_image_size(img_t, down_factor)
            output_t = net(img_t)
            output_t = output_t[:,:,:H,:W]

            if opt.model == 'lednet':
                output = tensor2img(output_t, rgb2bgr=True, min_max=(0, 1))
            else:
                output = tensor2img(output_t, rgb2bgr=True, min_max=(-1, 1))

        del output_t
        torch.cuda.empty_cache()

        output = output.astype('uint8')
        # save restored img
        save_restore_path = img_path.replace(opt.source, result_root)
        imwrite(output, save_restore_path)

    # print(f'\nAll results are saved in {result_root}')
    print('OK')

def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors 
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time() #12/8
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)  #line_thickness : 方框的邊界線寬度，決定粗細

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inputs/p', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.30, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--result_path', type=str, default='inputs')
    parser.add_argument('--model', type=str, default='lednet', help='options: lednet, lednet_retrain, lednetgan')
    opt = parser.parse_args()
    #check_requirements(exclude=('pycocotools', 'thop'))

    # 輸入圖片路徑
    if os.path.isdir(opt.source):
        image_paths = sorted(list(scandir(opt.source, suffix=('jpg', 'png','jpeg'), recursive=True, full_path=True)))
        for image_path in image_paths:
            # img_name = os.path.basename(img_path)
            image_name = image_path.replace(opt.source+'/', '')
            print(f'calculate_brightness: {image_name}')
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            # prepare data
            image_t = img2tensor(image / 255., bgr2rgb=True, float32=True)
        with torch.no_grad():
            if opt.update:  # update all models (to fix SourceChangeWarning)
                for opt.weights in ['yolov7.pt']:
                    # 判斷亮度並依據處理
                    process_image_based_on_brightness(image_path)
                    # detect()
                    strip_optimizer(opt.weights)
            else:
                # 判斷亮度並依據處理
                process_image_based_on_brightness(image_path)
                # detect()
    else:
        detect()