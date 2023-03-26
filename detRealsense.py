import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import scipy.io as sio
import sys
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

import pyrealsense2 as rs
import numpy as np
import math
from operator import length_hint
def find_plane(points):

    c = np.mean(points, axis=0)
    r0 = points - c
    u, s, v = np.linalg.svd(r0)
    nv = v[-1, :]
    ds = np.dot(points, nv)
    param = np.r_[nv, -np.mean(ds)]
    return param

def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    bandera=0
    flag=0
    zhong=0
    guo=0
    band=0
    hashp=0
    alpha_mat=[]
    gamma_mat=[]
    Xtarget_mat=[]
    Ytarget_mat=[]
    Ztarget_mat=[]
    labels_mat=[]
    num_objects_mat=[]
    object_coordinates_detail=[]
    time_mat=[]
    fl=0
    depth_generalistic=[np.arange(480*640).reshape(480,640)]#Original
    time_per_frame_arr=[]
    object_coordinates = []
    
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

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    pipeline = rs.pipeline()
    profile = pipeline.start(config)

    align_to = rs.stream.color
    align = rs.align(align_to)

    
    colorizer=rs.colorizer()
    colorizer.set_option(rs.option.visual_preset, 0)
    colorizer.set_option(rs.option.min_distance, 0)
    colorizer.set_option(rs.option.max_distance, 15)
    
    intr=profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics() # Get intrinsic parameters to calculate perspective projection calculation
    
    while(True):

        time_3=time.time()
        time_0=time.time()
        print('Time 0:')
        print(time_0)
        frames = pipeline.wait_for_frames()

        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not depth_frame or not color_frame:
            continue

        img = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.08), cv2.COLORMAP_JET)
       
        # Letterbox
        im0 = img.copy()
        img = img[np.newaxis, :, :, :]        

        # Stack
        img = np.stack(img, 0)

        # Convert
        img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        img = np.ascontiguousarray(img)


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

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            time_1=time.time()
            
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                print(len(det))
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = f'{names[c]} {conf:.2f}'
                    xyxy=[int(e_.item()) for e_ in xyxy] #Converting from tensrflow to numeric list representation
                    #details was here before
                    
                    
                    x=int((xyxy[0]+xyxy[2])/2)
                    y = int((xyxy[1] + xyxy[3])/2)
                    dist = depth_frame.get_distance(x + 4, y + 8)*1000
                    Xtarget = dist*(x+4 - intr.ppx)/intr.fx - 35 #the distance from RGB camera to realsense center
                    Ytarget = dist*(y+8 - intr.ppy)/intr.fy
                    Ztarget = dist

                    details=f'{names[c]} {conf:.2f} {xyxy}'
                    if bandera==0:
                        Xtarget_mat=np.array(Xtarget)
                        Ytarget_mat=np.array(Ytarget)
                        Ztarget_mat=np.array(Ztarget)
                        labels_mat=np.array(label)
                        num_objects_mat=len(det)
                        num_objects_mat_2=length_hint(np.array(label))
                        bandera=1
                    else:
                        Xtarget_mat=np.append(Xtarget_mat,Xtarget)
                        Ytarget_mat=np.append(Ytarget_mat,Ytarget)
                        Ztarget_mat=np.append(Ztarget_mat,Ztarget)
                        labels_mat=np.append(labels_mat,label)
                        num_objects_mat=np.append(num_objects_mat,len(det))
                        num_objects_mat_2=np.append(num_objects_mat_2,length_hint(np.array(label)))

                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)
                    plot_one_box(xyxy, depth_colormap, label=label, color=colors[int(cls)], line_thickness=2)
                    
                    #Points used to identify plane of the object
                    offset_x=int((xyxy[2]-xyxy[0])/10)
                    offset_y=int((xyxy[3]-xyxy[1])/10) 
                    interval_x=int((xyxy[2]-xyxy[0]-2*offset_x)/2) 
                    interval_y=int((xyxy[3]-xyxy[1]-2*offset_y)/2)
                    points=np.zeros([9,3])
                    for i in range(3):
                        for j in range(3):
                            x = int(xyxy[0]) + offset_x + interval_x*i
                            y = int(xyxy[1]) + offset_y + interval_y*j
                            dist = depth_frame.get_distance(x, y)*1000
                            Xtemp = dist*(x - intr.ppx)/intr.fx
                            Ytemp = dist*(y - intr.ppy)/intr.fy
                            Ztemp = dist
                            points[j+i*3][0] = Xtemp
                            points[j+i*3][1] = Ytemp
                            points[j+i*3][2] = Ztemp
                    param=find_plane(points)
                    alpha=math.atan(param[2]/param[0])*180/math.pi #Lateral rotation 
                    if(alpha < 0):
                        alpha = alpha + 90
                    else:
                        alpha = alpha - 90

                    gamma = math.atan(param[2]/param[1])*180/math.pi #Up-down rotation 
                    if(gamma < 0):
                        gamma = gamma + 90
                    else:
                        gamma = gamma - 90
                    
                    text1 = "alpha : " + str(round(alpha))
                    if zhong==0:
                        alpha_mat=alpha
                        zhong=1
                    else:
                        alpha_mat=np.append(alpha_mat, alpha)
                    text2 = "gamma : " + str(round(gamma))
                    if guo==0:
                        gamma_mat=gamma
                        guo=1
                    else:
                        gamma_mat=np.append(gamma_mat, gamma)
                    cv2.putText(im0, text1, (int((xyxy[0] + xyxy[2])/2) - 40, int((xyxy[1] + xyxy[3])/2)), cv2.FONT_HERSHEY_PLAIN, 3, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)
                    cv2.putText(im0, text2, (int((xyxy[0] + xyxy[2])/2) - 40, int((xyxy[1] + xyxy[3])/2) + 40), cv2.FONT_HERSHEY_PLAIN, 3, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)

                    print("alpha : " + str(alpha) + ", gamma : " + str(gamma))
                 
            # Print time (inference + NMS)
            #print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            cv2.imshow("Recognition result", im0)
            depth_matrix=np.asanyarray(im0)
            #print(depth_matrix)
            cv2.imshow("Recognition result depth",depth_colormap)
            #colour_mat=np.asanyarray(depth_colormap)
            #print(colour_mat)


            if flag==0:
                depth_generalistic=np.array(depth_image)
                flag=1
            else:
                depth_generalistic=np.dstack((depth_generalistic, np.array(depth_image)))
                flag=flag + 1
            #print(depth_generalistic)# This gets an instant matrix of distances in mm
            print(depth_generalistic.shape)
            time_2=time.time()
            if hashp==0:
                time_mat=time_2-time_1
                time_stamp=time_3
                hashp=1
            else:
                time_mat=np.append(time_mat,time_2-time_1)
                time_stamp=np.append(time_stamp,time_3)
            if flag%50 == 0:
                savedict = {
                  'Pixel_depth' : depth_generalistic,
                  'object_coordinates':object_coordinates,
                  'alpha':alpha_mat,
                  'gamma':gamma_mat,
                  'Xtarget_mat':Xtarget_mat,
                  'Ytarget_mat':Ytarget_mat,
                  'Ztarget_mat':Ztarget_mat,
                  'labels_mat':labels_mat,
                  'num_objects_mat':num_objects_mat,
                  'time_mat':time_mat ,
                  'time_stamp':time_stamp ,
                  'num_objects_mat_2':num_objects_mat_2
                }
                # save to disk
                sio.savemat('depth_item_data.mat', savedict)
                #load from disk
                data = sio.loadmat('depth_item_data.mat')
                #print('Saving achieved')
                #sys.exit() Use if you want to finalize execution after a sppecific number of cycles specified by the division number in flag%50
            else:
                continue


            print('Time 2-1')
            print(time_2-time_1)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7-tiny.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
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
    opt = parser.parse_args()
    print(opt)
    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
