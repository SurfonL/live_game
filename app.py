import asyncio
import logging
import queue
import threading
import urllib.request
from pathlib import Path
from typing import List, NamedTuple, Optional

import av
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pydub
import streamlit as st
# from models.yolov3 import load_model
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from aiortc.contrib.media import MediaPlayer
from streamlit_webrtc import (RTCConfiguration, WebRtcMode,
                              WebRtcStreamerContext, webrtc_streamer)
from torchvision.models.detection import (FasterRCNN_ResNet50_FPN_V2_Weights,
                                          fasterrcnn_resnet50_fpn_v2)

from torchvision.utils import draw_bounding_boxes
from pathlib import Path

from game.monsters import Monster
from PIL import Image
import datetime
import pickle
import os
import glob
from streamlit_autorefresh import st_autorefresh



# from session import SessionState

HERE = Path(__file__).parent

logger = logging.getLogger(__name__)

now = datetime.datetime.now()

st.set_page_config(initial_sidebar_state="collapsed")
def main():
    count = st_autorefresh(interval=2000, limit=10000, key="fizzbuzzcounter")
    st.header("gooutandmakesomefriendsplease")

    pages = {
        "Game Page": app_object_detection
    }
    
        # "Real time video transform with simple OpenCV filters (sendrecv)": app_video_filters,  # noqa: E501
        # "Real time audio filter (sendrecv)": app_audio_filter,
        # "Delayed echo (sendrecv)": app_delayed_echo,
        # "Consuming media files on server-side and streaming it to browser (recvonly)": app_streaming,  # noqa: E501
        # "WebRTC is sendonly and images are shown via st.image() (sendonly)": app_sendonly_video,  # noqa: E501
        # "WebRTC is sendonly and audio frames are visualized with matplotlib (sendonly)": app_sendonly_audio,  # noqa: E501
        # "Simple video and audio loopback (sendrecv)": app_loopback,
        # "Configure media constraints and HTML element styles with loopback (sendrecv)": app_media_constraints,  # noqa: E501
        # "Control the playing state programatically": app_programatically_play,
        # "Customize UI texts": app_customize_ui_texts,
    page_titles = pages.keys()
    
    page_title = st.sidebar.selectbox(
        "Choose the app mode",
        page_titles,
        label_visibility='hidden'
    )
    st.subheader(page_title)

    page_func = pages[page_title]
    page_func()

    logger.debug("=== Alive threads ===")
    for thread in threading.enumerate():
        if thread.is_alive():
            logger.debug(f"  {thread.name} ({thread.ident})")




@st.cache
def init_run(monster_name):
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    frcnn = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.75, rpn_post_nms_top_n_test=512, download_file=True).eval().cuda()
    
    cap = monster_name
    cap = cap.capitalize()
        
    mon_dir = 'monsters/{}/{}.jpg'.format(cap,monster_name)
    mask_dir = 'monsters/{}/{}_mask.jpg'.format(cap, monster_name)
    effect_dir = 'monsters/{}/{}.jpg'.format('effects', 'bang')
    effect_mask = 'monsters/{}/{}_mask.jpg'.format('effects', 'bang')
    
    monster = cv2.imread(mon_dir)
    mask = cv2.imread(mask_dir)
    effect = cv2.imread(effect_dir)
    effect_mask = cv2.imread(effect_mask)
    
    
    return  frcnn, weights, [monster, mask, effect, effect_mask]

global save_, retake_, hor
atk = threading.Event()
retake_ = threading.Event()
# saved_ = threading.Event()




def app_object_detection():
    
    
    # Session-specific caching   
    cache_key = "object_detection_dnn"

    transforms = T.Compose([
                            T.ToTensor(),
                            # T.Resize((512,512)),
                            ])    

    col1, col2, col3 = st.columns(3)
    monster_name = col1.text_input("name of the monster", "rabbit")
    original_label = col2.text_input("name of the object", "car")
    disp = col3.radio("disp", ["monster", "box"], index=0)
    
    t1= st.columns(1)
    t1 = t1[0]
    # t4.button("save", on_click = _save_data)
    
    
    
    # exp_dir = "photos/{}.{}.{}.{}-{}_{}_{}".format(now.month,now.day,now.hour,now.minute, exp_name, original_label, model_name)
    # exp_dir = "photos/{}_{}_{}/".format(exp_name, original_label, model_name)
    # exp_dir = exp_dir.lower()
    # Path(exp_dir).mkdir(parents=True, exist_ok=True)
    
    frcnn, weights, imgs = init_run(monster_name)
    net = frcnn
    # lock = threading.Lock()
    # lock.acquire()
    
    monster, mask = imgs[0], imgs[1]
    global Mon
    Mon = Monster(monster=monster,mask=mask)
    
    global Eff
    eff, eff_mask = imgs[2], imgs[3]
    Eff = Monster(eff, eff_mask)
    # lock.release()
    ra = 0.5
    font = 'Ubuntu-R.ttf'
    
    original_label = original_label.lower()
   
    
    # img_container = {"img": None, 'box':None, 'detection':None}
    def callback(frame: av.VideoFrame) -> av.VideoFrame:
        # lock = threading.Lock()
        global monster, mask, atk
        

        image = frame.to_ndarray(format="bgr24")
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = transforms(image).cuda()[None,...]

        with torch.no_grad():
            detections = net(image)

        
        image = image*255
        image = image[0].detach().cpu().to(torch.uint8)

        labels = [str(weights.meta["categories"][i.item()]) for i in detections[0]['labels']]

        
        det = {'labels':[], 'boxes':[], 'scores':[]}
        if not original_label == "any":
            try:
                obj_idx = labels.index(original_label)
                l = ['{} {}'.format(labels[obj_idx], int(round(detections[0]["scores"][obj_idx].item(),2)*100))]
                b = detections[0]['boxes'][obj_idx][None,...]
                det['labels']=[detections[0]['labels'][obj_idx].item()]
                det['boxes']=b
                det['scores'] = [detections[0]['scores'][obj_idx].item()]
            except ValueError:
                l = torch.tensor([])
                b = torch.tensor([])
        else:
            l = labels
            b = detections[0]['boxes']
            
        c,h,w = image.shape
        if disp == 'box':
            color = (255,0,0) if original_label in labels else (0,255,0)
            font_size = int(w/20)
            box = draw_bounding_boxes(image, b ,l, colors = color,font=font, font_size=font_size, width=4).permute(1,2,0).detach().cpu().numpy()
            
            disp_image = box
        elif disp == 'monster':
            disp_image = image.permute(1,2,0).to(torch.uint8).detach().cpu().numpy()
            if len(b) != 0:
                h_r, w_r = int(h/5), int(b[0,3]-b[0,1])
                # lock.acquire()
                
                
                posy = int((b[0,1]+b[0,3])/2)
                posx = int((b[0,0]+b[0,2])/2)
                
                image_monster = Mon.draw(disp_image, posy, posx, h_r,w_r)
                # lock.release()
                disp_image = image_monster if isinstance(image_monster, (np.ndarray, np.generic) ) else disp_image
                
                
                
                if atk.isSet():
                    atk.clear()
                    h_r, w_r = int(h/5), int(b[0,3]-b[0,1])
                    image_eff = Eff.draw(disp_image,posy, posx, h_r,w_r)
                    disp_image = image_eff if isinstance(image_eff, (np.ndarray, np.generic) ) else disp_image
        
        disp_image = disp_image.astype(np.uint8)       
        disp_image = cv2.cvtColor(disp_image,cv2.COLOR_RGB2BGR)
        return av.VideoFrame.from_ndarray(disp_image, format="bgr24")

    webrtc_ctx = webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        # rtc_configuration=RTC_CONFIGURATION,
        video_frame_callback=callback,
        media_stream_constraints={"video": {
            "width": 720, "height": 720, "framerate": {"max":2}}, "audio": False,}, async_processing=True)
    col1, col2, col3, col4 = st.columns(4)
    col4.button("ATTACK", on_click=reduce_health)
    col1.button("rst", on_click=reset_health)
    
    
    hor = pickle.load(open('metadata/hor.pkl', 'rb'))
    t1.slider("HP", min_value=0, max_value= 100, value=hor)


   

    
    # d = "{}-{}-{}/".format(vertical,distance,light)
    # Path(exp_dir+d).mkdir(parents=True, exist_ok=True)

   
    try: 
        if txt != 'none':
            txt = txt.lower()

    except NameError:
        txt = 'none'

    
    # with lock:
    #     img = img_container["img"]
    #     box = img_container["box"]
    #     det = img_container["detection"]
        
    # if save_ and img != None:
    #     print('save', save_)
    #     save_data(box,img,det)
        
    
def reduce_health():
    global hor, atk
    hor = pickle.load(open('metadata/hor.pkl', 'rb'))
    hor = hor - 10
    pickle.dump(hor, open('metadata/hor.pkl', 'wb'))
    atk.set()
        
def reset_health():
    global hor
    hor = 100
    pickle.dump(hor, open('metadata/hor.pkl', 'wb'))
    pr = pickle.load(open('metadata/hor.pkl', 'rb'))



if __name__ == "__main__":
    import os

    DEBUG = os.environ.get("DEBUG", "false").lower() not in ["false", "no", "0"]

    logging.basicConfig(
        format="[%(asctime)s] %(levelname)7s from %(name)s in %(pathname)s:%(lineno)d: "
        "%(message)s",
        force=True,
    )

    logger.setLevel(level=logging.DEBUG if DEBUG else logging.INFO)

    st_webrtc_logger = logging.getLogger("streamlit_webrtc")
    st_webrtc_logger.setLevel(logging.DEBUG)

    fsevents_logger = logging.getLogger("fsevents")
    fsevents_logger.setLevel(logging.WARNING)

    main()