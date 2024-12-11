#coding=utf-8
import asyncio
from argparse import ArgumentParser
import json
from aiortc import MediaStreamTrack,RTCPeerConnection, RTCSessionDescription
from aiortc.rtcconfiguration import RTCIceServer,RTCConfiguration
from aiortc.contrib.media import MediaPlayer, MediaRelay, MediaBlackhole
from aiortc.rtcrtpsender import RTCRtpSender
from aiortc.sdp import candidate_from_sdp
from av import VideoFrame
import websockets
import cv2
import torch
import numpy as np
from ultralytics import YOLO

# Initialize the YOLOv8 model
model = YOLO('/home/hsnl/Desktop/best.pt')  # Update path to your best.pt

# 類別名稱
class_names = {
    0: u'4',
    1: u'5',
    2: u'7',
    3: u'7.5',
    4: u'8',
    5: u'8.5',
}

uploader=None
downloader=None
webRtcConfig = RTCConfiguration(iceServers=[ 
    RTCIceServer(urls="stun:ice4.hsnl.tw:3478",username="hsnl2",credential="hsnl33564"),
    RTCIceServer(urls="turn:ice4.hsnl.tw:3478",username="hsnl2",credential="hsnl33564"),
])

class VideoTransformTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, track):
        super().__init__()  # don't forget this!
        self.track = track

    def stop(self):
        super().stop()
        self.track.stop()

    async def recv(self):
        frame = await self.track.recv()

        try:
            # 將 WebRTC 每幀轉換為 numpy 數組
            img = frame.to_ndarray(format="bgr24")

            # change picture size
            input_height = 640
            input_width = 640
            resized_frame = cv2.resize(img, (input_width, input_height))

            # 轉換為 PyTorch 張量
            image_np = resized_frame.transpose(2, 0, 1)  # HWC 轉 CHW
            image_np = np.expand_dims(image_np, axis=0)  # 添加批次维度
            image_tensor = torch.from_numpy(image_np).float() / 255.0  # 歸一化到 [0, 1]

            # 推理
            results = model(image_tensor)

            # 處理結果
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()  # 預測框座標
                    conf = box.conf[0]  # 置信度
                    cls = box.cls[0]  # 類別索引
                    class_name = class_names.get(int(cls), 'Unknown')  # 獲取類別名稱
                    print(f'Class: {class_name}, Confidence: {conf}, Box: {x1, y1, x2, y2}')

                    # 檢測結果
                    scale_x = img.shape[1] / input_width
                    scale_y = img.shape[0] / input_height
                    x1 = int(x1 * scale_x)
                    y1 = int(y1 * scale_y)
                    x2 = int(x2 * scale_x)
                    y2 = int(y2 * scale_y)

                    # 繪製矩形框
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # class label
                    label = f'{class_name}, Conf: {conf:.2f}'
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    label_x = x1
                    label_y = y1 - 10 if y1 - 10 > 10 else y1 + 10
                    cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # change back to WebRTC
            new_frame = VideoFrame.from_ndarray(img, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            return new_frame

        except Exception as e:
            print("error in VideoTransformTrack", e)
            return frame
class User():
    def __init__(self, uid,pc=None):
        self.id=uid #session id
        self.pc=pc #peerConnection object
        self.isClose=False


class Uploader(User):
    def __init__(self,uid,pc):
        super().__init__(uid,pc)
        self.videoTrack=None
        self.originVideoTrack=None
        self.audioTrack=None
        self.blackHole=MediaBlackhole()
        self.relay=MediaRelay()
    def addVideoTrack(self,track):
        self.originVideoTrack=track
        # self.videoTrack=VideoTransformTrack(track)
        self.videoTrack = track
    def addAudioTrack(self,track):
        self.audioTrack=track

class Downloader(User):
    def __init__(self,uid,pc=None):
        super().__init__(uid,pc)

async def closeUploader(ws):
    global uploader,downloader
    if uploader is None:
        return
    _uploader = uploader
    uploader = None
    await _uploader.blackHole.stop()
    await _uploader.pc.close()
    if downloader is None:
            await ws.close()
async def closeDownloader(ws):
    global uploader,downloader
    if downloader is None:
        return
    _downloader = downloader
    downloader = None
    await _downloader.pc.close()
    if uploader is None:
        await ws.close()
async def registerProcess(ws,uploaderId):
    await ws.send(json.dumps({"operation":"registerProcess","uploaderId":uploaderId}))

async def createDownloader(ws,message):
    global downloader,uploader
   
    downloaderId = message['downloaderId']
    
    pc = RTCPeerConnection(webRtcConfig)
    
    downloader = Downloader(downloaderId, pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print("[connectionstatechange] Connection uid %s state is %s" % (downloaderId,pc.connectionState))
        if pc.connectionState == "failed" or pc.connectionState=="closed":
            await closeDownloader(ws)

    
    if uploader is not None : 
        await uploader.blackHole.stop()
        if uploader.videoTrack:
            pc.addTrack(VideoTransformTrack(uploader.relay.subscribe(uploader.videoTrack,False)))

    # channel = pc.createDataChannel("chat")
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)

    await ws.send(json.dumps({"operation":"initDownloaderFromProcess", "sdp": pc.localDescription.sdp, 
                "type": pc.localDescription.type,"downloaderId":downloaderId}))

async def recvDownloaderAns(ws,message):
    global downloader
    answer = RTCSessionDescription(sdp=message["sdp"], type=message["type"]) 
    await downloader.pc.setRemoteDescription(answer)

async def createUploader(ws,message):
    global uploader,downloaders
    uploaderId = message['uploaderId']



    offer = RTCSessionDescription(sdp=message["sdp"], type=message["type"])

    pc = RTCPeerConnection(webRtcConfig)

    _uploader = Uploader(uploaderId,pc)
    uploader= _uploader
  
 
    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print("[connectionstatechange] uploader id:%s,  connection state is %s" % (uploaderId,pc.connectionState))
        if pc.connectionState == "failed" or pc.connectionState == "closed":
            print("[close peer connection] id:%s"%uploaderId)
            await closeUploader(ws)

    @pc.on("track")
    async def onTrack(_track):
        if _track.kind == "audio":
            print('recv audio track!!!!!')
            _uploader.addAudioTrack(_track)
            return
        _uploader.addVideoTrack(_track)
        
        _uploader.blackHole.addTrack(uploader.videoTrack)#重要,一定要記得如果沒有其他cosumer則需要把track丟進blackHole,否則frame會一直累積在記憶體中
        await _uploader.blackHole.start()
    
    await pc.setRemoteDescription(offer)
    

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    await ws.send(json.dumps({"operation":"registerUploaderResponse","sdp": pc.localDescription.sdp, 
            "type": pc.localDescription.type, 'uploaderId':uploaderId}))
    
async def websocketClient(uploaderId):
    async with websockets.connect("ws://localhost:8765") as websocket:
        await registerProcess(websocket,uploaderId)
        async for message in websocket:
            msg=json.loads(message)
            operation = msg["operation"]
            print("recv operation :%s"%operation)
            if operation == "registerUploaderWithProcess":
                asyncio.create_task(createUploader(websocket,msg))
            elif operation =="createDownloader":
                asyncio.create_task(createDownloader(websocket,msg))
            elif operation == "downloaderAns":
                asyncio.create_task(recvDownloaderAns(websocket,msg))
            

if __name__ == "__main__":
    print("----------start stream process----------")
    parser = ArgumentParser()

    parser.add_argument("-uploaderId", dest="uploaderId")

    
    args = parser.parse_args()

    asyncio.run(websocketClient(args.uploaderId))

    #clear 
    print("----------finish stream process----------")
