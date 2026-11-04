# Real-Time Banana Ripeness Detection on Smart Glasses

Undergraduate capstone project (CS, National Tsing Hua University). Live video from BT-300 smart glasses is streamed over WebRTC to an NVIDIA Jetson Xavier, where a YOLOv8 model detects banana ripeness; annotated frames are sent back and shown on the glasses in real time. The goal is to help banana farmers decide when to harvest.

## My contribution
- **Dataset:** recorded videos at a banana farm in cooperation with a local agricultural company, labeled them with Roboflow, and applied augmentation (1,200 images, 6 ripeness classes).
- **Model:** trained the YOLOv8 detector (`pythonServer/best.pt`).
- **Detection logic:** wrote the per-frame inference and the bounding-box/label drawing in `VideoTransformTrack.recv` (`pythonServer/streamProcess.py`).
- **Deployment:** set up the Jetson Xavier environment and ran the full system with the BT-300 glasses.
- **Report:** wrote the capstone report and poster (not included in this repo).

## Credits
The WebRTC streaming framework (`wsServer/`, `frontend/`, and the connection/signaling logic in `pythonServer/`) was written by a graduated senior student and inherited for this project. It carries video from the glasses to the Jetson and back.

## Results
- Validation set (240 images): precision 0.985, recall 0.988, mAP@50 0.992.
- Inference time: ~40 ms/frame on the Jetson Xavier GPU vs. ~800 ms on a laptop CPU.
- Limitation: the system was not validated in a field test at the farm.

## Resources
- Training script: `other_resources/yolov8_model/train_banana.py` (YOLOv8n, 30 epochs, 640×640; set the `DATA_YAML` environment variable to your dataset's `data.yaml`).
- Dataset: Roboflow link in `other_resources/dataset/dataset_URL`.
- Sample detection videos: `other_resources/video/`.

## Setup
- The trained weights `pythonServer/best.pt` are included, so retraining is optional.
- Install dependencies: `pip install -r pythonServer/requirements.txt` (on Jetson, use the PyTorch build provided by NVIDIA).
- Optional environment variables: `MODEL_PATH`, `STUN_URL`, `TURN_URL`, `TURN_USERNAME`, `TURN_CREDENTIAL`. Without STUN/TURN settings, aiortc falls back to its default public STUN server.

## Known issues
- In the inherited signaling server (`wsServer/index.js`), the duplicate-downloader check in `registerDownloader` does not work as intended, and one log line in `downloaderAns` references an undefined variable.

---

# 使用說明（中文）

## 本 repo 功能
可以讓兩個用戶 a（瀏覽器）將影片使用 WebRTC 傳送到 pythonServer 後，由 pythonServer 對影片的 frame 作處理，再透過 WebRTC 送給用戶 b（瀏覽器）。配合已訓練好的香蕉成熟度辨識模型，可以辨識畫面中的香蕉成熟度。

## 環境安裝
1. 進入 `pythonServer` 資料夾後（可直接按照 `requirements.txt` 安裝，或如下）
    - `pip install websockets`
    - `pip install aiortc`
    - `pip install av`
    - `pip install opencv-python`
2. 進入 `wsServer` 資料夾後
    - `npm install`

## 使用方法
1. 進入 `wsServer` 資料夾後使用 `node index.js` 開啟 websocket server（預設在 8765 port）。
2. 進入 `pythonServer` 資料夾，使用以下指令開啟 server，`serverId` 請填入想要設定的 serverId：
    ```
    python server.py -serverId <serverId>
    ```
3. 進入 `frontend` 資料夾，輸入 `node server.js` 開啟前端伺服器，預設在本地的 3000 port，這樣就可以從 3000 port 存取網頁，或是直接開啟 `frontend` 資料夾中的 `index.html`（請開啟兩個，一個作為上傳影片方，一個作為下載影片方）。
    1. 上傳方請填入 "my user id" 欄位跟 "to python server id" 欄位。"my user id" 是這個 uploader 的 id，可以隨便設；"to python server id" 請填剛剛開啟的 pythonServer 的 id。
    2. 等待 uploader 與 pythonServer 連線（成功連線時螢幕上會顯示 uploader 已成功連線），請等 uploader 成功連線後再進行第 3 步（連線 downloader）。
    3. 下載方請填入 "my user id" 跟 "connectToUserId"。"my user id" 代表這個 downloader 的 id，可以隨便設；"connectToUserId" 請填入想要連線到的 uploader 的 id。
4. 此時 downloader 頁面應該就能看到 uploader 方拍攝到的畫面，並且是經過影像處理的（可自行修改 `./pythonServer/streamProcess.py` 中 `VideoTransformTrack` 這個 class 內部的 `recv` method，將其回傳的 frame 修改為想要做的處理）。

**[注意] 請確認所有東西的 id 都不能重複！**

## 其他注意事項
* 首次使用網頁若遇到不安全而無法使用攝影機，要去 chrome flags 設定此網站為安全。
* `other_resources` 內的訓練腳本、資料集連結與範例影片，執行系統時不會用到，只提供想重現訓練或看範例結果的人。
* 在 uploader 的頁面可以呼叫 `setWH(w,h)`，傳入想要設定的上傳影片解析度。
* 在 uploader 的頁面可以呼叫 `setMaxCapability`，把上傳者的影片解析度跟 fps 調整到其攝影機的上限。
* `setWH` 跟 `setMaxCapability` 需要在 uploader 方呼叫才有用。WebRTC 協議會根據當前網路環境與接收/傳送方的資源，自動調整畫質（影片壓縮率）、解析度（WH）跟 fps，且這個自動調整功能無法關閉。因此如果發現畫質、fps 或寬高被自動調整，只能在 uploader 端呼叫 `setWH` 或 `setMaxCapability`，請求 WebRTC library 把畫質跟 fps 調成我們想要的。
