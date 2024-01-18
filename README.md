# myyolov7_study，我的yolov7學習紀錄
第一次接觸影像辨識，從一開始的cuda、cudnn安裝，到後面的opencv、pytorch、anaconda虛擬環境建置...，種種設置讓人關關難過關關過，只能從網路上文章和影片，去摸索設置，歷經多次挫敗終於成功!

本文是分享自己建置yolov7環境的做法與成功心得，參考過網路上許多文章，但或許一個配置不一樣，就有可能發生不同狀況，如果與我是相同配備，那或許這篇文章對您幫助很大，如果不同也可以參考看看。

以下是本身電腦配置 :  
HP Pavilion Gaming Laptop 17 (筆電)  
處理器	Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz   2.59 GHz  
顯示卡  NVIDIA GeForce GTX 1650Ti 4GB  
已安裝記憶體(RAM)	16.0 GB (15.8 GB 可用)  
Windows 10 家用版

---
**& 環境建置 &**

下載yolov7 :  
在D槽新建名為Yolov7資料夾(將之後相關資料存放至此，方便統整)  
在此資料夾執行cmd，git clone https://github.com/WongKinYiu/yolov7  

![image](https://github.com/ericxiang666/myyolov7_study/assets/89746072/f71f54d2-7e44-4d54-b0ab-1376454c10ea)  

![image](https://github.com/ericxiang666/myyolov7_study/assets/89746072/407c9deb-e725-4586-890d-70ab44ab4522)  

下載anaconda :  
一路next到安裝完成  
手動加入anaconda環境變數:設定-->關於-->進階系統設定-->環境變數-->系統變數往下拉找到path點進去-->我的變數增加4個:  
C:\Users\eric\anaconda3  
C:\Users\eric\anaconda3\Library\bin   
C:\Users\eric\anaconda3\Library\mingw-w64\bin  
C:\Users\eric\anaconda3\Scripts  

![image](https://github.com/ericxiang666/myyolov7_study/assets/89746072/753f53cb-c692-442a-abfa-af9c05e34502)  

打開anaconda prompt(win鍵後在anaconda3資料夾打開裡面)，開始以下操作 :  

![image](https://github.com/ericxiang666/myyolov7_study/assets/89746072/19af6195-6495-45b7-9287-64f77065d566)  

conda install git  
conda install pip  
conda create --name yolov7 python=3.9 -建立python3.9環境  
activate yolov7 -啟用環境  
cd /d D:\Yolov7\yolov7 -切換路徑   #因為anaconda裝在c槽，所以+/d  
Pip install -r requirements.txt -下載附加套件  

下載權重 :  
yolov7資料夾底下新增一個新的資料夾weights,進到weights裡執行cmd  
git clone https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt  

下載cuda :  
去NVIDIA 官方網站下載對應顯卡的CUDA Tool   #本機使用CUDA Toolkit 10.2 

![image](https://github.com/ericxiang666/myyolov7_study/assets/89746072/2d6e160e-19ee-4cbf-bc8e-c5ffeadf93ce)  

官網網址 --> https://developer.nvidia.com/cuda-toolkit-archive  
下載 > 安裝Next到底 > C:\Program Files\NVIDIA GPU Computing Toolkit ( 安裝成功會看到這個資料夾 )  

![image](https://github.com/ericxiang666/myyolov7_study/assets/89746072/51f3474f-e25d-4702-bb71-28d9ff1fbb23)  
 
下載cuDNN :  
依據自己的cuda版本去官網>登陸(申請帳號)>下載對應的版本(10.2)  
下載下來後,解壓縮,你可以看到一個資料夾,將資料夾裡面的三個資料夾 (bin,include,lib)  

![image](https://github.com/ericxiang666/myyolov7_study/assets/89746072/bbbce0e8-0145-489b-9881-26954d396c57)  

覆蓋到 C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2 裡面去  
可以使用 nvidia-smi 指令來查看當前驅動及CUDA版本  

下載pytorch :  
直接附上官方網址 --> https://pytorch.org/get-started/locally/  
下圖皆為預設即可，並找到 pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117  
要更改指令最後/cu117，改為/cu102，將指令複製貼上anaconda prompt(記得切換好環境)  

![image](https://github.com/ericxiang666/myyolov7_study/assets/89746072/dfa7e7e3-e9f2-4978-8b4e-45633de609b3)  




最後直接執行 python detect.py  
第一次執行會下載預訓練好的模型，下載好會預測放在 .\inference\images 的圖片，可以在 .\runs\detect\exp 看到預測結果  
也可以用 python detect.py --source 指定要預測的圖片或影片路徑  

本機執行遇到的2個問題 :  
_此error:AttributeError: module ‘distutils‘ has no attribute ‘version‘_  
(1)numpy要改版本->pip uninstall numpy->pip install numpy==1.23  
(2)pip uninstall setuptools->pip install setuptools==59.5.0  

---
**& 訓練 &**  

建立存放所有訓練檔案的資料夾(好整理) :  
創建圖片資料夾(命名all)-->放入訓練圖片(jpg)  

標記訓練圖片 :  
開啟cmd輸入labelimg，開啟標記程式  

![image](https://github.com/ericxiang666/myyolov7_study/assets/89746072/7de2299c-57b8-48ac-878b-c9d6eecc2479)  

(1)左邊第二點選open dir (開啟圖片存放資料夾，開啟前面提到的all資料夾)  
(2)左邊第八點選標記格式，這邊選YOLO格式(txt)  
(3)左邊第九點選creat rectbox (滑鼠左鍵拉方塊，框選圖片標記位置，命名標記名稱)。這裡筆者建議使用快捷鍵操作(W)  
(4)左邊第七點選Save，儲存標記好圖片(CTRL+S)  
(5)左邊第五點選Next Image ，下一張圖片繼續標記(D)  
(6)反覆上述3~5步驟操作，標到結束。  
查看all裡面，呈現的資料存放方式會是--> jpg、txt、jpg、txt、......依序排序下去，裡面會有一個classes.txt，是存放標記物的名稱  

![image](https://github.com/ericxiang666/myyolov7_study/assets/89746072/9af2d232-903d-42e1-9489-7f5ba0fb0592)  

回到訓練資料夾  
使用splitFile.py (將all裡面的jpg和txt分開來，會得到兩個資料夾，images和labels)  
使用creat_txt.py (建立train和val文字檔，作為data讀取位置，比較不會出錯)  

準備訓練yaml檔 :  
(1)./cfg/training/yolov7.yaml 複製到訓練用資料夾，更改檔案名稱-->yolov7-mask.yaml (這裡以配戴口罩為例，取名隨興)  
(2)更改 yolov7-mask.yaml 裡面程式碼，只需找到nc=80更改為nc=2 (此句是標記物名稱有幾個)  
(3)準備第二個yaml檔，./data/coco.yaml 複製到訓練用資料夾，更改檔案名稱，個人習慣取data,yaml  
(4)更改 data.yaml 程式碼(以下述程式碼講解):  

	train: D:\yolov7\yolov7\mydataset\train.txt	#讀取訓練圖片位置  
	val: D:\yolov7\yolov7\mydataset\val.txt		#讀取驗證圖片位置  

	nc: 2		#有兩個標記物  
	names: ['mask', 'no-mask']	#標記物名稱  

總結:標記好的圖片、兩個yaml檔、存放路徑文字檔  

_訓練指令_  
python train.py --workers 4 --device 0 --batch-size 8 --data mydataset/data.yaml --img 640 640 --cfg mydataset/yolov7-mask.yaml --weights yolov7.pt --hyp data/hyp.scratch.p5.yaml --epoch 1000  

_指令補充_  
batch指的就是一批要多少訓練樣本  
subdivisions是用來細分一次放多少樣本到記憶體中，例:batch=64，subdivisions=16-->可以一次放入 4 張圖片 （64/16）  
參考網址 : https://ithelp.ithome.com.tw/articles/10267031  

---
**&Google Colab&**

本段要介紹的是Google Colab，免費版本T4 GPU使用時間8小時，有需要更高算力則要升級付費，接下來開始講解使用方式。  

Google 搜尋 Google Colab :  
1.  
![image](https://github.com/ericxiang666/yolov7-lednet_study/assets/89746072/47178258-2194-49d6-bdaf-6fc00ae85279)  
2.  
![image](https://github.com/ericxiang666/yolov7-lednet_study/assets/89746072/ce3914e7-00e8-4189-822e-ef4b77ca7a19)  

新增筆記本 :  
1.  
![image](https://github.com/ericxiang666/yolov7-lednet_study/assets/89746072/6fd423b7-3f1c-49d4-9518-680d26488d39)  
2.  
左上重新命名  
![image](https://github.com/ericxiang666/yolov7-lednet_study/assets/89746072/4864d4bb-31d9-4fd4-b32b-a2f2d560dc8f)  
****
選擇T4 GPU並連線 :  
1.  
![image](https://github.com/ericxiang666/yolov7-lednet_study/assets/89746072/39888d9a-93ad-4f50-ab5b-58ae35b96951)  
2.  
![image](https://github.com/ericxiang666/yolov7-lednet_study/assets/89746072/ff764fd0-1f0e-4103-a366-0f5a8349e4cb)  

掛接雲端硬碟 :  
