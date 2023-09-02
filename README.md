# myyolov7_study，我的yolov7學習紀錄
第一次接觸影像辨識，從一開始的cuda、cudnn安裝，到後面的opencv、pytorch、anaconda虛擬環境建置...，種種設置讓人關關難過關關過，只能從網路上文章和影片，去摸索設置，歷經多次挫敗終於成功!

本文是分享自己建置yolov7環境的做法與成功心得，參考過網路上許多文章，但或許一個配置不一樣，就有可能發生不同狀況，如果與我是相同配備，那或許這篇文章對您幫助很大，如果不同也可以參考看看。

以下是本身電腦配置:  
HP Pavilion Gaming Laptop 17 (筆電)  
處理器	Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz   2.59 GHz  
顯示卡  NVIDIA GeForce GTX 1650Ti 4GB  
已安裝記憶體(RAM)	16.0 GB (15.8 GB 可用)  
Windows 10 家用版

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
**& 環境建置 &**

下載yolov7:  
在D槽新建名為Yolov7資料夾(將之後相關資料存放至此，方便統整)  
在此資料夾執行cmd，git clone https://github.com/WongKinYiu/yolov7  

下載anaconda:  
一路next到安裝完成  
手動加入anaconda環境變數:設定-->關於-->進階系統設定-->環境變數-->系統變數往下拉找到path點進去-->我的變數增加4個:  
C:\Users\eric\anaconda3  
C:\Users\eric\anaconda3\Library\bin   
C:\Users\eric\anaconda3\Library\mingw-w64\bin  
C:\Users\eric\anaconda3\Scripts  

打開anaconda prompt(win鍵後在anaconda3資料夾打開裡面)，開始以下操作:    
conda install git  
conda install pip  
conda create --name yolov7 python=3.9 -建立python3.9環境  
activate yolov7 -啟用環境  
cd /d D:\Yolov7\yolov7 -切換路徑   #因為anaconda裝在c槽，所以+/d  
Pip install -r requirements.txt -下載附加套件  

下載權重:  
yolov7資料夾底下新增一個新的資料夾weights,進到weights裡執行cmd  
git clone https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt  

下載cuda:  
去NVIDIA 官方網站下載對應顯卡的CUDA Tool   #本機使用CUDA Toolkit 10.2  
官網網址 --> https://developer.nvidia.com/cuda-toolkit-archive  
下載 > 安裝Next到底 > C:\Program Files\NVIDIA GPU Computing Toolkit ( 安裝成功會看到這個資料夾 )
 
下載cuDNN:  
依據自己的cuda版本去官網>登陸(申請帳號)>下載對應的版本(10.2)  
下載下來後,解壓縮,你可以看到一個資料夾,將資料夾裡面的三個資料夾 (bin,include,lib)  
覆蓋到 C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2 裡面去  
可以使用 nvidia-smi 指令來查看當前驅動及CUDA版本  

下載pytorch:  
直接附上官方網址 --> https://pytorch.org/get-started/locally/  
下圖皆為預設即可，並找到 Run this Command: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117  
要更改指令最後/cu117，改為/cu102，將指令貼上anaconda prompt(記得切換好環境)  

![image](https://github.com/ericxiang666/myyolov7_study/assets/89746072/ab3410b1-6aeb-4f4c-96ec-ea8dbe764be5)  



最後直接執行 python detect.py  
第一次執行會下載預訓練好的模型，下載好會預測放在 .\inference\images 的圖片，可以在 .\runs\detect\exp 看到預測結果  
也可以用 python detect.py --source 指定要預測的圖片或影片路徑  

本機cmd遇到的2個問題:  
_此error:AttributeError: module ‘distutils‘ has no attribute ‘version‘_  
(1)numpy要改版本->pip uninstall numpy->pip install numpy==1.23  
(2)pip uninstall setuptools->pip install setuptools==59.5.0  

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
