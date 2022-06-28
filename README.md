# Vietnamese_Handwriting_Recognition

An OCR model for Vietnamese Handwriting Recognition problems with CNN + LSTM implemented with PyTorch Deeplearning framework.

# Idea 
This model is based on the proposed architecture in this paper: https://arxiv.org/pdf/1507.05717.pdf
![image](https://user-images.githubusercontent.com/71833423/163555209-b24ae54c-02d6-4c95-9eb1-de583aa77db8.png)
![image](https://user-images.githubusercontent.com/71833423/163555293-807c8e6d-2af7-45fd-8b77-9e8612c8e6d7.png)
- I used pretrained VGG16 for CNN's backbone, and Bidirectional LSTM for recurrent layers


# Requirements
I highly recommend using conda virtual environment
```bash
conda install pytorch torchvision
pip install albumentations
pip install editdistance
```
# Dataset
This [dataset](https://drive.google.com/drive/folders/1Qa2YA6w6V5MaNV-qxqhsHHoYFRK5JB39) is provided by [Cinamon AI](https://cinnamon.is/vi/) for Cinamon's AI Challenge.

# Preprocessing
In this step, we have to 
- Convert images to grayscale and apply Ostu'threshold to remove backgrounds
- Remove noise
- Smooth boundaries by applying Contour Filter

![image](https://user-images.githubusercontent.com/71833423/163554719-ba48adbc-9d71-4eb7-8e20-239565c84089.png)

# Training 
I divided training process into 3 phases:
- Phase 1: 30 epochs, freezed VGG ,lr = 1e-3
- Phase 2: 30 epochs, unfreezed VGG, lr = 1e-4
- Phase 3: 40 epochs, unfreezed VGG, lr = 1e-5
```bash
python train.py --epoch [num of epochs] --img_path [path to img directory] --label_path [path to label directory] --lr [learning rate] --batch_size [batchsize] --ft [finetune: true or false] --mode [decode mode: 'greedy' or 'beam']
```
![image](https://user-images.githubusercontent.com/71833423/163555701-29d56de7-be85-4f22-a8f3-609137a59af7.png)

# Result
	
![image](https://user-images.githubusercontent.com/71833423/162364408-ef9347e9-1239-4f52-8a72-ae38d707dac9.png)
![image](https://user-images.githubusercontent.com/71833423/162364495-916d8b5e-a57e-439c-8ca4-e10d95d43a16.png)


