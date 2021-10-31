# KOfish(Korean fast and interactive Style Handler)
### 부자연스러운 한국어를 자연스러운 한국어로 변환하는 Cross Aligned Model

<br> 
GAN을 모태로하는 Style Transfer모델은 이미지 분야에서 주로 연구되어 왔으나, 2017년을 기점으로 자연어처리에서도 연구되었습니다.
최근 3년 동안 연구된 자연어 분야에서의 style transfer는 주로 감성 분석에 바탕해 긍정문을 부정문으로, 혹은 부정문을 긍정문으로 바꾸는 작업이 주된 작업이었습니다.
KOfish에서는 한국의 작가 웹진 브런치에서 크롤링한 데이터와 AI hub 구어체 데이터와 이를 역번역한 데이터를 통해 원문인 자연스러운 한국어와 역번역문인 부자연스러운 한국어 말뭉치 쌍을 가지고서 Style Trainsfer를 진행했습니다

Baseline 코드로 사용된 엄의섭님의 코드 (https://blog.diyaml.com/teampost/Text-Style-Transfer/)에서 데이터를 브런치 데이터로 바꾸고 classifier를 bart로 적용하는 등 자연스러운 한국어 변환 태스크를 수행하기 위한 변형을 가미하였습니다.


 ## 📌 Dependancy 
-   python == 3.7 
-   pytorch >= 1.4.0
  


 ## 🐕 EXAMPLES
 
 1. 브런치 데이터 크롤링


 2. 스타일 트랜스퍼 진행
    options.py 에서 train, test, val_text파일 설정 후 
    
    1) Classifier 모델 훈련

        ```
       python bert_pretrained/classifier.py --ckpt_path "./ckpt" --clf_ckpt_path "./clf_ckpt" 
       
       ```

    2) Style Transfer 모델 훈련
    
        ```
        python train.py --ckpt_path "./ckpt" --clf_ckpt_path "./clf_ckpt"

         ```
    3) Transfer!

          ```
        python trasfer.py --mode "transfer" --ckpt_path "./ckpt" --clf_ckpt_path "./clf_ckpt"

         ```
    

   ## 🍰 Results

      <img width="620" alt="스크린샷 2021-10-31 오후 12 31 27" src="https://user-images.githubusercontent.com/84896185/139566089-92580684-ba93-4a7d-a7ff-a481042b6d9e.png">



  ACC가 0.6의 결과는 참담했습니다. 왜냐하면 딥러닝 모델이 자연스러운 한국어와 부자연스러운 한국어"의 맥락 상의 차이를 발견하지 못했기 때문입니다. classifer도 구분확률이 0.5를 상회했기 때문에 원문과 역번역문의 구분이 거의 안됨을 확인할 수 있었습니다. bart classifier의 경우는 0.8이 넘는 구분 성능을 보여주었지만, 이 또한 성능 평가 지표로만 활용될 뿐 실제 훈련에 영향을 주지 않기 때문에 성능 향상에는 어려움을 겪었습니다. 

   ## 🍡 References
    - https://blog.diyaml.com/teampost/Text-Style-Transfer/
    - Style Transfer from Non-Parallel Text by Cross-Alignment, Tianxiao Shen et al, NIPS 2017

