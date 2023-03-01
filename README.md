# SKKU-DataAnalysisFoundation-AudioClassification

성균관대학교 2022년 1학기 데이터분석기초를 수강하면서 과제로 제출한 프로젝트입니다.

# 1. 분석 주제

- 분석 주제 : 음성 데이터를 이용해 음성 분류 모델 만들기
- 선정 이유 :  데이터분석기초 수업에서 주로 수치화된 데이터를 다뤘기 때문에 프로젝트로는 색다른 것을 시도해 보고 싶었다. 특히 우리가 일상생활에서 쉽게 접할 수 있는 데이터를 생각하다가 ‘음성 데이터’가 떠올랐다. 이에 더해 7주차 수업에서 데이터 분석 방법론을 배우고 머신러닝에 흥미가 생겨 ‘음성 파일을 이용해 음성 분류 모델 만들기’를 분석 주제로 설정하였다.

# 2. 분석 과정

## 2.1 데이터 준비

데이터는 캐글의 데이터셋 활용하였다. 

([https://www.kaggle.com/datasets/rtatman/speech-accent-archive](https://www.kaggle.com/datasets/rtatman/speech-accent-archive))

| 음성 데이터 | 2138개의 녹음 파일이 mp3 형식으로 있음 |
| --- | --- |
| speakers_all.csv | 음성을 녹음한 사람들의 성별, 파일 이름, 국가, 나이 등등 정보가 저장됨 |
| reading-passage.txt | 음성 녹음 대본 |

## 2.2 데이터 확인

### 2.3.1 기본 정보 확인

![Untitled](https://user-images.githubusercontent.com/96484143/222094999-6b80943c-4200-408b-99a9-93b75cdecb1d.png)

- 12개의 특성과 2172개의 샘플
- 필요한 특성
    - sex : 성별 분석을 위한 타깃 라벨로 활용할 수 있음
    - filename : 실제 audio 파일을 이용할 때 필요함
    - file_missing? : 결측치를 찾는 용도로 필요할 것으로 예상됨

### 2.3.2 결측치 확인

![Untitled (1)](https://user-images.githubusercontent.com/96484143/222095159-2b40e0dc-94cf-46e0-81c5-9e384e416ea1.png)

- birthplace와 country에 약간의 결측치가 있지만 타깃과 연관성이 적어 큰 문제는 없어 보임
- Unnamed column 3개는 사용이 어려워 보임

## 2. 3 핵심 특성 확인

### 2.3.1 sex

![Untitled (2)](https://user-images.githubusercontent.com/96484143/222095262-4a037101-2ddc-4a9f-811a-6f79b71e642b.png)

- 두 타깃 값의 분포가 51:38 정도로 적당한 비율을 가지고 있음
- 이상값 “famale” 발견

### 2.3.2 missing_file?

![Untitled (3)](https://user-images.githubusercontent.com/96484143/222095651-9c44414d-cbe2-4812-98fb-f9607a6c338b.png)

- 32개의 파일이 분실되었음

## 2.3 데이터 전처리

### 2.3.1 메타 데이터 전처리

**특성 정리**

![Untitled (4)](https://user-images.githubusercontent.com/96484143/222095874-dddd2dfb-449d-413c-b5b9-de533a0a2040.png)

- 3개의 특성만 남겨두고 나머지 열 삭제

**분실 파일(결측값) 확인 및 처리**

![Untitled (5)](https://user-images.githubusercontent.com/96484143/222095882-65a41479-1373-47cd-804e-0508c1ea368f.png)

- 메타데이터의 샘플 개수와 실제 오디오 파일의 개수가 차이남
- 34개의 오디오 파일이 없음
    
     ⇒ file_missing? 특성으로 표시된 값 외에 2개의 결측치가 존재함
    

**결측치 처리**

![Untitled (6)](https://user-images.githubusercontent.com/96484143/222095887-c4c2b89b-cee2-4159-8573-1a2dffbe1f17.png)

**이상치 수정 및 타겟 레이블 수치화**

![Untitled (7)](https://user-images.githubusercontent.com/96484143/222095892-a77cbff0-b161-4695-a54d-14a4395809f5.png)

- female, famale 을 0으로, male을 1로 수정

### 2.3.2 오디오 데이터 전처리

- librosa 라이브러리를 이용해 오디오 파일에서 특성을 추출하여 기존 메타데이터 데이터 프레임과 결합시킴

![Untitled (8)](https://user-images.githubusercontent.com/96484143/222096063-e30b4f03-5f1f-40a1-a956-bb36ee29532c.png)

- rms : root-mean-squale, 진폭의 제곱평균제곱근
- chroma_stft : pirtch class의 분포
- spec_cent : spectrum centroid, 스펙트럼의 질량중심
- spec_bw : spectral bandwith, 스펙트럼의 대역폭
- rolloff : 전체 스펙트럼 중 일정 비율이 포함되는 주파수
- zcr : zero crossing, 오디오 그래프에서 신호가 x축을 교차하는 횟수
- mfcc : Mel-Frequency Cepstral Coefficient, 음색추출

## 2.3 데이터 탐색 (모델 생성)

- 해당 과정은 음성에 따라 여성과 남성으로 분류하는 ‘이진 분류’이기 때문에 시그모이드 함수를 적용한 **로지스틱 회귀** 모델을 선택함
- 사이킷 런을 이용해 기존 데이터프레임을 x_train_split, y_train_split, x_val, y_val, x_test, y_test 로 쪼갬
    
    ![Untitled (9)](https://user-images.githubusercontent.com/96484143/222096149-1db72150-e458-45fb-876f-69873f8f19c6.png)
    
    - 세 개의 층을 가진 다층 신경망
    - 마지막 출력층에 시그모이드함수를 사용해 이진 분류가 가능하게 함
    - 손실함수로는 크로스엔트로피 함수를 이용
    - 300번의 epoch로 모델 훈련

# 3. 분석 결과 Review

## 3.1 모델 훈련 결과

![Untitled (10)](https://user-images.githubusercontent.com/96484143/222096454-e3384e36-ff3c-462c-b771-380f7d5a2f02.png)

![Untitled (12)](https://user-images.githubusercontent.com/96484143/222096483-9684a5d3-5f5d-46a1-b5fe-217e184d7249.png)

![Untitled (11)](https://user-images.githubusercontent.com/96484143/222096496-17a53e9f-4923-4829-8fde-a9de6863535f.png)

## 3.2 결과 분석 및 한계점

이 프로젝트에서는 오디오 데이터를 이용해 오디오 파일에서 7개의 특성 (rms, chroma_stft, spec_cent, spec_bw, rolloff, zcr, mfcc)을 이용해 목소리의 주인이 여자인지 남자인지 판단을 내려주는 모델을 만들었다. 모델을 만들어 데이터 분석을 하는 것은 대량의 데이터를 이용해 그 데이터의 규칙을 찾아내는 과정인데 이는 블랙박스 알고리즘으로 그 규칙은 사람이 해석할 수 없다. 

모델의 정확도는 66%로 실제로 사용하기에는 많이 부족한 정확도이다. 머신러닝을 하기에 데이터 수가 부족했을 수도 있고 더 적절한 모델 파라미터를 찾지 못했을 수도 있다. 아직 모델 정확도를 끌어올릴 수 있을만한 실력은 되지 않기 때문에 더 많은 공부가 필요하다는 것을 느꼈다.

## 3.3 wisdom

그렇다면 이렇게 데이터 분석을 이용해 생성한 모델은 어떻게 활용될 수 있을까? 만약 좀 더 정확도가 높은 모델이였다고 가정한다면, 그리고 해당 모델을 가지고 추가적인 응용이 들어간다면 활용 방안은 무궁무진하다. 프로젝트에서 사용했던 데이터가 아닌 새로운 음성 파일을 입력하여 그 음성의 성별을 판별할 수 있다. 이와 같은 응용 프로그램은 성별에 따른 서비스를 제공하는 기업에서 활발하게 이용할 수 있을 것이다. 예를 들어 최근 활성화가 된 ‘AI 스피커’의 경우 사용자와 감정적인 소통을 하기도 하는데 사용자의 목소리로 성별을 알아내 그에 맞는 적절한 대화 방식을 선택할 수 있다.
