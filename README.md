# Machine Learning Streamlit App

## 기술 요소
docker
- docker-compose (container management)

python
- streamlit (front-end)
- sklearn (machine learning model)

mongoDB 
- mongoDB (machine learning model management & data storage)

nginx 
- nginx (web server)

AWS
- AWS Lightsail (computing & deployment)


## 사용 방법

### 기본 설정

1 .env 파일 만들기
- 프로젝트 루트 디렉토리에 환경변수 설정을 위한 .env 파일 생성

```shell
# Sample .env file

MONGO_INITDB_ROOT_USERNAME=root
MONGO_INITDB_ROOT_PASSWORD=root
MONGO_INITDB_DATABASE=ml
# local pc 마운트 경로 + /data/db
DB_VOLUMES=<local mount directory>/data/db
MONGO_HOST=db
MONGO_PORT=27017
MONGO_USERNAME=test
MONGO_PASSWORD=test
MONGO_DATA_COLLECTION_NAME=data
MONGO_MODEL_COLLECTION_NAME=model
SAMPLE_FILE_PATH=/data/raw/train.csv
```

2. local pc 마운트 경로 디렉토리 생성

### 배포

1. AWS Lightsail Instance 생성
- OS only Ubuntu 20.04 선택
- Memory 2GB 이상 선택 (안 할 시 pip 라이브러리 설치 시 에러)

2. SSH 접속

3. Instance 환경 설정
- root 패스워드 재설정: 
  - `sudo passwd root`
- 사용자 전환: 
  - `su root`
- 패키지 매니저 업데이트: 
  - `apt update`
- 패키지 매니저 업그레이드 (필수 x): 
  - `apt upgrade`
- 필요 패키지 설치: 
  - `sudo apt-get install apt-transport-https ca-certificates curl gnupg-agent software-properties-common`
  - `curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -`
  - `sudo add-apt-repository \
    "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
    $(lsb_release -cs) \
    stable"`
  - `sudo systemctl enable docker && service docker start`
- pip install 에러 처리: 
  - `apt-get install libssl-dev`

4. `sh init.sh` 로 컨테이너 초기화 및 기본 데이터 불러오기
- `docker-compose up --build -d` 실행됨


### 사용
어플리케이션 접속: `https://<domain ip>:80/`

어플리케이션 멈추기: `docker-compose stop`
어플리케이션 다시 띄우기: `docker-compose start`

어플리케이션 삭제: `docker-compose down` (데이터까지 초기화 시키고 싶다면 data/db/ 경로 파일 지워야 함)
어플리케이션 다시 만들기: `sh init.sh`


## Version

### v1.0.0

Titanic 데이터 사용 샘플 어플리케이션 개발


## Memo

log
- https://hwangheek.github.io/2019/python-logging/
