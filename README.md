## 1. 프로젝트 소개

  # 명칭
  물류창고에서 이동체 위치 추적을 위한 BLE Zoning 시스템 개발

  # 목적
  BLE를 활용한 물류 작업의 정확한 추적과 근로자의 움직임을 실시간으로 관리.


## 2. 팀 소개

  - **강중헌** - 201824409, wwww1542@pusan.ac.kr, 데이터베이스 구축 및 웹서비스 구현
  - **김지명** - 201824454, ehxhfl1589@pusan.ac.kr, 모델최적화 및 모델결과분석
  - **이정현** - 201824556,  dlwkgh6906@pusan.ac.kr, 데이터 수집 및 실험환경구축


## 3. 구성도

  # 모델 구성
  ![image](https://github.com/pnucse-capstone/capstone-2023-1-31/assets/128675907/27674e59-9468-4c0d-8836-f1e14ad8fe54)

  # Zoning System Web Service
  ![image](https://github.com/pnucse-capstone/capstone-2023-1-31/assets/128675907/d93f931b-6aa7-4cc8-9cbb-62de2b04d26e)
  빨간 포인트를 통해 BLE beacon 위치를 추정할 수 있으며, 현재 어느 Zone에 위치했는지 추정할 수 있다.
  하단에는 각 BLE Scanner에서 수집한 신호세기를 볼 수 있다.
  각 Scanner의 위치는 1~6 순서로 (0,0), (5,0), (10,0), (0,5), (5,5), (10,5) (m)에 위치한다.
  

## 4. 소개 및 시연 영상

  예정.

## 5. 사용법

  1. BLE Scanner를 공간에 설치하고 scanner의 저장된 WIFI 정보와 일치하는 WIFI환경을 구축한다.

  2. BLE Beacon의 전원을 연결하여 무선신호를 BLE Scanner로 보낸다.

  3. web\main.py를 실행하면 자신의 IP 5000번 포트 혹은 http://127.0.0.1:5000으로 접속하여 실시간으로 
   BLE가 부착된 물체의 위치에 대한 Zoning을 확인 할 수 있다.
