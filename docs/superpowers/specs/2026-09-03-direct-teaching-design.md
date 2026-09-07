# 직접 교시(teach) 설계 — 2026-09-03

## 목표
1. 손으로 팔을 옮기면 그 자세를 그대로 유지한다.
2. 그렇게 만든 자세를 이름 붙여 저장하고 다시 만든다.
3. 손으로 시연한 궤적을 기록했다가 반복 재생한다.

대상은 `*_arm` 그룹(JTC 로 구동되는 팔). 손·그리퍼는 범위 밖.

## 제약(코드에서 확인)
- kp/kd 는 bringup 때 하드웨어 파라미터로 고정된다(`control_gains.yaml` → xacro →
  `openarm_simple_hardware.cpp`). 런타임 변경은 벤더 스냅샷 C++ 수정이 필요하므로 하지 않는다.
- 따라서 순응은 **소프트웨어**로 만든다: JTC 목표가 측정값을 따라가고, 중력 피드포워드를
  `*_forward_effort_controller` 로 같이 발행한다. `pose follow` 의 서보 루프와 같은 구조.
- 밀 때 저항 = kd·q̇ + kp·(한 주기 이동량). 어깨 기준 약 3~4 N·m/(rad/s).
- 흘러내림의 원인은 중력 모델 잔차(우팔 실측 약 2°)이며 하드웨어 방식이어도 같다.

## 상태기계(`teaching.py`, ROS 없음, 불변 상태)
- HOLD: 목표 고정. 래치 시점 기준선 `baseline = target − measured` 를 기억한다.
  처짐 `deflection = (target − measured) − baseline` 이 관절별 `push_rad` 를 넘으면 FOLLOW.
  래치 직후 `settle_sec` 동안은 감지를 멈추고 그 뒤 기준선을 다시 잰다.
- FOLLOW: 매 주기 원하는 지령 = `measured + baseline`. 호출자가 `CommandGate.follow` 로
  속도·lead·한계를 클램프하고 실제 지령을 `acknowledge` 로 돌려준다.
  모든 관절 속도가 `still_rad_s` 아래로 `still_sec` 이상이면 래치(HOLD).
  `max_follow_sec` 를 넘기면 흘러내림으로 보고 강제 래치.
- 잠금(space): 잠기면 FOLLOW 진입 불가, FOLLOW 중이면 즉시 래치.

## 명령(`teach_cli.py`)
- `teach hold`   : 순응 세션. 키 space=잠금, s=현재 자세 저장, q=종료.
- `teach record` : hold 와 같되 `/joint_states` 를 녹화해 `.npz` 로 저장(앞뒤 정지 구간 제거).
- `teach replay` : 파일을 지령 주기로 리샘플 → `authorize_trajectory` 로 사전 검증(거부) →
  시작점까지 PARK 속도로 접근 → 스트리밍 재생, `--repeat`, `--speed`.
- `teach save/goto/list` : YAML 자세 저장소(`poses/<profile>.yaml`). goto 는 PARK 속도 스트리밍.
- 모든 세션은 중력 ff 를 매 주기 측정 자세로 계산해 발행한다. 종료 시 기본은 토크 해제
  (팔이 처진다고 경고), `--keep-gravity` 면 유지하고 해제 명령을 안내한다.

## 파일
- `src/robot_control/teaching.py` 상태기계·리샘플·정지 구간 제거
- `src/robot_control/poses.py` 자세 저장소
- `src/robot_control/keys.py` 비차단 키 입력
- `src/robot_control/teach_cli.py` 명령·세션 루프
- `cli.py` 는 `teach` 파서 등록과 dispatch 두 줄만 바뀐다.

## 위험
- 흘러내림: `max_follow_sec`·잠금·게이트 속도 클램프가 방어선. 실기 문턱 튜닝 필요.
- 손목 kp 10 → 같은 `push_rad` 에서 토크 문턱이 어깨의 1/7. 관절별 값 허용.
- 우 j7 과열: 세션 길이 `--seconds` 기본 300 s.
- 실기 구동은 사용자 승인 후.
