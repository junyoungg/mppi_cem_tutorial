import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    # CSV 파일을 pandas DataFrame으로 읽어옵니다.
    df = pd.read_csv('/src/envs/circuit_generator/circuit.csv')

    # 컬럼 이름에서 공백과 '#' 문자를 제거하여 쉽게 접근할 수 있도록 합니다.
    df.columns = [col.strip().replace('# ', '') for col in df.columns]

    # 중심선 좌표와 좌우 트랙 폭을 변수에 저장합니다.
    center_x = df['x_m'].values
    center_y = df['y_m'].values
    width_left = df['w_tr_left_m'].values
    width_right = df['w_tr_right_m'].values

    # 트랙의 좌우 경계선 좌표를 계산합니다.
    # 각 점에서의 방향 벡터(tangent)를 계산하고, 그에 수직인 법선 벡터(normal)를 구합니다.
    points = np.vstack([center_x, center_y]).T
    
    # np.gradient를 사용하여 각 점에서의 접선 방향을 계산합니다.
    dx, dy = np.gradient(points[:, 0]), np.gradient(points[:, 1])
    
    # 법선 벡터를 계산합니다 (방향 벡터를 90도 회전).
    # 크기를 1로 정규화합니다.
    normals = np.array([-dy, dx]).T
    norm_lengths = np.sqrt(np.sum(normals**2, axis=1))
    
    # 0으로 나누는 것을 방지하기 위해 매우 작은 값을 더해줍니다.
    unit_normals = normals / (norm_lengths[:, np.newaxis] + 1e-6)

    # 좌우 경계선 좌표 계산
    left_boundary = points + unit_normals * width_left[:, np.newaxis]
    right_boundary = points - unit_normals * width_right[:, np.newaxis]

    # --- 시각화 ---
    plt.figure(figsize=(8, 8))

    # 계산된 트랙 경계선 그리기 (녹색 점선)
    plt.plot(left_boundary[:, 0], left_boundary[:, 1], '--', color='green', label='Track Boundaries')
    plt.plot(right_boundary[:, 0], right_boundary[:, 1], '--', color='green')

    # 중심선 그리기 (회색 점선)
    plt.plot(center_x, center_y, '--', color='gray', label='Center Line')

    # 그래프 제목 및 라벨 설정
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Race Track Visualization from circuit.csv")

    # 가로, 세로 비율을 동일하게 설정합니다.
    plt.axis('equal')

    # 축의 범위를 이미지와 유사하게 설정합니다.
    plt.xlim(-40, 40)
    plt.ylim(-40, 40)

    plt.legend()
    plt.grid(False)
    plt.show()

except FileNotFoundError:
    print("오류: 'circuit.csv' 파일을 찾을 수 없습니다.")
except Exception as e:
    print(f"코드를 실행하는 중 오류가 발생했습니다: {e}")