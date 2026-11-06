"""
블랙홀 시각화 (Black Hole Visualizer)
- 중심의 사건의 지평선(event horizon)
- 광자구(photon sphere) 주변 빛의 휘어짐(간이 중력렌즈 효과)
- 강착원반(accretion disk) - 도플러 효과로 한쪽이 더 밝게

실행하면 black_hole.png 파일이 생성됩니다.
필요 패키지: numpy, matplotlib
    pip install numpy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# -----------------------------
# 기본 설정
# -----------------------------
N = 800                  # 이미지 해상도 (N x N)
R_S = 1.0                # 슈바르츠실트 반지름 (사건의 지평선)
DISK_INNER = 3.0 * R_S   # 강착원반 안쪽 경계 (ISCO 근사)
DISK_OUTER = 9.0 * R_S   # 강착원반 바깥쪽 경계
VIEW_SIZE = 12.0         # 화면에 보이는 좌표 범위 (-VIEW_SIZE ~ +VIEW_SIZE)

# 화면 좌표 그리드 생성 (관측자 스크린)
x = np.linspace(-VIEW_SIZE, VIEW_SIZE, N)
y = np.linspace(-VIEW_SIZE, VIEW_SIZE, N)
X, Y = np.meshgrid(x, y)

# 관측자로부터 각 픽셀까지의 충격변수(impact parameter, 빛이 블랙홀에서 얼마나 떨어져 지나가는지)
b = np.sqrt(X**2 + Y**2)

# -----------------------------
# 간이 중력렌즈 모델
# -----------------------------
# 실제 광선 추적(geodesic ray tracing) 대신,
# 충격변수 b에 따라 "빛이 원래 어느 반지름에서 왔는지"를 근사하는 함수를 사용합니다.
# b가 임계충격변수(b_crit ≈ 2.6*R_S, 광자구 관련 값) 이하이면 블랙홀로 빨려들어가 화면엔 검게 표시됩니다.
b_crit = 2.6 * R_S

with np.errstate(divide='ignore', invalid='ignore'):
    # 렌즈 왜곡: 실제 반지름 r_source를 b의 비선형 함수로 근사
    # b가 커질수록 왜곡이 줄어들고, b_crit 근처에서 급격히 휘어짐
    distortion = 1.0 / (1.0 - (b_crit / np.maximum(b, 1e-6))**2)
    distortion = np.clip(distortion, 0, 6)
    r_source = b * distortion

# 사건의 지평선 내부 (검은 부분)
event_horizon_mask = b <= R_S

# -----------------------------
# 강착원반 밝기 계산
# -----------------------------
theta = np.arctan2(Y, X)

# 원반이 있는 반지름 범위인지 확인 (렌즈로 왜곡된 r_source 기준)
disk_mask = (r_source >= DISK_INNER) & (r_source <= DISK_OUTER) & (~event_horizon_mask)

# 온도/밝기: 안쪽일수록 뜨겁고 밝음 (r^-3/4 근사, 실제 강착원반 온도 분포와 유사)
brightness = np.zeros_like(X)
safe_r = np.where(disk_mask, r_source, DISK_OUTER)
brightness = np.where(disk_mask, (DISK_INNER / safe_r) ** 0.75, 0.0)

# 상대론적 도플러 빔 효과: 원반이 회전하면서 관측자 쪽으로 오는 부분이 더 밝게 보임
# 여기서는 X<0 쪽이 다가오는 쪽이라고 가정
doppler = 1.0 + 0.9 * np.cos(theta)  # 0.1 ~ 1.9 배
brightness = brightness * doppler
brightness = np.clip(brightness, 0, None)

# 광자구 근처 강착원반 뒷부분이 "위로 휘어져 보이는" 느낌을 위해 약한 이중 이미지 추가
secondary_mask = (b > R_S) & (b < DISK_OUTER * 0.5) & (~disk_mask)
ghost = np.where(secondary_mask, 0.15 * np.exp(-((b - b_crit) ** 2) / 0.5), 0.0)
brightness += ghost

# -----------------------------
# 색상 매핑 (뜨거운 플라즈마 느낌)
# -----------------------------
cmap = mcolors.LinearSegmentedColormap.from_list(
    "accretion",
    ["#000000", "#200000", "#7a1a00", "#ff4400", "#ffb300", "#fff3c4", "#ffffff"]
)

norm_brightness = brightness / (brightness.max() + 1e-9)

fig, ax = plt.subplots(figsize=(8, 8), facecolor="black")
ax.set_facecolor("black")

img = ax.imshow(
    norm_brightness ** 0.6,   # 감마 보정으로 은은한 부분도 살림
    extent=[-VIEW_SIZE, VIEW_SIZE, -VIEW_SIZE, VIEW_SIZE],
    cmap=cmap,
    origin="lower",
    interpolation="bilinear",
)

# 사건의 지평선 (완전한 검정 원) 덧그리기
horizon_circle = plt.Circle((0, 0), R_S, color="black", zorder=5)
ax.add_patch(horizon_circle)

# 광자구 경계를 은은한 링으로 표시 (선택 사항, 시각적 강조)
photon_ring = plt.Circle((0, 0), b_crit, color="white", fill=False,
                          linewidth=0.6, alpha=0.25, zorder=4)
ax.add_patch(photon_ring)

ax.set_xlim(-VIEW_SIZE, VIEW_SIZE)
ax.set_ylim(-VIEW_SIZE, VIEW_SIZE)
ax.set_xticks([])
ax.set_yticks([])
ax.set_aspect("equal")
for spine in ax.spines.values():
    spine.set_visible(False)

plt.title("Black Hole with Accretion Disk", color="white", fontsize=13, pad=12)
plt.tight_layout()
plt.savefig("black_hole.png", dpi=200, facecolor="black")
print("Saved: black_hole.png")
