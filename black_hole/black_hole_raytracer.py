"""
슈바르츠실트 블랙홀 레이트레이서 (Schwarzschild Black Hole Ray Tracer)
======================================================================
카메라의 각 픽셀에서 빛을 '거꾸로' 쏴서, 일반상대론의 광자 궤적
(슈바르츠실트 측지선)을 따라 실제로 적분합니다.

구현된 효과
- 중력렌즈: 원반 뒤쪽이 블랙홀 위/아래로 휘어져 보이는 효과 (인터스텔라 스타일)
- 광자 고리(photon ring): 블랙홀을 여러 번 돈 빛이 만드는 얇은 고리
- 사건의 지평선 그림자
- 상대론적 도플러 빔: 다가오는 쪽 원반이 훨씬 밝고 푸르게
- 중력 적색편이: 블랙홀 가까운 빛일수록 붉고 어둡게
- 흑체 복사 색상: 온도 -> 실제 색
- 난류 텍스처가 있는 강착원반 + 렌즈로 왜곡되는 별 배경
- 블룸(빛 번짐) + ACES 톤매핑

단위: 슈바르츠실트 반지름 Rs = 1
필요 패키지: pip install numpy scipy pillow
실행: python black_hole_raytracer.py   ->  black_hole_hq.png
"""

import time
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from PIL import Image

# ======================================================================
# 설정 (여기 값들을 바꿔가며 놀아보세요)
# ======================================================================
WIDTH, HEIGHT = 1920, 1080   # 출력 해상도 (느리면 960x540 으로)
FOV_DEG = 52.0               # 가로 시야각
CAM_DIST = 22.0              # 블랙홀로부터 카메라 거리
CAM_ELEV_DEG = 6.0           # 원반면 위로 몇 도 올라가 있는지 (0이면 정확히 옆에서)
CAM_ROLL_DEG = -8.0          # 화면 기울기 (살짝 기울이면 영화 느낌)

DISK_INNER = 3.0             # 원반 안쪽 반지름 (ISCO = 3 Rs)
DISK_OUTER = 14.0            # 원반 바깥 반지름
DISK_TEMP = 5200.0           # 원반 최고 온도(K) - 높을수록 하얗고 푸르게
DISK_BRIGHTNESS = 1.3        # 원반 밝기 배율
DOPPLER = True               # 도플러 효과 on/off
REDSHIFT = True              # 중력 적색편이 on/off

MAX_STEPS = 900              # 광선당 최대 적분 스텝
ESCAPE_R = 45.0              # 이 거리 밖으로 나가면 배경으로 판정
SEED = 7

rng = np.random.default_rng(SEED)


# ======================================================================
# 유틸: 흑체 온도 -> RGB (근사식, 벡터화)
# ======================================================================
def blackbody_rgb(T):
    """켈빈 온도 배열 -> 선형 RGB (0~1 근처). Tanner Helland 근사 기반."""
    t = np.clip(T, 1000.0, 40000.0) / 100.0
    r = np.where(t <= 66, 255.0, 329.698727446 * np.power(np.maximum(t - 60, 1e-6), -0.1332047592))
    g = np.where(t <= 66,
                 99.4708025861 * np.log(t) - 161.1195681661,
                 288.1221695283 * np.power(np.maximum(t - 60, 1e-6), -0.0755148492))
    b = np.where(t >= 66, 255.0,
                 np.where(t <= 19, 0.0, 138.5177312231 * np.log(np.maximum(t - 10, 1e-6)) - 305.0447927307))
    rgb = np.stack([r, g, b], axis=-1)
    rgb = np.clip(rgb, 0, 255) / 255.0
    return rgb ** 2.2  # sRGB -> 선형


# ======================================================================
# 배경 하늘 텍스처 (등장방형, equirectangular)
# ======================================================================
def make_sky(w=4096, h=2048):
    print("  배경 별/성운 생성 중...")
    sky = np.zeros((h, w, 3), dtype=np.float32)

    # 성운: 여러 스케일의 노이즈를 섞어 은하수 느낌
    lat = np.linspace(np.pi / 2, -np.pi / 2, h)[:, None]
    lon = np.linspace(-np.pi, np.pi, w)[None, :]
    band = np.exp(-((lat - 0.35 * np.sin(lon * 1.0 + 0.6)) ** 2) / 0.09)  # 기울어진 은하 띠

    neb = np.zeros((h, w), dtype=np.float32)
    for sigma, amp in [(60, 1.0), (25, 0.5), (8, 0.25)]:
        n = gaussian_filter(rng.standard_normal((h, w)).astype(np.float32), sigma, mode="wrap")
        neb += amp * n / (n.std() + 1e-9)
    neb = np.clip(neb * 0.35 + 0.2, 0, None) * band
    tint = np.array([0.55, 0.45, 0.75], dtype=np.float32)        # 보랏빛
    tint2 = np.array([0.8, 0.55, 0.4], dtype=np.float32)         # 주황빛 먼지
    mix = gaussian_filter(rng.random((h, w)).astype(np.float32), 40, mode="wrap")
    mix = (mix - mix.min()) / (mix.max() - mix.min() + 1e-9)
    sky += 0.035 * neb[..., None] * (tint * mix[..., None] + tint2 * (1 - mix[..., None]))

    # 별: 멱법칙 밝기 분포 + 별마다 다른 색온도
    n_stars = 45000
    ys = rng.integers(0, h, n_stars)
    xs = rng.integers(0, w, n_stars)
    # 극지방 과밀 보정 (구면 균일 분포)
    ys = ((np.arccos(rng.uniform(-1, 1, n_stars)) / np.pi) * (h - 1)).astype(int)
    mag = rng.pareto(2.2, n_stars) * 0.25 + 0.03
    temps = rng.choice([3200, 4500, 5800, 7500, 11000, 20000], n_stars,
                       p=[0.25, 0.25, 0.2, 0.15, 0.1, 0.05])
    cols = blackbody_rgb(temps.astype(np.float64)).astype(np.float32)
    cols /= cols.max(axis=1, keepdims=True)
    np.add.at(sky, (ys, xs), cols * mag[:, None].astype(np.float32))

    # 별을 살짝 퍼뜨려 점처럼 보이게
    stars_soft = gaussian_filter(sky, sigma=(0.7, 0.7, 0), mode="wrap")
    return np.maximum(sky * 0.6 + stars_soft * 1.8, 0)


# ======================================================================
# 강착원반 난류 텍스처 (r, phi 좌표)
# ======================================================================
def make_disk_texture(nr=512, nphi=2048):
    print("  강착원반 난류 텍스처 생성 중...")
    tex = np.zeros((nr, nphi), dtype=np.float32)
    for sr, sp, amp in [(3, 60, 1.0), (1.5, 25, 0.6), (0.8, 8, 0.35)]:
        n = gaussian_filter(rng.standard_normal((nr, nphi)).astype(np.float32),
                            (sr, sp), mode=("nearest", "wrap"))
        tex += amp * n / (n.std() + 1e-9)
    tex = (tex - tex.min()) / (tex.max() - tex.min())
    return tex


def sample_disk_texture(tex, r, phi):
    nr, nphi = tex.shape
    # 로그 반지름 + 나선 비틀림 -> 소용돌이 모양
    u = (np.log(r / DISK_INNER) / np.log(DISK_OUTER / DISK_INNER)) * (nr - 1)
    spiral = phi + 2.2 * np.log(r)
    v = (spiral % (2 * np.pi)) / (2 * np.pi) * nphi
    return map_coordinates(tex, [u, v], order=1, mode="grid-wrap")


# ======================================================================
# 원반 방출 모델
# ======================================================================
def disk_temperature_profile(r):
    """Shakura–Sunyaev / Novikov–Thorne 형태의 온도 분포 (정규화)."""
    x = np.maximum(1.0 - np.sqrt(DISK_INNER / r), 0.0)
    f = (x / r ** 3) ** 0.25
    rr = np.linspace(DISK_INNER, DISK_OUTER, 2000)
    fmax = ((1.0 - np.sqrt(DISK_INNER / rr)) / rr ** 3).max() ** 0.25
    return f / fmax


def disk_emission(p, d, tex):
    """
    p: 원반과의 교차점 (N,3), d: 그 지점에서 광선(역방향) 진행 방향 (N,3)
    반환: (rgb 방출, 불투명도 alpha)
    """
    r = np.linalg.norm(p, axis=1)
    phi = np.arctan2(p[:, 2], p[:, 0])

    # 케플러 궤도 속도 (Schwarzschild, Rs=1 => M=1/2):  v = sqrt(M / (r - 2M))
    beta = np.sqrt(0.5 / np.maximum(r - 1.0, 1e-3))
    beta = np.minimum(beta, 0.99)
    gamma = 1.0 / np.sqrt(1.0 - beta ** 2)
    # 원반 회전 방향 (y축 기준 반시계)
    vel_dir = np.stack([-np.sin(phi), np.zeros_like(phi), np.cos(phi)], axis=1)

    # 관측자 쪽으로 가는 광자 방향 = 역추적 방향의 반대
    n_photon = -d / np.linalg.norm(d, axis=1, keepdims=True)
    cos_t = np.sum(vel_dir * n_photon, axis=1)

    g = np.ones_like(r)
    if DOPPLER:
        g *= 1.0 / (gamma * (1.0 - beta * cos_t))
    if REDSHIFT:
        g *= np.sqrt(np.maximum(1.0 - 1.0 / r, 1e-4))

    T_emit = DISK_TEMP * disk_temperature_profile(r)
    T_obs = T_emit * g
    color = blackbody_rgb(T_obs)

    # 관측 세기 ~ g^4 * T^4 (보존량 I/nu^3 이용한 근사)
    lum = (disk_temperature_profile(r) ** 4) * g ** 4

    # 난류 + 바깥쪽으로 갈수록 희미해지는 밀도
    turb = sample_disk_texture(tex, r, phi)
    density = (0.35 + 1.1 * turb ** 1.6)
    edge_in = np.clip((r - DISK_INNER) / 0.4, 0, 1)
    edge_out = np.clip((DISK_OUTER - r) / (0.35 * DISK_OUTER), 0, 1) ** 1.5
    density *= edge_in * edge_out

    alpha = np.clip(density * 0.95, 0, 1)
    rgb = color * (lum * density * DISK_BRIGHTNESS)[:, None]
    return rgb, alpha


# ======================================================================
# 카메라
# ======================================================================
def camera_rays():
    elev = np.radians(CAM_ELEV_DEG)
    cam_pos = np.array([0.0, CAM_DIST * np.sin(elev), -CAM_DIST * np.cos(elev)])
    forward = -cam_pos / np.linalg.norm(cam_pos)
    world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(world_up, forward); right /= np.linalg.norm(right)
    up = np.cross(forward, right)

    roll = np.radians(CAM_ROLL_DEG)
    right, up = (np.cos(roll) * right + np.sin(roll) * up,
                 -np.sin(roll) * right + np.cos(roll) * up)

    aspect = WIDTH / HEIGHT
    half_w = np.tan(np.radians(FOV_DEG) / 2)
    half_h = half_w / aspect
    xs = (np.arange(WIDTH) + 0.5) / WIDTH * 2 - 1
    ys = 1 - (np.arange(HEIGHT) + 0.5) / HEIGHT * 2
    px, py = np.meshgrid(xs * half_w, ys * half_h)

    dirs = (forward[None, None, :] + px[..., None] * right + py[..., None] * up).reshape(-1, 3)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    pos = np.broadcast_to(cam_pos, dirs.shape).copy()
    return pos, dirs


# ======================================================================
# 광선 적분 (슈바르츠실트 광자 궤적)
#   d²x/dλ² = -1.5 * h² * x / r⁵     (h = |x × v| , Rs = 1)
# ======================================================================
def accel(x, h2):
    r2 = np.einsum("ij,ij->i", x, x)
    return -1.5 * h2[:, None] * x / (r2 ** 2.5)[:, None]


def trace(pos, vel, sky, disk_tex):
    N = pos.shape[0]
    color = np.zeros((N, 3))
    transmit = np.ones(N)
    final_dir = np.zeros((N, 3))
    escaped = np.zeros(N, dtype=bool)

    h2_all = np.sum(np.cross(pos, vel) ** 2, axis=1)  # 보존량 (각운동량²)
    idx = np.arange(N)
    x, v, h2 = pos.copy(), vel.copy(), h2_all.copy()

    t0 = time.time()
    for step in range(MAX_STEPS):
        if idx.size == 0:
            break
        r = np.linalg.norm(x, axis=1)
        # 적응형 스텝: 블랙홀 근처에선 촘촘하게, 멀리선 성큼성큼
        dt = np.clip(0.035 * r * np.minimum(1.0, (r - 0.9) / 1.5), 0.008, 1.2)[:, None]

        # RK4
        k1v = accel(x, h2);               k1x = v
        k2v = accel(x + 0.5 * dt * k1x, h2); k2x = v + 0.5 * dt * k1v
        k3v = accel(x + 0.5 * dt * k2x, h2); k3x = v + 0.5 * dt * k2v
        k4v = accel(x + dt * k3x, h2);       k4x = v + dt * k3v
        x_new = x + dt / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
        v_new = v + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)

        # --- 원반(y=0 평면) 통과 검사 ---
        cross = (x[:, 1] * x_new[:, 1]) < 0
        if cross.any():
            ci = np.nonzero(cross)[0]
            f = x[ci, 1] / (x[ci, 1] - x_new[ci, 1])
            p = x[ci] + f[:, None] * (x_new[ci] - x[ci])
            pr = np.linalg.norm(p, axis=1)
            ok = (pr > DISK_INNER) & (pr < DISK_OUTER)
            if ok.any():
                ci, p = ci[ok], p[ok]
                d = v[ci] + f[ok][:, None] * (v_new[ci] - v[ci])
                rgb, a = disk_emission(p, d, disk_tex)
                gid = idx[ci]
                color[gid] += transmit[gid, None] * a[:, None] * rgb
                transmit[gid] *= (1 - a)

        x, v = x_new, v_new
        r = np.linalg.norm(x, axis=1)

        # --- 종료 조건 ---
        fell = r < 1.0                                           # 사건의 지평선
        outward = np.einsum("ij,ij->i", x, v) > 0
        esc = (r > ESCAPE_R) & outward                           # 무한히 탈출
        opaque = transmit[idx] < 0.003                           # 원반에 완전히 가려짐
        done = fell | esc | opaque
        if esc.any():
            final_dir[idx[esc]] = v[esc]
            escaped[idx[esc]] = True
        if done.any():
            keep = ~done
            idx, x, v, h2 = idx[keep], x[keep], v[keep], h2[keep]

        if step % 100 == 0:
            print(f"  step {step:4d} | 남은 광선 {idx.size:>9,d} | {time.time() - t0:5.1f}s")

    # 남은 광선(광자구 근처를 계속 도는 것) 은 그냥 검게 처리
    # --- 배경 하늘 샘플링 ---
    e = np.nonzero(escaped)[0]
    d = final_dir[e] / np.linalg.norm(final_dir[e], axis=1, keepdims=True)
    lon = np.arctan2(d[:, 0], d[:, 2])
    lat = np.arcsin(np.clip(d[:, 1], -1, 1))
    sh, sw, _ = sky.shape
    u = (0.5 - lat / np.pi) * (sh - 1)
    w = (lon / (2 * np.pi) + 0.5) * sw
    bg = np.stack([map_coordinates(sky[..., c], [u, w], order=1, mode="grid-wrap") for c in range(3)], axis=1)
    color[e] += transmit[e, None] * bg
    return color


# ======================================================================
# 후처리: 블룸 + ACES 톤매핑 + 감마
# ======================================================================
def aces(x):
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0, 1)


def postprocess(img):
    print("  블룸 + 톤매핑...")
    bright = np.maximum(img - 0.8, 0)
    bloom = np.zeros_like(img)
    for s, w in [(2, 0.5), (8, 0.35), (24, 0.25), (60, 0.15)]:
        bloom += w * gaussian_filter(bright, sigma=(s, s, 0))
    img = img + bloom
    img = aces(img * 1.1)
    img = np.power(img, 1 / 2.2)
    # 비네팅
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH]
    rr = np.sqrt(((xx - WIDTH / 2) / (WIDTH / 2)) ** 2 + ((yy - HEIGHT / 2) / (HEIGHT / 2)) ** 2)
    img *= (1 - 0.25 * np.clip(rr - 0.5, 0, 1) ** 2)[..., None]
    return np.clip(img, 0, 1)


# ======================================================================
def main():
    t_start = time.time()
    print("[1/4] 텍스처 준비")
    sky = make_sky()
    disk_tex = make_disk_texture()

    print(f"[2/4] 카메라 광선 생성 ({WIDTH}x{HEIGHT} = {WIDTH * HEIGHT:,} 개)")
    pos, dirs = camera_rays()

    print("[3/4] 측지선 적분 (시간이 좀 걸려요)")
    col = trace(pos, dirs, sky, disk_tex)
    img = col.reshape(HEIGHT, WIDTH, 3)

    print("[4/4] 후처리")
    img = postprocess(img)
    Image.fromarray((img * 255).astype(np.uint8)).save("black_hole_hq.png")
    print(f"완료! black_hole_hq.png 저장 ({time.time() - t_start:.1f}s)")


if __name__ == "__main__":
    main()
