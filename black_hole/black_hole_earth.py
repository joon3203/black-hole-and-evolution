"""
블랙홀에 빨려들어가는 지구 (Real-time)
======================================
black_hole_realtime.py 에 지구를 추가한 버전.

- 지구 궤도: 슈바르츠실트 시공간의 질량 입자 운동 방정식으로 적분
  (각운동량이 작아서 한두 번 휘돌다가 사건의 지평선으로 추락)
- 스파게티화: 블랙홀에 가까울수록 조석력 때문에 지구가 반지름 방향으로 늘어나고 옆으로 얇아짐
- 중력렌즈: 지구도 광선 추적 안에 있어서, 블랙홀 반대편에 있을 때 지구의 두 번째 상(이중 이미지)이 보임
- 중력 적색편이: 지평선에 다가갈수록 지구가 붉어지고 어두워지다 사라짐
  (바깥 관측자에겐 지평선에 '얼어붙듯' 보이는 효과를 밝기 감소로 표현)
- 지구 표면(대륙, 바다, 극지방 얼음, 구름, 대기)은 전부 절차적 생성. 빛은 강착원반에서 옴

※ 지구 크기는 보기 좋게 과장했습니다 (실제 비율이면 점보다 작음)

조작법
  마우스 드래그   : 시점 회전
  스크롤          : 줌 인/아웃
  E               : 지구 다시 떨어뜨리기
  Space           : 일시정지
  [ / ]           : 시간 느리게/빠르게
  - / =           : 렌더 해상도 낮추기/높이기 (버벅이면 - 누르기)
  D               : 도플러 효과 on/off
  R               : 자동 카메라 회전 on/off
  ESC             : 종료

설치/실행 (맥):
  source ~/venv/bin/activate
  pip install moderngl moderngl-window
  python black_hole_earth.py
"""

import math
import moderngl
import moderngl_window as mglw

# ----------------------------------------------------------------------
# 1패스: 광선 추적 (HDR 결과를 텍스처에 저장)
# ----------------------------------------------------------------------
VERT = """
#version 330
out vec2 uv;
void main() {
    // 화면 전체를 덮는 삼각형 하나 (정점 버퍼 없이)
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    uv = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"""

TRACE_FRAG = """
#version 330
in vec2 uv;
out vec4 fragColor;

uniform vec2  resolution;
uniform float time;
uniform vec3  camPos;
uniform vec3  camRight;
uniform vec3  camUp;
uniform vec3  camFwd;
uniform float tanHalfFov;
uniform float doppler;      // 0 or 1

// 지구
uniform vec3  earthPos;     // 중심 위치
uniform float earthR;       // 원래 반지름
uniform float earthStretch; // 반지름 방향 늘어난 배율 (>=1)
uniform float earthSpin;    // 자전 각
uniform float earthOn;      // 0이면 안 그림

const float DISK_IN  = 3.0;
const float DISK_OUT = 14.0;
const float DISK_TEMP = 5200.0;
const float DISK_BRIGHT = 1.3;
const float ESCAPE_R = 45.0;
const int   MAX_STEPS = 420;
const float FMAX = 0.21396;   // 온도 분포 정규화 상수

// ---------------- 노이즈 ----------------
float hash13(vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.zyx + 31.32);
    return fract((p.x + p.y) * p.z);
}
vec3 hash33(vec3 p) {
    p = fract(p * vec3(0.1031, 0.1030, 0.0973));
    p += dot(p, p.yxz + 33.33);
    return fract((p.xxy + p.yxx) * p.zyx);
}
float valueNoise3(vec3 x) {
    vec3 i = floor(x), f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(mix(hash13(i + vec3(0,0,0)), hash13(i + vec3(1,0,0)), f.x),
                   mix(hash13(i + vec3(0,1,0)), hash13(i + vec3(1,1,0)), f.x), f.y),
               mix(mix(hash13(i + vec3(0,0,1)), hash13(i + vec3(1,0,1)), f.x),
                   mix(hash13(i + vec3(0,1,1)), hash13(i + vec3(1,1,1)), f.x), f.y), f.z);
}
float fbm(vec3 p) {
    float a = 0.5, s = 0.0;
    for (int i = 0; i < 5; i++) { s += a * valueNoise3(p); p *= 2.03; a *= 0.5; }
    return s;
}

// ---------------- 흑체 색 ----------------
vec3 blackbody(float T) {
    float t = clamp(T, 1000.0, 40000.0) / 100.0;
    vec3 c;
    c.r = t <= 66.0 ? 255.0 : 329.698727446 * pow(t - 60.0, -0.1332047592);
    c.g = t <= 66.0 ? 99.4708025861 * log(t) - 161.1195681661
                    : 288.1221695283 * pow(t - 60.0, -0.0755148492);
    c.b = t >= 66.0 ? 255.0 : (t <= 19.0 ? 0.0 : 138.5177312231 * log(t - 10.0) - 305.0447927307);
    c = clamp(c, 0.0, 255.0) / 255.0;
    return pow(c, vec3(2.2));
}

// ---------------- 배경 하늘 ----------------
vec3 starLayer(vec3 d, float scale, float density) {
    vec3 p = d * scale;
    vec3 cell = floor(p);
    vec3 col = vec3(0.0);
    for (int i = 0; i < 2; i++) {        // 이웃 셀 일부만 검사 (가벼움)
        vec3 c = cell + vec3(float(i), 0.0, 0.0) * step(0.5, fract(p.x));
        vec3 h = hash33(c);
        if (h.x > density) {
            vec3 sp = c + 0.2 + 0.6 * hash33(c + 7.0);
            float dist = length(p - sp);
            float b = pow(max(0.0, 1.0 - dist * 2.2), 6.0) * (0.3 + 3.0 * pow(h.y, 6.0));
            float temp = mix(3200.0, 14000.0, h.z * h.z);
            vec3 sc = blackbody(temp); sc /= max(max(sc.r, sc.g), sc.b);
            col += sc * b;
        }
    }
    return col;
}
vec3 sky(vec3 d) {
    vec3 col = starLayer(d, 160.0, 0.86) * 1.6 + starLayer(d, 380.0, 0.9) * 0.9;
    // 은하수 띠 + 성운
    float band = exp(-pow(d.y - 0.3 * sin(atan(d.x, d.z) + 0.6), 2.0) / 0.08);
    float n = fbm(d * 3.0) ;
    float n2 = fbm(d * 7.0 + 3.1);
    vec3 neb = mix(vec3(0.55, 0.45, 0.75), vec3(0.8, 0.55, 0.4), n2);
    col += 0.06 * band * max(n - 0.35, 0.0) * neb * 3.0;
    return col;
}

// ---------------- 강착원반 ----------------
float tempProfile(float r) {
    float x = max(1.0 - sqrt(DISK_IN / r), 0.0);
    return pow(x / (r * r * r), 0.25) / FMAX;
}

vec4 diskEmission(vec3 p, vec3 dir) {
    float r = length(p);
    float phi = atan(p.z, p.x);

    // 케플러 각속도로 원반 무늬를 회전 (안쪽이 더 빠름)
    float omega = sqrt(0.5 / (r * r * r));
    float a = phi - omega * time * 6.0;
    float lr = log(r);
    float sp = a + 2.2 * lr;  // 나선
    vec3 q = vec3(cos(sp) * 3.0, sin(sp) * 3.0, lr * 9.0);
    float turb = fbm(q * vec3(1.0, 1.0, 1.0) + vec3(0.0, 0.0, 0.0));
    float fine = fbm(vec3(cos(sp) * 12.0, sin(sp) * 12.0, lr * 40.0));
    turb = clamp(turb * 0.7 + fine * 0.5 - 0.15, 0.0, 1.0);

    // 상대론 효과
    float beta = min(sqrt(0.5 / max(r - 1.0, 1e-3)), 0.99);
    float gamma = 1.0 / sqrt(1.0 - beta * beta);
    vec3 vdir = vec3(-sin(phi), 0.0, cos(phi));
    vec3 nph = -normalize(dir);
    float g = mix(1.0, 1.0 / (gamma * (1.0 - beta * dot(vdir, nph))), doppler);
    g *= sqrt(max(1.0 - 1.0 / r, 1e-4));

    float tp = tempProfile(r);
    vec3 col = blackbody(DISK_TEMP * tp * g);
    float lum = tp * tp * tp * tp * g * g * g * g;

    float density = 0.35 + 1.1 * pow(turb, 1.6);
    density *= clamp((r - DISK_IN) / 0.4, 0.0, 1.0);
    density *= pow(clamp((DISK_OUT - r) / (0.35 * DISK_OUT), 0.0, 1.0), 1.5);

    float alpha = clamp(density * 0.95, 0.0, 1.0);
    return vec4(col * lum * density * DISK_BRIGHT, alpha);
}

// ---------------- 지구 ----------------
// 반지름 방향(e)으로 earthR*s, 그에 수직으로 earthR/sqrt(s) 인 타원체 (부피 보존)
// 월드 좌표 -> 단위구 좌표
vec3 toEarthLocal(vec3 c, vec3 e, float ra, float rp) {
    float a = dot(c, e);
    return (c - a * e) / rp + e * (a / ra);
}

// 선분 [x0,x1] 이 지구와 만나는지. 만나면 t(0~1), 아니면 -1
float earthHit(vec3 x0, vec3 x1, out vec3 qh) {
    qh = vec3(0.0);
    if (earthOn < 0.5) return -1.0;
    vec3 e = normalize(earthPos);
    float ra = earthR * earthStretch, rp = earthR / sqrt(earthStretch);
    // 빠른 거르기: 선분과 중심 사이 거리
    vec3 c0 = x0 - earthPos, d = x1 - x0;
    float tc = clamp(-dot(c0, d) / max(dot(d, d), 1e-9), 0.0, 1.0);
    if (length(c0 + tc * d) > ra) return -1.0;

    vec3 q0 = toEarthLocal(c0, e, ra, rp);
    vec3 q1 = toEarthLocal(x1 - earthPos, e, ra, rp);
    vec3 dq = q1 - q0;
    float A = dot(dq, dq), B = dot(q0, dq), C = dot(q0, q0) - 1.0;
    float disc = B * B - A * C;
    if (disc < 0.0) return -1.0;
    float t = (-B - sqrt(disc)) / A;
    if (t < 0.0 || t > 1.0) return -1.0;
    qh = q0 + t * dq;
    return t;
}

vec3 earthSurface(vec3 qh, vec3 rayDir) {
    vec3 e = normalize(earthPos);
    float ra = earthR * earthStretch, rp = earthR / sqrt(earthStretch);
    // 타원체 법선 (월드)
    float qa = dot(qh, e);
    vec3 n = normalize((qh - qa * e) / rp + e * (qa / ra));

    // 자전 적용한 몸체 좌표 (y축 기준)
    float cs = cos(earthSpin), sn = sin(earthSpin);
    vec3 b = vec3(cs * qh.x + sn * qh.z, qh.y, -sn * qh.x + cs * qh.z);

    // 대륙 / 바다
    float h = fbm(b * 2.3 + 11.0) + 0.25 * fbm(b * 9.0);
    float land = smoothstep(0.60, 0.63, h);
    vec3 ocean = mix(vec3(0.005, 0.03, 0.12), vec3(0.02, 0.09, 0.22), smoothstep(0.4, 0.6, h));
    float dry = fbm(b * 4.0 + 5.0);
    vec3 ground = mix(vec3(0.05, 0.16, 0.04), vec3(0.32, 0.24, 0.12), smoothstep(0.45, 0.65, dry));
    vec3 albedo = mix(ocean, ground, land);
    // 극지방 얼음
    float ice = smoothstep(0.78, 0.84, abs(b.y) + 0.05 * fbm(b * 6.0));
    albedo = mix(albedo, vec3(0.85, 0.9, 0.95), ice);
    // 구름 (지표보다 살짝 빠르게 흐름)
    float ct = time * 0.05;
    vec3 bc = vec3(b.x * cos(ct) - b.z * sin(ct), b.y, b.x * sin(ct) + b.z * cos(ct));
    float cl = smoothstep(0.5, 0.72, fbm(bc * 3.5 + 40.0));
    albedo = mix(albedo, vec3(0.95), cl * 0.85);

    // 조명: 강착원반(블랙홀 방향)에서 오는 빛 + 약한 별빛
    vec3 toBH = normalize(-earthPos);
    float r = length(earthPos);
    float diskLight = 2.4 / (1.0 + 0.02 * r * r);
    vec3 lightCol = blackbody(4300.0);
    float ndl = max(dot(n, toBH), 0.0);
    // 원반은 넓게 퍼진 광원이라 '감싸는' 조명으로 근사
    float wrap = max((dot(n, toBH) + 0.35) / 1.35, 0.0);
    vec3 lit = albedo * lightCol * diskLight * (0.6 * ndl + 0.4 * wrap) + albedo * 0.02;

    // 바다 반사광
    vec3 vdir = -normalize(rayDir);
    vec3 hv = normalize(toBH + vdir);
    lit += (1.0 - land) * (1.0 - cl) * lightCol * diskLight * 0.6 * pow(max(dot(n, hv), 0.0), 60.0);

    // 밤쪽 도시 불빛
    float city = land * (1.0 - ice) * step(0.72, fbm(b * 25.0)) * (1.0 - smoothstep(-0.1, 0.15, dot(n, toBH)));
    lit += vec3(1.0, 0.65, 0.3) * city * 0.25 * (1.0 - cl);

    // 대기 테두리
    float rim = pow(1.0 - max(dot(n, vdir), 0.0), 3.0);
    lit += vec3(0.25, 0.5, 1.0) * rim * (0.08 + 0.2 * wrap * diskLight);

    // 중력 적색편이: 지평선에 가까울수록 붉고 어두워짐
    float g = sqrt(max(1.0 - 1.0 / max(r, 1.0), 0.0));
    float lum = dot(lit, vec3(0.3, 0.55, 0.15));
    vec3 reddened = vec3(1.0, 0.18, 0.05) * lum * 1.6;
    lit = mix(reddened, lit, smoothstep(0.25, 0.85, g));
    return lit * g * g;
}

// ---------------- 측지선 ----------------
vec3 accel(vec3 x, float h2) {
    float r2 = dot(x, x);
    return -1.5 * h2 * x / pow(r2, 2.5);
}

void main() {
    vec2 ndc = uv * 2.0 - 1.0;
    float aspect = resolution.x / resolution.y;
    vec3 dir = normalize(camFwd + ndc.x * tanHalfFov * camRight
                                + ndc.y * tanHalfFov / aspect * camUp);

    vec3 x = camPos;
    vec3 v = dir;
    vec3 L = cross(x, v);
    float h2 = dot(L, L);

    vec3 col = vec3(0.0);
    float T = 1.0;

    for (int i = 0; i < MAX_STEPS; i++) {
        float r = length(x);
        float dt = clamp(0.04 * r * min(1.0, (r - 0.9) / 1.5), 0.01, 1.2);

        // RK4
        vec3 k1v = accel(x, h2),                 k1x = v;
        vec3 k2v = accel(x + 0.5 * dt * k1x, h2), k2x = v + 0.5 * dt * k1v;
        vec3 k3v = accel(x + 0.5 * dt * k2x, h2), k3x = v + 0.5 * dt * k2v;
        vec3 k4v = accel(x + dt * k3x, h2),       k4x = v + dt * k3v;
        vec3 xn = x + dt / 6.0 * (k1x + 2.0 * k2x + 2.0 * k3x + k4x);
        vec3 vn = v + dt / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v);

        // 지구와 충돌?
        vec3 qh;
        float te = earthHit(x, xn, qh);

        // 원반(y=0) 통과
        if (x.y * xn.y < 0.0) {
            float f = x.y / (x.y - xn.y);
            vec3 p = mix(x, xn, f);
            float pr = length(p);
            // 지구가 원반보다 앞에 있으면 원반은 건너뜀
            if (pr > DISK_IN && pr < DISK_OUT && !(te >= 0.0 && te < f)) {
                vec4 e = diskEmission(p, mix(v, vn, f));
                col += T * e.a * e.rgb;
                T *= 1.0 - e.a;
            }
        }
        if (te >= 0.0) {
            col += T * earthSurface(qh, mix(v, vn, te));
            T = 0.0;
            break;
        }
        x = xn; v = vn;
        r = length(x);

        if (r < 1.0) { T = 0.0; break; }                       // 사건의 지평선
        if (T < 0.005) break;                                   // 원반에 가려짐
        if (r > ESCAPE_R && dot(x, v) > 0.0) {                 // 탈출 -> 배경
            col += T * sky(normalize(v));
            break;
        }
    }
    fragColor = vec4(col, 1.0);
}
"""

# ----------------------------------------------------------------------
# 2패스: 업스케일 + 블룸 + 톤매핑
# ----------------------------------------------------------------------
POST_FRAG = """
#version 330
in vec2 uv;
out vec4 fragColor;
uniform sampler2D hdr;
uniform float maxLod;

vec3 aces(vec3 x) {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}
void main() {
    vec3 c = texture(hdr, uv).rgb;
    // 밉맵을 이용한 저렴한 블룸
    vec3 bloom = vec3(0.0);
    float w[4] = float[4](0.5, 0.35, 0.25, 0.15);
    for (int i = 0; i < 4; i++) {
        float lod = min(2.0 + float(i) * 1.5, maxLod);
        bloom += w[i] * max(textureLod(hdr, uv, lod).rgb - 0.6, 0.0);
    }
    c += bloom;
    c = aces(c * 1.1);
    c = pow(c, vec3(1.0 / 2.2));
    vec2 q = uv * 2.0 - 1.0;
    c *= 1.0 - 0.25 * pow(clamp(length(q) - 0.5, 0.0, 1.0), 2.0);
    fragColor = vec4(c, 1.0);
}
"""


class Earth:
    """
    슈바르츠실트 시공간에서 질량 입자의 궤도 (Rs=1, M=0.5).
        d²x/dτ² = -M x / r³  -  3 M h² x / r⁵      (h = |x × v|, 각운동량)
    두 번째 항이 일반상대론 보정. 이 항 때문에 h < 2 (=4M) 이면
    원심력 장벽이 무너져서 결국 지평선으로 떨어진다.
    """
    M = 0.5
    RADIUS = 0.75          # 보기 좋게 과장된 지구 반지름
    SPEED = 7.0            # 시뮬레이션 시간 배속 (1초에 7 단위시간)
    START = (-11.0, 2.5, 6.5)
    TANGENT_SPEED = 0.1438 # 접선 속도 (작을수록 빨리 떨어짐)

    def __init__(self):
        self.reset()

    def reset(self):
        self.pos = list(self.START)
        # 살짝 기울어진 궤도면을 갖도록 접선 방향 잡기
        axis = (0.25, 1.0, 0.1)
        t = _cross(axis, self.pos)
        tl = math.sqrt(sum(c * c for c in t))
        self.vel = [c / tl * self.TANGENT_SPEED for c in t]
        self.spin = 0.0
        self.gone = 0.0      # 지평선 통과 후 경과 시간
        self.alive = True

    def _acc(self, x, h2):
        r2 = x[0] ** 2 + x[1] ** 2 + x[2] ** 2
        r = math.sqrt(r2)
        k = -self.M / (r2 * r) * (1.0 + 3.0 * h2 / r2)
        return [k * c for c in x]

    def update(self, dt):
        self.spin += dt * 0.6
        if not self.alive:
            self.gone += dt
            if self.gone > 4.0:      # 4초 쉬고 다시 떨어뜨림
                self.reset()
            return
        T = dt * self.SPEED
        n = max(1, int(T / 0.02))
        h = T / n
        for _ in range(n):
            x, v = self.pos, self.vel
            L = _cross(x, v)
            h2 = sum(c * c for c in L)
            # RK4
            a1 = self._acc(x, h2)
            x2 = [x[i] + 0.5 * h * v[i] for i in range(3)]; v2 = [v[i] + 0.5 * h * a1[i] for i in range(3)]
            a2 = self._acc(x2, h2)
            x3 = [x[i] + 0.5 * h * v2[i] for i in range(3)]; v3 = [v[i] + 0.5 * h * a2[i] for i in range(3)]
            a3 = self._acc(x3, h2)
            x4 = [x[i] + h * v3[i] for i in range(3)]; v4 = [v[i] + h * a3[i] for i in range(3)]
            a4 = self._acc(x4, h2)
            self.pos = [x[i] + h / 6 * (v[i] + 2 * v2[i] + 2 * v3[i] + v4[i]) for i in range(3)]
            self.vel = [v[i] + h / 6 * (a1[i] + 2 * a2[i] + 2 * a3[i] + a4[i]) for i in range(3)]
            if self.r() < 1.0:
                self.alive = False
                print("지구가 사건의 지평선을 넘었습니다. (4초 후 다시)")
                break

    def r(self):
        return math.sqrt(sum(c * c for c in self.pos))

    def stretch(self):
        # 조석력 ~ 1/r³ : 가까워질수록 반지름 방향으로 늘어남 (스파게티화)
        return min(1.0 + 10.0 / self.r() ** 3, 9.0)

    def set_uniforms(self, prog):
        prog["earthPos"].value = tuple(self.pos)
        prog["earthR"].value = self.RADIUS
        prog["earthStretch"].value = self.stretch()
        prog["earthSpin"].value = self.spin
        prog["earthOn"].value = 1.0 if self.alive else 0.0


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


class BlackHole(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "Black Hole - real-time"
    window_size = (1280, 720)
    aspect_ratio = None
    resizable = True
    vsync = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.trace = self.ctx.program(vertex_shader=VERT, fragment_shader=TRACE_FRAG)
        self.post = self.ctx.program(vertex_shader=VERT, fragment_shader=POST_FRAG)
        self.vao_trace = self.ctx.vertex_array(self.trace, [])
        self.vao_post = self.ctx.vertex_array(self.post, [])

        # 카메라 상태
        self.yaw = 0.0
        self.pitch = math.radians(6.0)
        self.dist = 22.0
        self.auto_rotate = False
        self.paused = False
        self.sim_time = 0.0
        self.doppler = 1.0
        self.scale = 0.5          # 렌더 해상도 배율 (레티나에서 0.5면 충분히 선명)
        self.time_scale = 1.0
        self.earth = Earth()
        self.fbo = None
        self._make_fbo()
        print("E: 지구 다시 떨어뜨리기 | Space: 일시정지 | [ ]: 시간 속도 | - =: 해상도")

    # ------------------------------------------------------------------
    def _make_fbo(self):
        w, h = self.wnd.buffer_size
        w, h = max(64, int(w * self.scale)), max(36, int(h * self.scale))
        if self.fbo is not None:
            self.fbo.release(); self.tex.release()
        self.tex = self.ctx.texture((w, h), 4, dtype="f2")
        self.tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        self.fbo = self.ctx.framebuffer(color_attachments=[self.tex])
        self.rsize = (w, h)
        self.max_lod = float(int(math.log2(max(w, h))))
        print(f"렌더 해상도: {w}x{h}  (배율 {self.scale:.2f})")

    def on_resize(self, width, height):
        self._make_fbo()

    # ------------------------------------------------------------------
    def _camera(self):
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        pos = (self.dist * cp * sy, self.dist * sp, -self.dist * cp * cy)
        f = [-c / self.dist for c in pos]
        up0 = (0.0, 1.0, 0.0)
        r = (up0[1] * f[2] - up0[2] * f[1], up0[2] * f[0] - up0[0] * f[2], up0[0] * f[1] - up0[1] * f[0])
        rl = math.sqrt(sum(c * c for c in r)) or 1.0
        r = [c / rl for c in r]
        u = (f[1] * r[2] - f[2] * r[1], f[2] * r[0] - f[0] * r[2], f[0] * r[1] - f[1] * r[0])
        # 살짝 기울인 롤
        roll = math.radians(-8.0)
        cr, sr = math.cos(roll), math.sin(roll)
        r2 = [cr * r[i] + sr * u[i] for i in range(3)]
        u2 = [-sr * r[i] + cr * u[i] for i in range(3)]
        return pos, r2, u2, f

    def on_render(self, time, frametime):
        frametime = min(frametime, 0.05)
        if not self.paused:
            dt = frametime * self.time_scale
            self.sim_time += dt
            self.earth.update(dt)
        if self.auto_rotate:
            self.yaw += frametime * 0.15

        pos, right, up, fwd = self._camera()
        t = self.trace
        t["resolution"].value = self.rsize
        t["time"].value = self.sim_time
        t["camPos"].value = tuple(pos)
        t["camRight"].value = tuple(right)
        t["camUp"].value = tuple(up)
        t["camFwd"].value = tuple(fwd)
        t["tanHalfFov"].value = math.tan(math.radians(52.0) / 2)
        t["doppler"].value = self.doppler
        self.earth.set_uniforms(t)

        self.fbo.use()
        self.vao_trace.render(moderngl.TRIANGLES, vertices=3)
        self.tex.build_mipmaps()

        self.ctx.screen.use()
        self.tex.use(0)
        self.post["hdr"].value = 0
        self.post["maxLod"].value = self.max_lod
        self.vao_post.render(moderngl.TRIANGLES, vertices=3)

    # ------------------------------------------------------------------
    def on_mouse_drag_event(self, x, y, dx, dy):
        self.yaw -= dx * 0.005
        self.pitch = max(-1.45, min(1.45, self.pitch + dy * 0.005))

    def on_mouse_scroll_event(self, x_offset, y_offset):
        self.dist = max(6.0, min(60.0, self.dist * (0.93 ** y_offset)))

    def on_key_event(self, key, action, modifiers):
        keys = self.wnd.keys
        if action != keys.ACTION_PRESS:
            return
        if key == keys.SPACE:
            self.paused = not self.paused
        elif key == keys.D:
            self.doppler = 0.0 if self.doppler > 0.5 else 1.0
            print("도플러:", "ON" if self.doppler else "OFF")
        elif key == keys.R:
            self.auto_rotate = not self.auto_rotate
        elif key == keys.E:
            self.earth.reset()
            print("지구 투하!")
        elif key == keys.LEFT_BRACKET:
            self.time_scale = max(0.125, self.time_scale / 2); print(f"시간 x{self.time_scale}")
        elif key == keys.RIGHT_BRACKET:
            self.time_scale = min(8.0, self.time_scale * 2); print(f"시간 x{self.time_scale}")
        elif key == keys.MINUS:
            self.scale = max(0.15, self.scale - 0.1); self._make_fbo()
        elif key == keys.EQUAL:
            self.scale = min(1.0, self.scale + 0.1); self._make_fbo()


if __name__ == "__main__":
    mglw.run_window_config(BlackHole)
