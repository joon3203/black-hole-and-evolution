"""
실시간 블랙홀 (Real-time Schwarzschild Black Hole)
==================================================
black_hole_raytracer.py 와 같은 물리(슈바르츠실트 광자 궤적 + 도플러 + 적색편이)를
GPU 셰이더로 옮겨서 실시간으로 돌립니다.

조작법
  마우스 드래그   : 시점 회전
  스크롤          : 줌 인/아웃
  Space           : 원반 회전 일시정지
  D               : 도플러 효과 on/off
  R               : 자동 카메라 회전 on/off
  [ / ]           : 렌더 해상도 낮추기/높이기 (버벅이면 [ 누르기)
  ESC             : 종료

설치/실행 (맥):
  pip install moderngl moderngl-window
  python black_hole_realtime.py
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

        // 원반(y=0) 통과
        if (x.y * xn.y < 0.0) {
            float f = x.y / (x.y - xn.y);
            vec3 p = mix(x, xn, f);
            float pr = length(p);
            if (pr > DISK_IN && pr < DISK_OUT) {
                vec4 e = diskEmission(p, mix(v, vn, f));
                col += T * e.a * e.rgb;
                T *= 1.0 - e.a;
            }
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
        self.fbo = None
        self._make_fbo()

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
        if not self.paused:
            self.sim_time += frametime
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
        elif key == keys.LEFT_BRACKET:
            self.scale = max(0.15, self.scale - 0.1); self._make_fbo()
        elif key == keys.RIGHT_BRACKET:
            self.scale = min(1.0, self.scale + 0.1); self._make_fbo()


if __name__ == "__main__":
    mglw.run_window_config(BlackHole)
