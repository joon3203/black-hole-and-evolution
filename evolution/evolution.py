"""
진화하는 AI 생명체 (Neuroevolution Ecosystem)
=============================================
생명체마다 작은 신경망 '뇌'가 있습니다.
  입력 7개 (먹이 위치, 위험 위치, 배고픔)  ->  은닉층 12개  ->  출력 2개 (회전, 속도)

아무도 '먹이 쪽으로 가라'고 가르치지 않습니다.
처음엔 뇌가 완전히 랜덤이라 멍청하게 빙빙 돌기만 하지만,
- 먹이를 잘 먹은 놈은 에너지가 쌓여 번식하고 (뇌가 살짝 변이된 자식)
- 못 먹은 놈은 굶어 죽고
- 포식자에게 잡힌 놈도 죽습니다.
이게 반복되면서 세대가 지날수록 점점 똑똑해집니다. (자연선택)

  초록 계열 = 초식 생명체 (먹이 = 노란 점)
  빨강 계열 = 포식자 (먹이 = 초식 생명체)
  같은 가문(혈통)은 비슷한 색. 변이가 쌓이면 색도 조금씩 달라짐

조작법
  클릭       : 생명체 선택 -> 그 녀석의 뇌(신경망)가 실시간으로 보임
  F          : 시간 배속 (x1 / x4 / x16 / x64)  - 빨리 진화시키기
  P          : 포식자 등장/퇴장
  Space      : 일시정지
  H          : 정보창 숨기기/보이기
  R          : 처음부터 다시
  ESC        : 종료

실행 (맥):
  source ~/venv/bin/activate
  python evolution.py            # 전체화면 (ESC로 종료)
  python evolution.py --window   # 화면을 꽉 채우는 일반 창
  python evolution.py --small    # 작은 창 (1280x760)
(필요한 numpy, pyglet 은 앞에서 이미 설치됨)
"""

import math
import numpy as np

# ======================================================================
# 설정
# ======================================================================
W, H = 1280, 760          # 세계 크기 = 창 크기 (가장자리는 반대편과 이어짐)
FOOD_N = 150              # 먹이 개수 (먹히면 다른 곳에 새로 생김)
FOOD_ENERGY = 40.0
N_IN, N_HID, N_OUT = 7, 12, 2

PREY = dict(
    name="초식", start_n=80, min_n=25, max_n=170,
    max_speed=2.6, turn_rate=0.22, sight=150.0, radius=6.0,
    cost_base=0.10, cost_move=0.22,
    start_energy=100.0, repro_energy=170.0, max_energy=220.0, max_age=4000,
)
PRED = dict(
    name="포식자", start_n=8, min_n=4, max_n=30,
    max_speed=3.4, turn_rate=0.18, sight=220.0, radius=9.0,
    cost_base=0.11, cost_move=0.30,
    start_energy=220.0, repro_energy=280.0, max_energy=380.0, max_age=6000,
)
PREY_ENERGY_FOR_PRED = 120.0

INPUT_NAMES = ["먹이 앞뒤", "먹이 좌우", "먹이 보임", "위험 앞뒤", "위험 좌우", "위험 보임", "배부름"]
OUTPUT_NAMES = ["회전", "속도"]

rng = np.random.default_rng()


def wrap_delta(d):
    """토러스 세계에서 가장 짧은 차이 벡터."""
    d[..., 0] = (d[..., 0] + W / 2) % W - W / 2
    d[..., 1] = (d[..., 1] + H / 2) % H - H / 2
    return d


def nearest(src, tgt, exclude_self=False):
    """src 각각에 대해 tgt 중 가장 가까운 것의 (상대벡터, 거리, 인덱스)."""
    n = len(src)
    if n == 0 or len(tgt) == 0 or (exclude_self and len(tgt) < 2):
        return np.zeros((n, 2)), np.full(n, np.inf), np.full(n, -1)
    d = wrap_delta(tgt[None, :, :] - src[:, None, :])
    d2 = np.einsum("ijk,ijk->ij", d, d)
    if exclude_self:
        np.fill_diagonal(d2, np.inf)
    idx = np.argmin(d2, axis=1)
    ar = np.arange(n)
    return d[ar, idx], np.sqrt(d2[ar, idx]), idx


# ======================================================================
# 종 (한 종의 모든 개체를 numpy 배열로 한꺼번에 관리)
# ======================================================================
class Species:
    FIELDS = ["pos", "heading", "energy", "age", "gen", "eaten", "hue", "uid",
              "W1", "b1", "W2", "b2"]

    def __init__(self, cfg, hue_center):
        self.cfg = cfg
        self.hue_center = hue_center
        self.next_uid = 0
        self.chase_acc = 0.5
        self.archive = []   # 지금까지 가장 잘한 뇌들 (멸종 방지용)
        self._empty()
        self.add_random(cfg["start_n"])
        # 시각화용 (마지막 생각)
        self.last_in = np.zeros((0, N_IN))
        self.last_h = np.zeros((0, N_HID))
        self.last_out = np.zeros((0, N_OUT))
        self.target_vec = np.zeros((0, 2)); self.target_seen = np.zeros(0, bool)
        self.danger_vec = np.zeros((0, 2)); self.danger_seen = np.zeros(0, bool)
        self.speed = np.zeros(0)

    def _empty(self):
        self.pos = np.zeros((0, 2)); self.heading = np.zeros(0)
        self.energy = np.zeros(0); self.age = np.zeros(0, int)
        self.gen = np.zeros(0, int); self.eaten = np.zeros(0, int)
        self.hue = np.zeros(0); self.uid = np.zeros(0, int)
        self.W1 = np.zeros((0, N_IN, N_HID)); self.b1 = np.zeros((0, N_HID))
        self.W2 = np.zeros((0, N_HID, N_OUT)); self.b2 = np.zeros((0, N_OUT))

    def __len__(self):
        return len(self.pos)

    def _append(self, **kw):
        n = len(kw["pos"])
        kw["uid"] = np.arange(self.next_uid, self.next_uid + n)
        self.next_uid += n
        for f in self.FIELDS:
            setattr(self, f, np.concatenate([getattr(self, f), kw[f]]))

    def add_random(self, n):
        if n <= 0:
            return
        self._append(
            pos=rng.random((n, 2)) * [W, H], heading=rng.random(n) * 2 * np.pi,
            energy=np.full(n, self.cfg["start_energy"]), age=np.zeros(n, int),
            gen=np.zeros(n, int), eaten=np.zeros(n, int),
            hue=(self.hue_center + rng.normal(0, 0.06, n)) % 1.0,
            W1=rng.normal(0, 0.9, (n, N_IN, N_HID)), b1=rng.normal(0, 0.3, (n, N_HID)),
            W2=rng.normal(0, 0.9, (n, N_HID, N_OUT)), b2=rng.normal(0, 0.3, (n, N_OUT)),
        )

    @staticmethod
    def _mutate(a, rate=0.12, scale=0.35):
        mask = rng.random(a.shape) < rate
        return a + mask * rng.normal(0, scale, a.shape)

    def add_children(self, parent_idx, energy, pos=None, from_archive=None):
        """parent_idx 개체를 복제 + 변이해서 자식 추가."""
        if from_archive is not None:
            src = from_archive
            n = len(src["W1"])
        else:
            n = len(parent_idx)
            if n == 0:
                return
            src = dict(W1=self.W1[parent_idx], b1=self.b1[parent_idx],
                       W2=self.W2[parent_idx], b2=self.b2[parent_idx],
                       hue=self.hue[parent_idx], gen=self.gen[parent_idx])
        if pos is None:
            pos = rng.random((n, 2)) * [W, H]
        self._append(
            pos=pos, heading=rng.random(n) * 2 * np.pi,
            energy=np.asarray(energy, float) * np.ones(n), age=np.zeros(n, int),
            gen=src["gen"] + 1, eaten=np.zeros(n, int),
            hue=(src["hue"] + rng.normal(0, 0.015, n)) % 1.0,
            W1=self._mutate(src["W1"]), b1=self._mutate(src["b1"]),
            W2=self._mutate(src["W2"]), b2=self._mutate(src["b2"]),
        )

    def remove(self, dead_mask):
        if not dead_mask.any():
            return
        # 잘했던 뇌는 기록 보관소에 저장 (생애 동안 먹은 속도 기준)
        idx = np.nonzero(dead_mask & (self.age > 300))[0]
        for i in idx:
            score = self.eaten[i] / self.age[i] * 1000
            self.archive.append((score, dict(W1=self.W1[i:i+1].copy(), b1=self.b1[i:i+1].copy(),
                                             W2=self.W2[i:i+1].copy(), b2=self.b2[i:i+1].copy(),
                                             hue=self.hue[i:i+1].copy(), gen=self.gen[i:i+1].copy())))
        self.archive.sort(key=lambda t: -t[0])
        del self.archive[20:]
        keep = ~dead_mask
        for f in self.FIELDS:
            setattr(self, f, getattr(self, f)[keep])
        for f in ["last_in", "last_h", "last_out", "target_vec", "target_seen",
                  "danger_vec", "danger_seen", "speed"]:
            a = getattr(self, f)
            if len(a) == len(keep):
                setattr(self, f, a[keep])

    # ---------------- 뇌 ----------------
    def think(self, target_vec, target_d, danger_vec, danger_d):
        c, s = np.cos(self.heading), np.sin(self.heading)
        sight = self.cfg["sight"]

        def body_frame(vec, d):
            seen = d < sight
            fwd = (vec[:, 0] * c + vec[:, 1] * s) / sight
            side = (-vec[:, 0] * s + vec[:, 1] * c) / sight
            return np.where(seen, fwd, 0.0), np.where(seen, side, 0.0), seen.astype(float)

        tf, ts, tseen = body_frame(target_vec, target_d)
        df, ds, dseen = body_frame(danger_vec, danger_d)
        x = np.stack([tf, ts, tseen, df, ds, dseen,
                      self.energy / self.cfg["max_energy"]], axis=1)
        h = np.tanh(np.einsum("ni,nih->nh", x, self.W1) + self.b1)
        o = np.einsum("nh,nho->no", h, self.W2) + self.b2
        turn = np.tanh(o[:, 0])
        speed = 1.0 / (1.0 + np.exp(-o[:, 1]))

        self.last_in, self.last_h = x, h
        self.last_out = np.stack([turn, speed], axis=1)

        # 똑똑함 지표: 목표가 보일 때 그쪽으로 회전하는 비율 (랜덤 뇌면 약 50%)
        m = (tseen > 0) & (np.abs(ts) > 0.08)
        if m.sum() >= 3:
            acc = float(np.mean(np.sign(turn[m]) == np.sign(ts[m])))
            self.chase_acc = 0.995 * self.chase_acc + 0.005 * acc
        self.target_vec, self.target_seen = target_vec, tseen > 0
        self.danger_vec, self.danger_seen = danger_vec, dseen > 0
        return turn, speed

    def move(self, turn, speed01):
        cfg = self.cfg
        self.heading = (self.heading + turn * cfg["turn_rate"]) % (2 * np.pi)
        sp = speed01 * cfg["max_speed"]
        self.speed = sp
        self.pos += np.stack([np.cos(self.heading), np.sin(self.heading)], axis=1) * sp[:, None]
        self.pos %= [W, H]
        self.energy -= cfg["cost_base"] + cfg["cost_move"] * speed01 ** 2
        self.age += 1

    def color(self, i, light=0.55):
        return hsl_to_rgb(self.hue[i], 0.75, light)

    def eat_rate(self):
        """살아있는 개체들의 '1000스텝당 먹은 횟수' 평균 (충분히 산 개체만)."""
        m = self.age > 250
        if not m.any():
            return None
        return float(np.mean(self.eaten[m] / self.age[m] * 1000))


def hsl_to_rgb(h, s, l):
    def f(n):
        k = (n + h * 12) % 12
        a = s * min(l, 1 - l)
        return l - a * max(-1, min(k - 3, 9 - k, 1))
    return int(f(0) * 255), int(f(8) * 255), int(f(4) * 255)


# ======================================================================
# 세계
# ======================================================================
class World:
    def __init__(self):
        self.prey = Species(PREY, hue_center=0.33)   # 초록
        self.pred = Species(PRED, hue_center=0.99)   # 빨강
        self.food = rng.random((FOOD_N, 2)) * [W, H]
        self.predators_on = True
        self.step_n = 0
        self.history = []        # (step, 초식 먹는속도, 포식자 사냥속도, 초식 수, 포식자 수)

    def toggle_predators(self):
        self.predators_on = not self.predators_on
        if not self.predators_on:
            self.pred.remove(np.ones(len(self.pred), bool))
        else:
            self.pred.add_random(PRED["start_n"])

    def _repopulate(self, sp):
        cfg = sp.cfg
        need = cfg["min_n"] - len(sp)
        if need <= 0:
            return
        if sp.archive and rng.random() < 0.7:
            pick = [sp.archive[rng.integers(len(sp.archive))][1] for _ in range(need)]
            src = {k: np.concatenate([p[k] for p in pick]) for k in pick[0]}
            sp.add_children(None, cfg["start_energy"], from_archive=src)
        else:
            sp.add_random(need)

    def step(self):
        prey, pred = self.prey, self.pred
        self.step_n += 1

        # 1) 보기 + 생각
        fv, fd, _ = nearest(prey.pos, self.food)
        if self.predators_on:
            dv, dd, _ = nearest(prey.pos, pred.pos)
        else:
            dv, dd = np.zeros((len(prey), 2)), np.full(len(prey), np.inf)
        prey_turn, prey_speed = prey.think(fv, fd, dv, dd)

        if self.predators_on and len(pred):
            pv, pdist, _ = nearest(pred.pos, prey.pos)
            ov, od, _ = nearest(pred.pos, pred.pos, exclude_self=True)
            pred_turn, pred_speed = pred.think(pv, pdist, ov, od)

        # 2) 움직이기
        prey.move(prey_turn, prey_speed)
        if self.predators_on and len(pred):
            pred.move(pred_turn, pred_speed)

        # 3) 초식이 먹이 먹기
        if len(prey):
            _, d, who = nearest(self.food, prey.pos)
            got = d < (prey.cfg["radius"] + 4)
            if got.any():
                np.add.at(prey.energy, who[got], FOOD_ENERGY)
                np.add.at(prey.eaten, who[got], 1)
                self.food[got] = rng.random((got.sum(), 2)) * [W, H]

        # 4) 포식자가 사냥
        caught = np.zeros(len(prey), bool)
        if self.predators_on and len(pred) and len(prey):
            _, d, who = nearest(prey.pos, pred.pos)
            caught = d < (pred.cfg["radius"] + prey.cfg["radius"] - 2)
            if caught.any():
                np.add.at(pred.energy, who[caught], PREY_ENERGY_FOR_PRED)
                np.add.at(pred.eaten, who[caught], 1)

        # 5) 번식 + 죽음
        for sp, extra_dead in [(prey, caught), (pred, None)]:
            if len(sp) == 0:
                continue
            cfg = sp.cfg
            np.minimum(sp.energy, cfg["max_energy"], out=sp.energy)
            dead = (sp.energy <= 0) | (sp.age > cfg["max_age"])
            if extra_dead is not None:
                dead |= extra_dead
            parents = np.nonzero((sp.energy >= cfg["repro_energy"]) & ~dead)[0]
            room = cfg["max_n"] - (len(sp) - dead.sum())
            if len(parents) > room:
                # 자리가 꽉 찼으면 가장 굶주린 개체가 밀려남 -> 꽉 찬 상태에서도 경쟁(진화)이 계속됨
                cand = np.nonzero(~dead & (sp.energy < cfg["repro_energy"]))[0]
                n_out = min(len(parents) - max(room, 0), len(cand))
                if n_out > 0:
                    weakest = cand[np.argsort(sp.energy[cand])[:n_out]]
                    dead[weakest] = True
                    room += n_out
            parents = parents[:max(room, 0)]
            snapshot = None
            if len(parents):
                # 부모 에너지를 반으로 나눠 자식에게 줌. 뇌는 복사 후 변이
                half = sp.energy[parents] / 2
                sp.energy[parents] = half
                child_pos = (sp.pos[parents] + rng.normal(0, 8, (len(parents), 2))) % [W, H]
                snapshot = dict(W1=sp.W1[parents].copy(), b1=sp.b1[parents].copy(),
                                W2=sp.W2[parents].copy(), b2=sp.b2[parents].copy(),
                                hue=sp.hue[parents].copy(), gen=sp.gen[parents].copy())
            sp.remove(dead)
            if snapshot is not None:
                sp.add_children(None, half, pos=child_pos, from_archive=snapshot)

        # 6) 최소 개체수 유지 (멸종 방지)
        self._repopulate(prey)
        if self.predators_on:
            self._repopulate(pred)

        if self.step_n % 200 == 0:
            self.history.append((self.step_n, prey.chase_acc,
                                 pred.chase_acc if self.predators_on else None,
                                 len(prey), len(pred)))
            del self.history[:-300]


# ======================================================================
# 화면 (pyglet)
# ======================================================================
SPEEDS = [1, 4, 16, "MAX"]
FONT = ["Apple SD Gothic Neo", "AppleGothic", "Noto Sans CJK KR", "Malgun Gothic", "Arial"]


class Pool:
    """도형을 매 프레임 새로 만들지 않고 재사용."""
    def __init__(self, factory):
        self.factory, self.items, self.used = factory, [], 0

    def begin(self):
        self.used = 0

    def get(self):
        if self.used == len(self.items):
            self.items.append(self.factory())
        s = self.items[self.used]
        s.visible = True
        self.used += 1
        return s

    def end(self):
        for s in self.items[self.used:]:
            s.visible = False


class App:
    def __init__(self, window, headless=False):
        import pyglet
        from pyglet import shapes
        self.pyglet, self.shapes = pyglet, shapes
        self.win = window
        self.world = World()
        self.speed_i = 0
        self.paused = False
        self.show_hud = True
        self.sel = None          # (종, uid)
        self.sel_msg = ""

        B = self.batch = pyglet.graphics.Batch()
        g0, g1, g2, g3 = (pyglet.graphics.Group(order=i) for i in range(4))
        S = shapes

        self.food_pool = Pool(lambda: S.Circle(0, 0, 2.6, segments=8, color=(235, 205, 90, 255), batch=B, group=g0))
        self.body_pool = Pool(lambda: S.Triangle(0, 0, 0, 0, 0, 0, color=(255, 255, 255, 255), batch=B, group=g1))
        self.line_pool = Pool(lambda: S.Line(0, 0, 0, 0, thickness=1.5, batch=B, group=g1))
        self.arc_pool = Pool(lambda: S.Arc(0, 0, 10, segments=72, thickness=1.5, batch=B, group=g1))

        # HUD
        self.hud_bg = S.Rectangle(10, H - 210, 352, 200, color=(10, 12, 20, 200), batch=B, group=g2)
        self.hud_text = pyglet.text.Label("", font_name=FONT, font_size=12, x=22, y=H - 24,
                                          multiline=True, width=310, anchor_y="top",
                                          color=(225, 230, 240, 255), batch=B, group=g3)
        self.graph_pool = Pool(lambda: S.Line(0, 0, 0, 0, thickness=1.6, batch=B, group=g3))
        self.graph_label = pyglet.text.Label("먹이 쪽으로 도는 비율 (랜덤 50%)", font_name=FONT, font_size=10,
                                             x=22, y=H - 122, color=(160, 165, 180, 255), batch=B, group=g3)
        # 그래프 눈금 (그래프 y범위: 40%~100%, 위치는 draw_hud 의 gx,gy,gw,gh 와 맞춤)
        self.graph_ticks = [
            pyglet.text.Label(txt, font_name=FONT, font_size=9, x=326, y=H - 200 + 70 * (v - 0.4) / 0.6,
                              anchor_y="center", color=(130, 135, 150, 255), batch=B, group=g3)
            for txt, v in [("100%", 1.0), ("50%", 0.5)]
        ]
        self.help = pyglet.text.Label("클릭: 뇌 보기   F: 배속   P: 포식자   Space: 정지   H: 정보창   R: 리셋",
                                      font_name=FONT, font_size=11, x=12, y=10,
                                      color=(150, 155, 170, 255), batch=B, group=g3)

        # 뇌 패널
        self.bx, self.by, self.bw, self.bh = W - 420, 12, 408, 330
        self.brain_bg = S.Rectangle(self.bx, self.by, self.bw, self.bh, color=(10, 12, 20, 215), batch=B, group=g2)
        self.brain_title = pyglet.text.Label("", font_name=FONT, font_size=12, x=self.bx + 12,
                                             y=self.by + self.bh - 12, anchor_y="top", multiline=True,
                                             width=self.bw - 24, color=(230, 235, 245, 255), batch=B, group=g3)
        self.edge_pool = Pool(lambda: S.Line(0, 0, 0, 0, thickness=1.0, batch=B, group=g2))
        self.node_pool = Pool(lambda: S.Circle(0, 0, 7, segments=16, batch=B, group=g3))
        self.node_labels = []
        for name in INPUT_NAMES:
            self.node_labels.append(pyglet.text.Label(name, font_name=FONT, font_size=10, anchor_x="right",
                                                      anchor_y="center", color=(170, 175, 190, 255), batch=B, group=g3))
        for name in OUTPUT_NAMES:
            self.node_labels.append(pyglet.text.Label(name, font_name=FONT, font_size=10, anchor_x="left",
                                                      anchor_y="center", color=(170, 175, 190, 255), batch=B, group=g3))

        if not headless:
            pyglet.clock.schedule_interval(self.update, 1 / 60)
            window.push_handlers(self)

    # ---------------- 시뮬레이션 진행 ----------------
    def update(self, dt):
        if self.paused:
            return
        sp = SPEEDS[self.speed_i]
        if sp == "MAX":
            import time
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < 0.025:
                self.world.step()
        else:
            for _ in range(sp):
                self.world.step()

    # ---------------- 입력 ----------------
    def on_key_press(self, sym, mod):
        k = self.pyglet.window.key
        if sym == k.F:
            self.speed_i = (self.speed_i + 1) % len(SPEEDS)
        elif sym == k.SPACE:
            self.paused = not self.paused
        elif sym == k.P:
            self.world.toggle_predators()
        elif sym == k.H:
            self.show_hud = not self.show_hud
        elif sym == k.R:
            self.world = World(); self.sel = None

    def on_mouse_press(self, x, y, button, mod):
        if self.sel and self.bx <= x <= self.bx + self.bw and self.by <= y <= self.by + self.bh:
            return
        best, bd = None, 30.0
        for sp in (self.world.prey, self.world.pred):
            if len(sp) == 0:
                continue
            d = wrap_delta(sp.pos - [x, y])
            dist = np.hypot(d[:, 0], d[:, 1])
            i = int(np.argmin(dist))
            if dist[i] < bd:
                best, bd = (sp, int(sp.uid[i])), dist[i]
        self.sel = best
        self.sel_msg = ""

    # ---------------- 그리기 ----------------
    def on_draw(self):
        self.win.clear()
        self.draw_world()
        self.draw_hud()
        self.draw_brain()
        self.batch.draw()

    def draw_world(self):
        w = self.world
        self.food_pool.begin()
        for fx, fy in w.food:
            c = self.food_pool.get(); c.x, c.y = fx, fy
        self.food_pool.end()

        self.body_pool.begin()
        for sp in (w.prey, w.pred):
            r = sp.cfg["radius"]
            c, s = np.cos(sp.heading), np.sin(sp.heading)
            px, py = sp.pos[:, 0], sp.pos[:, 1]
            tip = np.stack([px + c * r * 1.7, py + s * r * 1.7], 1)
            bl = np.stack([px - c * r + -s * r * 0.8, py - s * r + c * r * 0.8], 1)
            br = np.stack([px - c * r - -s * r * 0.8, py - s * r - c * r * 0.8], 1)
            # 배고플수록 어둡게
            light = 0.42 + 0.22 * np.clip(sp.energy / sp.cfg["max_energy"], 0, 1)
            for i in range(len(sp)):
                t = self.body_pool.get()
                t.x, t.y = tip[i]; t.x2, t.y2 = bl[i]; t.x3, t.y3 = br[i]
                t.color = (*sp.color(i, light[i]), 255)
        self.body_pool.end()

        # 선택된 생명체: 시야, 보고 있는 목표/위험
        self.line_pool.begin(); self.arc_pool.begin()
        idx = self._sel_index()
        if idx is not None:
            sp, i = idx
            x, y = sp.pos[i]
            a = self.arc_pool.get(); a.x, a.y, a.radius = x, y, sp.cfg["sight"]; a.color = (255, 255, 255, 35)
            a = self.arc_pool.get(); a.x, a.y, a.radius = x, y, sp.cfg["radius"] * 2.4; a.color = (255, 255, 255, 220)
            if i < len(sp.target_seen):
                if sp.target_seen[i]:
                    l = self.line_pool.get(); l.x, l.y = x, y
                    l.x2, l.y2 = x + sp.target_vec[i, 0], y + sp.target_vec[i, 1]
                    l.color = (235, 205, 90, 200) if sp is w.prey else (120, 230, 140, 200)
                if sp.danger_seen[i]:
                    l = self.line_pool.get(); l.x, l.y = x, y
                    l.x2, l.y2 = x + sp.danger_vec[i, 0], y + sp.danger_vec[i, 1]
                    l.color = (255, 80, 80, 200)
        self.line_pool.end(); self.arc_pool.end()

    def draw_hud(self):
        w = self.world
        for o in (self.hud_bg, self.hud_text, self.graph_label, self.help, *self.graph_ticks):
            o.visible = self.show_hud
        self.graph_pool.begin()
        if not self.show_hud:
            self.graph_pool.end(); return
        sp = SPEEDS[self.speed_i]
        p, q = w.prey, w.pred
        lines = [
            f"스텝 {w.step_n:,}   배속 {'MAX' if sp == 'MAX' else 'x' + str(sp)}" + ("   [일시정지]" if self.paused else ""),
            f"초식  {len(p):3d}마리   최고 세대 {p.gen.max() if len(p) else 0}   추적 {p.chase_acc * 100:4.1f}%",
        ]
        if w.predators_on:
            lines.append(f"포식자 {len(q):3d}마리   최고 세대 {q.gen.max() if len(q) else 0}   추적 {q.chase_acc * 100:4.1f}%")
        else:
            lines.append("포식자 없음 (P로 등장)")
        self.hud_text.text = "\n".join(lines)

        # 그래프: 시간에 따른 '목표 쪽으로 도는 비율'
        gx, gy, gw, gh = 22, H - 200, 300, 70
        def gpt(j, n, v):
            return gx + gw * j / max(n - 1, 1), gy + gh * (v - 0.4) / 0.6
        base = self.graph_pool.get(); base.x, base.y = gpt(0, 2, 0.5); base.x2, base.y2 = gpt(1, 2, 0.5)
        base.color = (120, 120, 140, 120)
        hist = w.history[-150:]
        for col_i, color in [(1, (110, 220, 120, 255)), (2, (240, 90, 80, 255))]:
            vals = [h[col_i] for h in hist]
            for j in range(1, len(vals)):
                if vals[j - 1] is None or vals[j] is None:
                    continue
                l = self.graph_pool.get()
                l.x, l.y = gpt(j - 1, len(vals), min(max(vals[j - 1], 0.4), 1.0))
                l.x2, l.y2 = gpt(j, len(vals), min(max(vals[j], 0.4), 1.0))
                l.color = color
        self.graph_pool.end()

    def _sel_index(self):
        if not self.sel:
            return None
        sp, uid = self.sel
        hit = np.nonzero(sp.uid == uid)[0]
        if len(hit) == 0:
            self.sel = None
            self.sel_msg = "선택했던 녀석이 죽었습니다..."
            return None
        return sp, int(hit[0])

    def draw_brain(self):
        idx = self._sel_index()
        show = idx is not None or bool(self.sel_msg)
        self.brain_bg.visible = self.brain_title.visible = show
        self.edge_pool.begin(); self.node_pool.begin()
        for lb in self.node_labels:
            lb.visible = idx is not None
        if idx is None:
            self.brain_title.text = self.sel_msg
            self.edge_pool.end(); self.node_pool.end()
            return
        sp, i = idx
        kind = sp.cfg["name"]
        self.brain_title.text = (f"{kind} #{sp.uid[i]}   {sp.gen[i]}세대   나이 {sp.age[i]}\n"
                                 f"에너지 {sp.energy[i]:.0f}   먹은 횟수 {sp.eaten[i]}")

        has = i < len(sp.last_in)
        x_in = sp.last_in[i] if has else np.zeros(N_IN)
        h = sp.last_h[i] if has else np.zeros(N_HID)
        out = sp.last_out[i] if has else np.zeros(N_OUT)
        out_disp = np.array([out[0], out[1] * 2 - 1])   # 속도 0~1 -> -1~1 로 표시

        top, bottom = self.by + self.bh - 62, self.by + 18
        def col_y(n, j):
            return top - (top - bottom) * (j + 0.5) / n
        X = [self.bx + 110, self.bx + 215, self.bx + 320]
        P_in = [(X[0], col_y(N_IN, j)) for j in range(N_IN)]
        P_h = [(X[1], col_y(N_HID, j)) for j in range(N_HID)]
        P_o = [(X[2], col_y(N_OUT * 3, j * 3 + 1)) for j in range(N_OUT)]

        def edges(P_a, P_b, Wm, act_a):
            for a in range(len(P_a)):
                for b in range(len(P_b)):
                    wv = Wm[a, b]
                    sig = abs(wv * act_a[a])
                    l = self.edge_pool.get()
                    l.x, l.y = P_a[a]; l.x2, l.y2 = P_b[b]
                    l.thickness = 0.6 + min(abs(wv), 2.5) * 0.9
                    alpha = int(np.clip(20 + sig * 140, 20, 255))
                    l.color = (90, 170, 255, alpha) if wv > 0 else (255, 100, 90, alpha)

        edges(P_in, P_h, sp.W1[i], x_in)
        edges(P_h, P_o, sp.W2[i], h)

        def node(p, a):
            c = self.node_pool.get(); c.x, c.y = p
            a = float(np.clip(a, -1, 1))
            if a >= 0:
                c.color = (int(60 + 30 * a), int(70 + 110 * a), int(90 + 165 * a), 255)
            else:
                c.color = (int(90 - 165 * a), int(70 + 30 * a), int(80 + 10 * a), 255)
        for p, a in zip(P_in, x_in):
            node(p, a)
        for p, a in zip(P_h, h):
            node(p, a)
        for p, a in zip(P_o, out_disp):
            node(p, a)

        for j, lb in enumerate(self.node_labels[:N_IN]):
            lb.x, lb.y = P_in[j][0] - 12, P_in[j][1]
        vals = [f"{out[0]:+.2f}", f"{out[1]:.2f}"]
        for j, lb in enumerate(self.node_labels[N_IN:]):
            lb.x, lb.y = P_o[j][0] + 12, P_o[j][1]
            lb.text = f"{OUTPUT_NAMES[j]} {vals[j]}"
        self.edge_pool.end(); self.node_pool.end()


def set_world_size(w, h):
    """세계 크기를 바꾸고, 먹이/개체 수를 넓이에 비례해서 조정 (밀도 유지)."""
    global W, H, FOOD_N
    k = (w * h) / (1280 * 760)
    W, H = int(w), int(h)
    FOOD_N = int(150 * k)
    PREY.update(start_n=int(80 * k), min_n=int(25 * k), max_n=int(170 * k))
    PRED.update(start_n=max(4, int(8 * k)), min_n=max(3, int(4 * k)), max_n=int(30 * k))


def main():
    import sys
    import pyglet
    screen = pyglet.display.get_display().get_default_screen()
    if "--small" in sys.argv:
        set_world_size(1280, 760)
        win = pyglet.window.Window(W, H, caption="진화하는 AI 생명체", vsync=True)
    elif "--window" in sys.argv:
        # 메뉴바/독/제목줄 공간만 빼고 화면을 꽉 채우는 창
        set_world_size(screen.width, screen.height - 110)
        win = pyglet.window.Window(W, H, caption="진화하는 AI 생명체", vsync=True)
        win.set_location(0, 30)
    else:
        win = pyglet.window.Window(caption="진화하는 AI 생명체", fullscreen=True, vsync=True)
        set_world_size(win.width, win.height)
    print(f"세계 크기 {W}x{H}, 먹이 {FOOD_N}개")
    pyglet.gl.glClearColor(0.045, 0.05, 0.08, 1.0)
    App(win)

    @win.event
    def on_key_press(sym, mod):
        if sym == pyglet.window.key.ESCAPE:
            win.close()

    pyglet.app.run()


if __name__ == "__main__":
    main()
