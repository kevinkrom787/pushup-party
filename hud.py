"""
Game-style overlay for the pushup counter: a rep counter that pops on every
rep, and a persistent footer showing the latest milestone unlocked plus a
progress track through all of them. Hitting a milestone (5, 10, 25, 35 reps
per session) also flashes the screen and fires confetti.

Pure OpenCV drawing (no emoji support in cv2.putText, so everything is
shapes and text). Colors are BGR.
"""

import math
import random
import time
from dataclasses import dataclass

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_DUPLEX

# rep count -> (title, subtitle, color)
MILESTONES = {
    5: ("WARMED UP!", "The couch is already jealous", (80, 220, 80)),
    10: ("DOUBLE DIGITS!", "Your arms have entered the chat", (0, 140, 255)),
    25: ("BEAST MODE", "Gravity is filing a complaint", (200, 60, 255)),
    35: ("LEGENDARY", "Somebody call the Olympics", (0, 215, 255)),
}

CONFETTI_COLORS = [
    (255, 80, 80), (80, 220, 80), (60, 60, 255),
    (0, 215, 255), (255, 80, 255), (255, 255, 80),
]
GRAVITY = 900.0  # px/s^2
CONFETTI_SEC = 3.5
REP_POP_SEC = 0.35
PLUS_ONE_SEC = 0.8

PANEL_X, PANEL_Y, PANEL_W, PANEL_H = 16, 16, 300, 150
FOOTER_FRAC = 0.2  # footer height as a fraction of frame height
FLASH_SEC = 0.25
TITLE_POP_SEC = 0.4


@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    angle: float
    spin: float
    size: float
    color: tuple
    born: float
    life: float
    gravity: float = 1.0


def _text(img, text, org, scale, color, thickness):
    """Text with a dark outline so it reads on any background.

    The outline is the same text shifted in 8 directions rather than a thicker
    stroke: Hershey glyph spacing grows with thickness, so a thicker black pass
    drifts out of alignment with the colored one.
    """
    d = max(2, thickness)
    x, y = org
    for dx in (-d, 0, d):
        for dy in (-d, 0, d):
            if dx or dy:
                cv2.putText(img, text, (x + dx, y + dy), FONT, scale, (0, 0, 0), thickness, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, thickness, cv2.LINE_AA)


def _text_width(text, scale, thickness):
    (w, _), _ = cv2.getTextSize(text, FONT, scale, thickness)
    return w


def _ease_out_back(s):
    """0 -> 1 with a little overshoot, for a springy pop-in."""
    c1 = 1.70158
    return 1 + (c1 + 1) * (s - 1) ** 3 + c1 * (s - 1) ** 2


class Hud:
    def __init__(self):
        self.reset()
        self.frame_size = (720, 1280)  # (h, w), updated on every draw
        self.last_draw = time.monotonic()

    def reset(self):
        self.particles = []
        self.rep_time = -math.inf
        self.plus_ones = []
        self.milestone = None  # latest milestone reached this session
        self.milestone_time = -math.inf

    def on_rep(self, count):
        now = time.monotonic()
        self.rep_time = now
        self.plus_ones.append(now)
        cx, cy = PANEL_X + PANEL_W + 35, PANEL_Y + 80  # beside the counter, where "+1" floats
        self._burst(cx, cy, n=20, speed=(150, 350), life=0.8, now=now)

        if count in MILESTONES:
            self.milestone, self.milestone_time = count, now
            h, w = self.frame_size
            self._burst(w / 2, h * (1 - FOOTER_FRAC), n=120, speed=(300, 800), life=2.5, now=now)
            for _ in range(150):  # confetti rain from the top edge
                self.particles.append(Particle(
                    x=random.uniform(0, w), y=random.uniform(-h * 0.4, 0),
                    vx=random.uniform(-80, 80), vy=random.uniform(50, 250),
                    angle=random.uniform(0, math.pi), spin=random.uniform(-10, 10),
                    size=random.uniform(8, 16), color=random.choice(CONFETTI_COLORS),
                    born=now, life=CONFETTI_SEC, gravity=0.25,  # rain drifts down slowly
                ))

    def _burst(self, x, y, n, speed, life, now):
        for _ in range(n):
            a = random.uniform(-math.pi, 0)  # upward half-circle
            v = random.uniform(*speed)
            self.particles.append(Particle(
                x=x, y=y, vx=v * math.cos(a), vy=v * math.sin(a),
                angle=random.uniform(0, math.pi), spin=random.uniform(-12, 12),
                size=random.uniform(6, 14), color=random.choice(CONFETTI_COLORS),
                born=now, life=life,
            ))

    def draw(self, frame, count):
        now = time.monotonic()
        dt = min(now - self.last_draw, 0.1)
        self.last_draw = now
        self.frame_size = frame.shape[:2]

        frame = self._draw_flash(frame, now)
        frame = self._draw_footer(frame, count, now)
        frame = self._draw_panel(frame, count, now)
        self._draw_plus_ones(frame, now)
        self._draw_particles(frame, now, dt)
        return frame

    def _draw_panel(self, frame, count, now):
        overlay = frame.copy()
        cv2.rectangle(overlay, (PANEL_X, PANEL_Y),
                      (PANEL_X + PANEL_W, PANEL_Y + PANEL_H), (30, 30, 30), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        _text(frame, "PUSHUPS", (PANEL_X + 16, PANEL_Y + 32), 0.7, (200, 200, 200), 1)

        # Number pops bigger and flashes green right after a rep.
        pop = max(0.0, 1 - (now - self.rep_time) / REP_POP_SEC)
        scale = 3.0 + 1.2 * pop
        color = (80, 255, 80) if pop > 0 else (255, 255, 255)
        num = str(count)
        x = PANEL_X + (PANEL_W - _text_width(num, scale, 6)) // 2
        _text(frame, num, (x, PANEL_Y + 125), scale, color, 6)
        return frame

    def _draw_plus_ones(self, frame, now):
        self.plus_ones = [t for t in self.plus_ones if now - t < PLUS_ONE_SEC]
        for t in self.plus_ones:
            age = (now - t) / PLUS_ONE_SEC
            y = int(PANEL_Y + 100 - 70 * age)
            _text(frame, "+1", (PANEL_X + PANEL_W + 12, y), 1.4, (80, 255, 80), 3)

    def _draw_particles(self, frame, now, dt):
        h, w = frame.shape[:2]
        alive = []
        for p in self.particles:
            p.vy += GRAVITY * p.gravity * dt
            p.x += p.vx * dt
            p.y += p.vy * dt
            p.angle += p.spin * dt
            if now - p.born > p.life or p.y > h + 20:
                continue
            alive.append(p)
            c, s = math.cos(p.angle), math.sin(p.angle)
            hw, hh = p.size / 2, p.size / 4
            corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
            pts = np.array([(p.x + dx * c - dy * s, p.y + dx * s + dy * c)
                            for dx, dy in corners], dtype=np.int32)
            cv2.fillPoly(frame, [pts], p.color, cv2.LINE_AA)
        self.particles = alive

    def _draw_flash(self, frame, now):
        """Quick full-screen tint in the milestone's color right after it's hit."""
        t = now - self.milestone_time
        if self.milestone is None or t >= FLASH_SEC:
            return frame
        a = 0.5 * (1 - t / FLASH_SEC)
        tint = np.full_like(frame, MILESTONES[self.milestone][2])
        return cv2.addWeighted(tint, a, frame, 1 - a, 0)

    def _draw_footer(self, frame, count, now):
        """Always-on footer: latest milestone on the left, milestone track on the right."""
        h, w = frame.shape[:2]
        u = w / 1280  # scale everything off a 1280px-wide reference layout
        y0 = h - int(h * FOOTER_FRAC)
        mid_y = (y0 + h) // 2

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, y0), (w, h), (20, 20, 20), -1)
        frame = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)
        accent = MILESTONES[self.milestone][2] if self.milestone else (120, 120, 120)
        cv2.rectangle(frame, (0, y0), (w, y0 + max(4, int(6 * u))), accent, -1)

        # Left: what you've unlocked (or a nudge toward the first milestone).
        x = int(28 * u)
        if self.milestone:
            title, subtitle, color = MILESTONES[self.milestone]
            kicker = f"MILESTONE UNLOCKED  |  {self.milestone} PUSHUPS"
            pop = _ease_out_back(min(1.0, (now - self.milestone_time) / TITLE_POP_SEC))
        else:
            first = min(MILESTONES)
            title, subtitle, color = "LET'S GO!", f"First milestone at {first} pushups", (230, 230, 230)
            kicker = "PUSHUP PARTY"
            pop = 1.0
        _text(frame, kicker, (x, mid_y - int(34 * u)), 0.55 * u, (180, 180, 180), 1)
        _text(frame, title, (x, mid_y + int(18 * u)), max(0.1, 1.6 * u * pop), color, max(2, int(4 * u)))
        _text(frame, subtitle, (x, mid_y + int(50 * u)), 0.7 * u, (255, 255, 255), max(1, int(2 * u)))

        # Right: a track from 0 through every milestone, filled up to the current count.
        marks = sorted(MILESTONES)
        track_x0, track_x1 = int(w * 0.56), w - int(60 * u)
        xs = np.linspace(track_x0, track_x1, len(marks) + 1)
        stops = [0] + marks
        ty = mid_y - int(8 * u)
        r = int(24 * u)
        cv2.line(frame, (int(xs[0]), ty), (int(xs[-1]), ty), (70, 70, 70), max(4, int(8 * u)), cv2.LINE_AA)
        fill_x = int(np.interp(count, stops, xs))
        if fill_x > xs[0]:
            cv2.line(frame, (int(xs[0]), ty), (fill_x, ty), accent, max(4, int(8 * u)), cv2.LINE_AA)
        cv2.circle(frame, (int(xs[0]), ty), int(8 * u), accent, -1, cv2.LINE_AA)

        for m, mx in zip(marks, xs[1:]):
            mx = int(mx)
            name, _, m_color = MILESTONES[m]
            reached = count >= m
            if reached:
                cv2.circle(frame, (mx, ty), r, m_color, -1, cv2.LINE_AA)
            else:
                cv2.circle(frame, (mx, ty), r, (40, 40, 40), -1, cv2.LINE_AA)
                cv2.circle(frame, (mx, ty), r, (110, 110, 110), max(2, int(3 * u)), cv2.LINE_AA)
            num = str(m)
            ns = 0.75 * u
            num_color = (20, 20, 20) if reached else (160, 160, 160)
            (tw, th), _ = cv2.getTextSize(num, FONT, ns, 2)
            cv2.putText(frame, num, (mx - tw // 2, ty + th // 2), FONT, ns, num_color, 2, cv2.LINE_AA)
            ls = 0.42 * u
            label_color = m_color if reached else (130, 130, 130)
            cv2.putText(frame, name, (mx - _text_width(name, ls, 1) // 2, ty + r + int(26 * u)),
                        FONT, ls, label_color, 1, cv2.LINE_AA)
        return frame
