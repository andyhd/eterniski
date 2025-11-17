"""
Eterniski: An endless runner ski slalom game
"""
import random
from bisect import bisect
from collections import deque
from collections.abc import Generator, Iterable, Iterator
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import NamedTuple

import pygame as pg

from ogam import assets, controls
from ogam.controls import Action
from ogam.menu import MenuOption, menu
from ogam.scene import Scene, State, scene_manager

DEADZONE = 0.2


def inc(v: float) -> bool:
    return v > 0


def dec(v: float) -> bool:
    return v < 0


ASSETS_DIR = Path(__file__).parent / "assets"
DEFAULT_CONTROLS = {
    "left": [pg.Event(pg.KEYDOWN, key=pg.K_LEFT)],
    "left_released": [pg.Event(pg.KEYUP, key=pg.K_LEFT)],
    "left_stick": [pg.Event(pg.JOYAXISMOTION, axis=0, value=dec)],
    "right": [pg.Event(pg.KEYDOWN, key=pg.K_RIGHT)],
    "right_released": [pg.Event(pg.KEYUP, key=pg.K_RIGHT)],
    "right_stick": [pg.Event(pg.JOYAXISMOTION, axis=0, value=inc)],
    "forward": [pg.Event(pg.KEYDOWN, key=pg.K_UP)],
    "forward_stick": [pg.Event(pg.JOYAXISMOTION, axis=1, value=dec)],
}
HEIGHT = 800
SECTION_HEIGHT = 80
WIDTH = 600

font = assets.manager(ASSETS_DIR / "fonts")
images = assets.manager(ASSETS_DIR / "images")


def main_menu() -> Scene:
    title = font("arco", 64).render("Eterniski", True, "black")

    def render_option(name: str):
        def render(surface: pg.Surface, rect: pg.Rect, focused: bool) -> None:
            if focused:
                pg.draw.rect(surface, "cyan", rect)
            label = font("arco", 32).render(name, True, "black")
            surface.blit(label, label.get_rect(center=rect.center))
        return render

    main_menu = menu(
        [
            MenuOption(render_option("Start Game"), play),
            MenuOption(render_option("Options"), configure_controls)
        ],
        rect=pg.Rect(100, 400, 400, 55),
    )

    def _scene(screen: pg.Surface, events: list[pg.Event], state: State) -> Scene | None:
        if not getattr(state, "controls", None):
            state.controls = dict(DEFAULT_CONTROLS)

        screen.fill("white")
        screen.blit(title, title.get_rect(center=(WIDTH // 2, HEIGHT // 3)))

        return main_menu(screen, events)

    return _scene


def configure_controls() -> Scene:
    control = State(selected=None)
    title = font("arco", 48).render("Keyboard Controls", True, "black")
    fields = {k: v for k, v in DEFAULT_CONTROLS.items() if k in ("left", "right", "forward")}

    def render_option(name: str):
        def render(surface: pg.Surface, rect: pg.Rect, focused: bool) -> None:
            if focused:
                pg.draw.rect(surface, "cyan", rect)
            value = ", ".join(pg.key.name(_.key) for _ in fields[name] if _.type == pg.KEYDOWN)
            surface.blit(font("arco", 24).render(name, True, "black"), rect)
            surface.blit(font("arco", 24).render(value, True, "black"), (rect.x + 300, rect.y))
        return render

    controls_menu = menu(
        [
            MenuOption(render_option(name), partial(setattr, control, "selected", name))
            for name in fields
        ],
        rect=pg.Rect(50, 200, WIDTH - 100, 30),
    )

    def _scene(screen: pg.Surface, events: list[pg.Event], state: State) -> Scene | None:
        if control.selected:
            if input_ := next((event for event in events if event.type == pg.KEYDOWN), None):
                for template_event in state.controls[control.selected]:
                    if template_event.type == pg.KEYDOWN:
                        template_event.key = input_.key
                for template_event in state.controls.get(f"{control.selected}_released", []):
                    if template_event.type == pg.KEYUP:
                        template_event.key = input_.key
                fields.update(state.controls)
                control.selected = None
                return

        elif any(event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE for event in events):
            return main_menu()

        screen.fill("white")
        screen.blit(title, title.get_rect(center=(screen.width // 2, 50)))

        controls_menu(screen, events)

    return _scene


class WorldSection(NamedTuple):
    mobs: pg.sprite.Group
    rect: pg.FRect


@dataclass
class Player:
    image: pg.Surface
    rect: pg.FRect
    velocity: pg.Vector2 = field(default_factory=pg.Vector2)
    acceleration: pg.Vector2 = field(default_factory=pg.Vector2)
    facing: float = 0.0
    turning: int = 0
    rotation_vel: float = 0.0
    descent_accumulator: float = 0.0


@dataclass
class SlalomGate:
    rect: pg.FRect
    passed: bool = False


def play() -> Scene:

    bindings = dict(DEFAULT_CONTROLS)

    get_actions = controls.bind(bindings)

    prompt = font("arco", 48).render("Get Ready!", True, "black")
    countdown = [font("arco", 72).render(str(i), True, "black") for i in range(3, 0, -1)]

    scene = State(
        get_ready_timer=3000,
        camera=pg.FRect(0, 0, WIDTH, HEIGHT),
        particles=[],
        slalom_gates=deque(),
        trees=[],
        score=0,
        gate_score_multiplier=1,
        gate_score_labels=[],
        time_since_last_gate=0.0,
    )

    def get_slalom_gate_image(passed: bool):
        img = pg.Surface((160, 80), pg.SRCALPHA)
        color = "green" if passed else "red"
        pg.draw.rect(img, color, (0, 0, 10, 80))
        pg.draw.rect(img, color, (150, 0, 10, 80))
        return img

    slalom_gate_images = [get_slalom_gate_image(passed) for passed in (False, True)]

    def generate_slalom() -> Iterator[SlalomGate]:
        pos = pg.Vector2(WIDTH // 2, HEIGHT * 0.75)

        while True:
            if pos.y < scene.camera.bottom + HEIGHT // 2:
                offset_x = random.randint(-150, 150)
                pos.x = min(max(50, pos.x + offset_x), WIDTH - 50)
                yield SlalomGate(pg.FRect(0, 0, 160, 80).move_to(center=pos.copy()))
                pos.y += HEIGHT // 2
            yield None

    slalom = generate_slalom()

    trees = generate_trees(scene.camera, scene.slalom_gates)

    distance_score = get_distance_score_generator()

    def _scene(screen: pg.Surface, events: list[pg.Event], state: State) -> Scene | None:
        if not hasattr(state, "player"):
            img = images("skiier_00")
            state.player = Player(img, pg.FRect(img.get_rect(center=(WIDTH // 2, 100))))
            state.player.img_orig = img

        dt = state.clock.get_time() / 1000.0

        scene.time_since_last_gate += dt

        if scene.get_ready_timer <= 0:
            control_player(get_actions(events), state.player, dt)
            follow_player(state.player, scene.camera, dt)
            scene.score += distance_score.send(state.player.rect.centery)

        if scene.get_ready_timer > 0:
            scene.get_ready_timer -= state.clock.get_time()

        if slalom_gate := next(slalom):
            scene.slalom_gates.append(slalom_gate)

        # Remove trees that are off the top of the screen and add new ones at
        # the bottom
        scene.trees[:bisect(scene.trees, scene.camera.top, key=lambda t: t.bottom)] = []
        for tree in next(trees):
            scene.trees.insert(bisect(scene.trees, tree.y, key=lambda t: t.y), tree)

        screen.fill("white")

        for gate in list(scene.slalom_gates):
            screen.blit(
                slalom_gate_images[gate.passed],
                gate.rect.move(-scene.camera.x, -scene.camera.y)
            )
            if not gate.passed and gate.rect.colliderect(state.player.rect):
                gate.passed = True
                if scene.time_since_last_gate > 2.0:
                    scene.gate_score_multiplier = 1
                gate_score = scene.gate_score_multiplier * 100
                scene.score += gate_score
                scene.gate_score_multiplier += 1
                scene.time_since_last_gate = 0.0

                scene.gate_score_labels.append((
                    font("arco", 16).render(f"+{gate_score}", True, "red"),
                    pg.Vector2(gate.rect.midtop)
                ))

            if gate.rect.bottom < scene.camera.top:
                scene.slalom_gates.popleft()

        for tree in scene.trees:
            r = tree.move(-scene.camera.x, -scene.camera.y)
            pg.draw.rect(screen, "brown", r)
            p = pg.Vector2(r.midtop)
            pg.draw.polygon(
                screen,
                "darkgreen",
                [(p.x - 30, p.y + 60), (p.x, p.y - 30), (p.x + 30, p.y + 60)],
            )

        screen.blit(
            state.player.image,
            (
                (state.player.rect.center - pg.Vector2(state.player.image.get_size()) / 2)
                - scene.camera.topleft
            )
        )

        for label, pos in list(scene.gate_score_labels):
            screen.blit(label, pos - scene.camera.topleft)
            pos.y -= 50 * dt
            if pos.y + label.get_height() < scene.camera.top:
                scene.gate_score_labels.remove((label, pos))

        if scene.get_ready_timer > 0:
            screen.blit(prompt, prompt.get_rect(center=(WIDTH // 2, HEIGHT // 2)))
            num = 3 + int(-scene.get_ready_timer // 1000)
            screen.blit(
                countdown[num],
                countdown[num].get_rect(center=(WIDTH // 2, HEIGHT // 3))
            )

        else:
            score_label = font("arco", 32).render(f"Score: {scene.score}", True, "black")
            screen.blit(score_label, (10, 10))

    return _scene


def control_player(actions: list[str], player: Player, dt: float) -> None:
    player_input = pg.Vector2(0, 0)

    for action in actions:
        match action:
            case Action(name="left"):
                player.rotation_vel = 300 * dt
            case Action(name="right"):
                player.rotation_vel = -300 * dt
            case Action(name="left_released"):
                player.rotation_vel = 0
            case Action(name="right_released"):
                player.rotation_vel = 0
            case Action(name="forward"):
                player_input.y = 4500
            case Action(name="left_stick", event=event):
                print(event.value)
                player.rotation_vel = -event.value * 300 * dt
            case Action(name="right_stick", event=event):
                print(event.value)
                player.rotation_vel = -event.value * 300 * dt
            case Action(name="forward_stick", event=event):
                player_input.y = -event.value * 4500

    player.facing = min(max(player.facing + player.rotation_vel, -90), 90)

    if -90 < player.facing < -54:
        player.image = images("skiier_-90")
    if -54 <= player.facing < -18:
        player.image = images("skiier_-45")
    if -18 <= player.facing < 18:
        player.image = images("skiier_00")
    if 18 <= player.facing < 54:
        player.image = images("skiier_45")
    if 54 <= player.facing < 90:
        player.image = images("skiier_90")

    base_drag = player.velocity * -0.2
    gravity = pg.Vector2(0, 5)

    facing_vector = pg.Vector2(0, 1).rotate(-player.facing)
    player_input_force = facing_vector * player_input.y

    player.acceleration = base_drag + gravity + player_input_force
    player.velocity += player.acceleration * dt

    if (current_speed := player.velocity.magnitude()) > 0:
        angle_diff = player.velocity.angle_to(facing_vector)
        speed_loss_factor = abs(angle_diff) / 180.0
        turning_speed_reduction_rate = current_speed * speed_loss_factor * 5
        new_speed = max(0, current_speed - turning_speed_reduction_rate * dt)
        if facing_vector:
            player.velocity = facing_vector.normalize() * new_speed
        else:
            player.velocity = pg.Vector2(0, 0)

    if player.velocity.magnitude() < 20:
        player.velocity = pg.Vector2(0, 0)

    player.rect.center += player.velocity * dt


def follow_player(player: Player, camera: pg.FRect, dt: float) -> None:
    if (target_y := player.rect.centery - HEIGHT * 0.3) > camera.y:
        camera.y += (target_y - camera.y) * 5 * dt

    target_x = camera.x
    dead_zone_left = camera.x + WIDTH * 0.4
    dead_zone_right = camera.x + WIDTH * 0.6
    if player.rect.centerx < dead_zone_left:
        target_x = player.rect.centerx - WIDTH * 0.4
    elif player.rect.centerx > dead_zone_right:
        target_x = player.rect.centerx - WIDTH * 0.6

    camera.x += (target_x - camera.x) * 5 * dt


def get_distance_score_generator(
    initial_y: float = 100.0,
    distance_score_threshold: float = 100.0,
    score_increment: int = 10,
) -> Generator[int, float, None]:

    def add_score_for_distance_descended() -> Generator[int, float, None]:
        last_y = current_y = initial_y
        accumulator = 0.0

        while True:
            score = 0
            accumulator += current_y - last_y
            last_y = current_y
            while accumulator >= distance_score_threshold:
                accumulator -= distance_score_threshold
                score += 10
            current_y = yield score

    generator = add_score_for_distance_descended()
    next(generator)
    return generator


def generate_trees(
    camera: pg.FRect,
    slalom_gates: Iterable[SlalomGate],
) -> Iterator[list[pg.Rect]]:
    buffer = 100

    while True:
        if random.random() > 0.5:
            max_x = min(gate.rect.left for gate in slalom_gates) - buffer
            min_x = max_x - WIDTH
        else:
            min_x = max(gate.rect.right for gate in slalom_gates) + buffer
            max_x = min_x + WIDTH

        center = pg.Vector2(
            random.uniform(min_x, max_x),
            camera.bottom + buffer + random.randrange(HEIGHT // 4),
        )

        if random.random() > 0.9:
            # cluster of trees centered around position
            yield [
                pg.Rect(0, 0, 20, 80).move_to(midbottom=pg.Vector2(
                    center.x + random.randint(-4, 4) * 10,
                    center.y + random.randint(-2, 2) * 10
                ))
                for _ in range(random.randint(2, 5))
            ]

        else:
            yield []


if __name__ == "__main__":
    import asyncio

    asyncio.run(
        scene_manager(
            main_menu(),
            size=(WIDTH, HEIGHT),
            title="Eterniski",
        )
    )
