"""Emulator factory - isolates stable_retro creation and checks."""
from __future__ import annotations

import fcntl
import hashlib
import json
import shutil
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator

from utils.logging import get_logger

logger = get_logger(__name__)

GAME_NAME = "Contra-Nes"

_CONTRA_INFO = {
    "lives": {"address": 50, "type": "|u1"},
    "lives_p2": {"address": 51, "type": "|u1"},
    "stage": {"address": 48, "type": "|u1"},
    "screen_type": {"address": 44, "type": "|u1"},
    "game_routine": {"address": 24, "type": "|u1"},
    "game_status": {"address": 56, "type": "|u1"},
    # NOTE: no "game_over" alias - the semantics of address 56 are unverified,
    # treating its value 1 as game-over caused spurious deaths. Deaths are
    # detected via lives/death_flag/player_state in reward.py.
    "demo_mode": {"address": 28, "type": "|u1"},
    "x_pos": {"address": 820, "type": "|u1"},
    "y_pos": {"address": 794, "type": "|u1"},
    "score": {"address": 2018, "type": ">d2"},
    "level_screen": {"address": 100, "type": "|u1"},
    "level_scroll": {"address": 101, "type": "|u1"},
    "x_scroll": {"address": 101, "type": "|u1"},
    "player_state": {"address": 144, "type": "|u1"},
    "jump_status": {"address": 160, "type": "|u1"},
    "edge_fall": {"address": 164, "type": "|u1"},
    "water_state": {"address": 178, "type": "|u1"},
    "death_flag": {"address": 180, "type": "|u1"},
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def custom_retro_dir() -> Path:
    return project_root() / ".custom_retro"


def resolve_rom_path(rom_path: str | Path | None) -> Path | None:
    if rom_path is None or str(rom_path).strip() in ("", "null", "None"):
        return None
    p = Path(rom_path)
    if p.exists():
        return p.resolve()
    alt = project_root() / p
    if alt.exists():
        return alt.resolve()
    return p


def check_rom(rom_path: str | Path | None) -> tuple[bool, str]:
    resolved = resolve_rom_path(rom_path)
    if resolved is None:
        return False, "rom_path is None - set env.rom_path to a legally obtained Contra.nes"
    if not resolved.exists():
        return False, f"ROM not found at {resolved} - provide legally obtained ROM via configs"
    if resolved.stat().st_size < 1024:
        return False, f"ROM file too small ({resolved.stat().st_size} bytes) - likely invalid"
    return True, f"ROM found: {resolved} ({resolved.stat().st_size} bytes)"


def check_retro_install() -> tuple[bool, str]:
    try:
        import retro  # type: ignore

        games_stable = retro.data.list_games(inttype=retro.data.Integrations.STABLE)  # type: ignore
        try:
            games_custom = retro.data.list_games(inttype=retro.data.Integrations.CUSTOM)  # type: ignore
        except Exception:
            games_custom = []
        msg = (
            f"stable-retro {retro.__version__ if hasattr(retro, '__version__') else 'installed'} - "
            f"{len(games_stable)} stable, {len(games_custom)} custom"
        )
        contra_games = [g for g in games_stable + games_custom if "Contra" in g]
        if contra_games:
            msg += f" | Contra found: {contra_games}"
        else:
            msg += " | Contra not in list (expected - needs custom ROM)"
        return True, msg
    except ImportError as e:
        return False, f"stable-retro not installed: {e} - pip install stable-retro"
    except Exception as e:
        return False, f"retro check failed: {e}"


@contextmanager
def _exclusive_lock(lock_path: Path, timeout: float = 60.0) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(lock_path, "a+")
    start = time.time()
    try:
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.time() - start > timeout:
                    raise TimeoutError(f"Timed out waiting for {lock_path}")
                time.sleep(0.05)
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        handle.close()


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def _ensure_custom_contra(rom_path: Path) -> Path:
    """Install custom Contra integration. Returns custom_base path."""
    import retro  # type: ignore

    custom_base = custom_retro_dir()
    game_dir = custom_base / GAME_NAME
    game_dir.mkdir(parents=True, exist_ok=True)

    data_json = {"info": _CONTRA_INFO}
    scenario_json = {"done": {}, "reward": {}}
    metadata_json = {"default_state": None, "default_player_state": []}
    data_text = json.dumps(data_json, indent=2)
    scenario_text = json.dumps(scenario_json, indent=2)
    metadata_text = json.dumps(metadata_json, indent=2)

    src = rom_path.resolve()
    dest_rom = game_dir / "rom.nes"
    md5 = hashlib.md5(src.read_bytes()).hexdigest()

    with _exclusive_lock(game_dir / ".lock"):
        sha_path = game_dir / "rom.sha"
        need_rom = (
            not dest_rom.exists()
            or dest_rom.stat().st_size < 1024
            or not sha_path.exists()
            or sha_path.read_text().strip() != md5
        )
        if need_rom:
            logger.info(f"Creating custom retro ROM copy at {dest_rom}")
            tmp_rom = game_dir / "rom.nes.tmp"
            shutil.copyfile(src, tmp_rom)
            tmp_rom.replace(dest_rom)
            _atomic_write_text(sha_path, md5)

        if not dest_rom.exists() or dest_rom.stat().st_size < 1024:
            raise FileNotFoundError(f"Custom integration missing rom.nes at {dest_rom}")

        if not (game_dir / "data.json").exists() or (game_dir / "data.json").read_text() != data_text:
            _atomic_write_text(game_dir / "data.json", data_text)
            _atomic_write_text(game_dir / "scenario.json", scenario_text)
            _atomic_write_text(game_dir / "metadata.json", metadata_text)
            logger.info(f"Custom Contra integration RAM map written: {game_dir}")

    try:
        retro.data.Integrations.add_custom_path(str(custom_base.resolve()))
    except Exception as e:
        logger.warning(f"Failed to add custom path {custom_base}: {e}")

    return custom_base


def make_retro_env(
    rom_path: str | Path | None,
    state: str | None = None,
    render_mode: str | None = None,
    use_restricted_actions: Any = None,
) -> Any:
    """Create stable_retro env from the custom integration. Never uses STABLE (no ROM)."""
    import retro  # type: ignore

    ok, msg = check_rom(rom_path)
    if not ok:
        raise FileNotFoundError(msg)
    p = resolve_rom_path(rom_path)
    assert p is not None
    logger.info(f"Creating retro env: rom={p} state={state}")

    custom_base = _ensure_custom_contra(p)
    rom_in_custom = custom_base / GAME_NAME / "rom.nes"
    if not rom_in_custom.exists():
        raise FileNotFoundError(f"No rom.nes in custom integration: {rom_in_custom}")

    if use_restricted_actions is None:
        try:
            use_restricted_actions = retro.Actions.ALL
        except Exception:
            pass

    last_err: Exception | None = None
    for attempt in range(5):
        try:
            retro.data.Integrations.add_custom_path(str(custom_base.resolve()))
        except Exception:
            pass
        try:
            env = retro.make(
                game=GAME_NAME,
                state=state,
                render_mode=render_mode,
                use_restricted_actions=use_restricted_actions,
                inttype=retro.data.Integrations.CUSTOM,
            )
            logger.info(f"Retro env created via CUSTOM: game={GAME_NAME} action_space={env.action_space}")
            return env
        except Exception as e:
            last_err = e
            logger.warning(f"retro.make CUSTOM attempt {attempt + 1}/5 failed: {e}")
            time.sleep(0.15 * (attempt + 1))
            try:
                _ensure_custom_contra(p)
            except Exception as ensure_err:
                last_err = ensure_err

    raise FileNotFoundError(
        f"Could not create Contra env from {p}. Custom dir={custom_base}. "
        f"Need {rom_in_custom}. Error: {last_err}"
    )


def verify_emulator(env: Any) -> Dict[str, Any]:
    """Verify reset, step, framebuffer, controller."""
    report: Dict[str, Any] = {}
    try:
        obs, info = env.reset()
        report["reset"] = f"OK obs shape {getattr(obs, 'shape', type(obs))} info keys {list(info.keys())[:10]}"
    except Exception as e:
        report["reset"] = f"FAIL: {e}"
        return report
    try:
        obs, rew, terminated, truncated, info = env.step(
            env.action_space.sample() if hasattr(env, "action_space") else 0
        )
        report["step"] = f"OK rew={rew} term={terminated} trunc={truncated}"
        if hasattr(obs, "shape"):
            report["framebuffer"] = f"shape {obs.shape} dtype {obs.dtype}"
        else:
            report["framebuffer"] = f"type {type(obs)}"
    except Exception as e:
        report["step"] = f"FAIL: {e}"
    try:
        if hasattr(env, "render"):
            frame = env.render()
            report["render"] = f"OK {type(frame)}"
    except Exception as e:
        report["render"] = f"FAIL: {e}"
    return report
