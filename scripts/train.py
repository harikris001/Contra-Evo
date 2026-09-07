#!/usr/bin/env python3
"""CLI: python scripts/train.py --config configs/default.yaml --render"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

# ensure src on path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.config import load_config
from training.trainer import Trainer
from utils.logging import get_logger

logger = get_logger(__name__)


def _publish_population_tiles(shared_frames, population, stats_list) -> None:
    """Write each tile's end-of-generation state into the shared frame dict.

    Always overwrites existing entries:
    - live worker publishes are stale once evaluation finished,
    - carried elites are never published by workers at all, so without an
      overwrite their stale gen-0 raw-info entry would block the elite badge
      (elite_carry/evaluated_at_gen) from ever reaching the dashboard.
    Tiles without a final frame are left untouched.
    """
    for ind, st in zip(population, stats_list):
        if not isinstance(st, dict):
            continue
        frame = st.get("_frame")
        if frame is None:
            continue
        try:
            shared_frames[ind.id] = (frame, st)
        except Exception:
            pass


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--render", action="store_true", help="enable live population visualization")
    p.add_argument("--generations", type=int, default=None)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--render-mode", type=str, default="population", choices=["population", "best", "comparison"])
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    # overrides
    if args.workers:
        cfg["evaluation"]["workers"] = args.workers
    if args.render:
        cfg["visualization"]["enabled"] = True
        cfg["training"]["mode"] = "live_training"
        cfg["training"]["headless"] = False
    if args.generations:
        cfg["training"]["generations"] = args.generations

    trainer = Trainer(cfg)

    # Live rendering setup
    render_callback = None
    dashboard = None
    shared_frames = None
    manager = None
    dashboard_thread = None
    if cfg["visualization"]["enabled"]:
        from visualization.dashboard import Dashboard
        import multiprocessing

        try:
            dashboard = Dashboard(
                population_size=cfg["evolution"]["population_size"],
                fps=cfg["visualization"].get("fps", 15),
                layout=cfg["visualization"].get("layout", "3x3"),
                show_charts=cfg["visualization"].get("show_charts", True),
            )
            dashboard._init_pygame()
            # Initial draw so window appears immediately
            dashboard.update({}, {}, {"generation": 0, "best_fitness": 0, "mean_fitness": 0})
            dashboard.draw()
            print(f"Pygame window opened: {dashboard.window_size[0]}x{dashboard.window_size[1]} {dashboard.layout} | Close window or ESC to exit")
        except Exception as e:
            print(f"[VIS] Pygame window failed ({e}), falling back to headless text mode")
            dashboard = None

        # Create shared dict for live frames (Manager for cross-process)
        shared_global = None
        stop_flag = {"stop": False}
        training_thread = None
        if dashboard is not None:
            try:
                manager = multiprocessing.Manager()
                shared_frames = manager.dict()
                shared_global = manager.dict()
            except Exception as e:
                print(f"[VIS] Manager failed {e}, using in-proc dict")
                manager = None
                shared_frames = {}  # type: ignore
                shared_global = {}  # type: ignore

            # Pass shared dict to trainer via monkey-patch (trainer will check)
            trainer.evaluator.shared_frames = shared_frames  # type: ignore

            # Per-generation callback for chart history and tile states
            def render_callback_fn(gen, ga_stats, population, stats_list):
                try:
                    for k, v in ga_stats.items():
                        shared_global[k] = v  # type: ignore
                    shared_global["generation"] = gen  # type: ignore
                except Exception:
                    pass
                # Overwrite every tile with this generation's final state
                # (carried elites surface their ELITE badge here; playing
                # agents surface DEAD/IDLE-OUT/DONE end states).
                _publish_population_tiles(shared_frames, population, stats_list)
                print(f"[VIS] Gen {gen} best {ga_stats['best_fitness']:.1f} avg {ga_stats['mean_fitness']:.1f}")

            render_callback = render_callback_fn

            # Start training in background thread, dashboard on main thread (macOS requirement)
            import threading

            def train_fn():
                try:
                    trainer.train(generations=args.generations, resume_path=Path(args.resume) if args.resume else None, render_callback=render_callback)
                except SystemExit:
                    pass
                except Exception as e:
                    print(f"[VIS] Training failed: {e}")
                    import traceback

                    traceback.print_exc()
                finally:
                    stop_flag["stop"] = True

            training_thread = threading.Thread(target=train_fn, daemon=True)
            training_thread.start()

            # Main thread: dashboard loop at 15 FPS, polling shared dict
            import time

            print("[VIS] Dashboard running on main thread at 15 FPS (workers publish live frames)")
            try:
                while not stop_flag["stop"]:
                    # Collect frames from shared dict
                    frames: dict = {}
                    states: dict = {}
                    for k in list(shared_frames.keys()):  # type: ignore
                        try:
                            v = shared_frames[k]  # type: ignore
                            if isinstance(v, tuple) and len(v) == 2:
                                frame, info = v
                                frames[int(k)] = frame
                                states[int(k)] = info
                            else:
                                frames[int(k)] = v
                        except Exception:
                            pass
                    global_state = dict(shared_global) if shared_global else {}  # type: ignore
                    # Update and draw (even if empty, to show grid + stats)
                    try:
                        if frames or states or global_state:
                            dashboard.update(frames, states, global_state)
                        dashboard.draw()
                    except Exception as e:
                        print(f"[VIS] Draw failed: {e}")
                    if not dashboard.handle_events():
                        print("[VIS] Window closed, stopping training")
                        stop_flag["stop"] = True
                        break
                    time.sleep(1 / max(1, dashboard.fps))
                    # Check if training finished
                    if not training_thread.is_alive() and not frames:
                        # Training done, keep window for a bit then exit
                        pass
            except KeyboardInterrupt:
                print("[VIS] Interrupted")
                stop_flag["stop"] = True
            finally:
                dashboard.close()
                if manager is not None:
                    try:
                        manager.shutdown()
                    except Exception:
                        pass
                # Wait for training thread to finish
                if training_thread.is_alive():
                    training_thread.join(timeout=2)
                print("Training done, window closed")
                return  # Exit main, don't run the second trainer.train below

        else:
            # Headless fallback
            def render_callback_fn(gen, ga_stats, population, stats_list):
                print(f"[VIS] Gen {gen} best {ga_stats['best_fitness']:.1f} avg {ga_stats['mean_fitness']:.1f}")

            render_callback = render_callback_fn
            if args.render:
                print("Live visualization enabled (headless): install pygame and run in GUI for window.")

    # Also support headless callback when visualization disabled but --render not set
    if render_callback is None and cfg["visualization"]["enabled"]:
        def render_callback_fn(gen, ga_stats, population, stats_list):
            print(f"[VIS] Gen {gen} best {ga_stats['best_fitness']:.1f} avg {ga_stats['mean_fitness']:.1f}")

        render_callback = render_callback_fn

    # Non-dashboard path (headless or no window) - blocking train
    if dashboard is None:
        trainer.train(generations=args.generations, resume_path=Path(args.resume) if args.resume else None, render_callback=render_callback)
    else:
        # Dashboard path already handled above with threaded training; if we reach here, dashboard was not created
        # Fallback: if dashboard was created but we didn't take the early return (should not happen), just train
        if training_thread is None:
            trainer.train(generations=args.generations, resume_path=Path(args.resume) if args.resume else None, render_callback=render_callback)

    # Keep window open after training (only for non-threaded fallback)
    if dashboard is not None and (training_thread is None or not training_thread.is_alive()):
        print("Training done, keeping window open (close window or ESC to exit, auto-close in 30s)")
        import time

        for _ in range(30 * max(1, dashboard.fps)):
            try:
                if stop_flag["stop"]:
                    break
                dashboard.draw()
                if not dashboard.handle_events():
                    break
            except Exception:
                break
            time.sleep(1 / max(1, dashboard.fps))
        dashboard.close()
        if manager is not None:
            try:
                manager.shutdown()
            except Exception:
                pass


if __name__ == "__main__":
    main()
