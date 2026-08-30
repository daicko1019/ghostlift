"""
LLM-based agent in 2D worlds with multiple places.
"""
import argparse
import logging
import os
import shutil
import sys
import time
from typing import Optional, Tuple
from simulation import Simulation
from utils import load_config
from visualization import Visualizer

# Constants
DEFAULT_FRAME_INTERVAL_CONFIG = 50  # Used only when config.yaml omits frame_interval


def setup_logging(config: dict):
    """Setup logging configuration"""
    log_config = config.get('logging', {})
    level = getattr(logging, log_config.get('level', 'INFO'))
    
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler()]
    
    if 'log_file' in log_config:
        handlers.append(logging.FileHandler(log_config['log_file']))
    
    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=handlers
    )


def check_ollama_setup(sim: Simulation, logger: logging.Logger) -> bool:
    """Check Ollama connection and model availability"""
    client = sim.llm_client

    # One request answers both questions: whether the server is reachable and
    # which models it has. None means unreachable, [] means reachable but empty.
    available_models = client.fetch_models()

    # One dropped probe over a private network should not discard a run that
    # costs many minutes of GPU time, so give the server a couple of chances.
    for attempt in range(4):
        if available_models is not None:
            break
        time.sleep(5)
        logger.warning(f"Ollama not reachable, retrying ({attempt + 1}/4)")
        available_models = client.fetch_models()

    if available_models is None:
        logger.error("Cannot connect to Ollama. Please make sure Ollama is running.")
        logger.error(f"Expected URL: {client.base_url}")
        return False

    if client.model not in available_models:
        logger.warning(f"Model '{client.model}' not found in Ollama.")
        if available_models:
            logger.info(f"Available models: {', '.join(available_models)}")
            logger.info("Please update 'llm.model' in config.yaml or download the model:")
            logger.info(f"  ollama pull {client.model}")
        else:
            logger.error("No models found in Ollama. Please download a model first.")
            logger.error(f"Example: ollama pull {client.model}")
        return False

    logger.info(f"Using model: {client.model}")
    return True


def determine_visualization_settings(args, config: dict) -> Tuple[bool, int, str]:
    """Determine visualization settings from args and config

    Returns (should_save, frame_interval, output_dir). Visualization means
    writing PNG frames; no window is ever opened.
    """
    visualization_config = config.get('visualization', {})

    should_save = args.save_frames or visualization_config.get('save_frames', False)

    frame_interval = (
        args.frame_interval or
        visualization_config.get('frame_interval', DEFAULT_FRAME_INTERVAL_CONFIG)
    )

    output_dir = visualization_config.get('output_dir', 'output')

    return should_save, frame_interval, output_dir


def save_frame(
    visualizer: Visualizer,
    sim: Simulation,
    step: int,
    frame_interval: int,
    output_dir: str,
    logger: logging.Logger
):
    """Write the current simulation state to output_dir as a PNG frame"""
    if step % frame_interval != 0 and step != sim.duration - 1:
        return

    save_path = os.path.join(output_dir, f"frame_{step:04d}.png")

    # A drawing failure must not abort a run that costs hours of LLM calls,
    # so visualization errors are logged and the simulation continues.
    try:
        visualizer.visualize_step(
            sim.agents,
            sim.get_place_status(),
            step,
            save_path,
            communication_radius=sim.communication_radius,
            fire_states=sim.fire_states
        )
    except Exception as e:
        logger.error(f"Error drawing frame for step {step}: {e}", exc_info=True)
        return

    logger.info(f"Saved frame: {save_path}")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Simulation of LLM-based agent in 2D worlds with multiple places.'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--save-frames',
        action='store_true',
        help='Save PNG frames even if visualization.save_frames is false in the config'
    )
    parser.add_argument(
        '--frame-interval',
        type=int,
        default=None,
        help='Interval between visualization frames (overrides config)'
    )
    parser.add_argument(
        '--llm-url',
        type=str,
        default=None,
        help='Ollama base URL, overriding llm.base_url in the config. Lets a run '
             'be pointed at a machine with a GPU without editing the scenario file '
             '(e.g. http://100.78.189.115:11434 over a private network)'
    )

    return parser.parse_args()


def prepare_output_directory(output_dir: str, should_save: bool, logger: logging.Logger):
    """Start each run from an empty output directory

    The directory is always cleared so frames from a previous run cannot be
    mistaken for the current one. It is recreated here only when frames are
    saved; the jsonl logs in Simulation create it on demand otherwise.
    """
    if os.path.exists(output_dir):
        logger.info(f"Removing existing output directory: {output_dir}")
        shutil.rmtree(output_dir)

    if should_save:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")


def run_simulation(
    sim: Simulation,
    visualizer: Optional[Visualizer],
    frame_interval: int,
    output_dir: str,
    logger: logging.Logger
):
    """Run every simulation step, saving a frame after each one"""
    sim.initialize_agents()

    if not check_ollama_setup(sim, logger):
        raise RuntimeError(f"Ollama unreachable or model missing at {sim.llm_client.base_url}")

    # Save the initial state (step 0) before the first LLM step. One step costs
    # one LLM call per agent twice over, so it can take minutes; without this
    # the output directory looks empty for the whole first step.
    if visualizer:
        save_frame(visualizer, sim, sim.step, frame_interval, output_dir, logger)

    logger.info("Starting simulation...")

    while sim.step < sim.duration:
        sim.step_simulation()
        if visualizer:
            save_frame(visualizer, sim, sim.step, frame_interval, output_dir, logger)

    logger.info("Simulation completed")


def main():
    """Main function"""
    args = parse_arguments()
    config = load_config(args.config)

    if args.llm_url:
        config['llm']['base_url'] = args.llm_url

    setup_logging(config)
    logger = logging.getLogger(__name__)

    if args.llm_url:
        logger.info(f"LLM endpoint overridden: {args.llm_url}")

    should_save, frame_interval, output_dir = determine_visualization_settings(args, config)
    prepare_output_directory(output_dir, should_save, logger)

    sim = Simulation(config, output_dir=output_dir)

    # Without frame saving there is nothing for the visualizer to produce,
    # since this project never opens a window.
    visualizer = None
    if should_save:
        visualizer = Visualizer(
            half_space_size=sim.half_space_size,
            places=sim.places,
            num_agents=sim.num_agents
        )

    try:
        run_simulation(sim, visualizer, frame_interval, output_dir, logger)
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Error during simulation: {e}", exc_info=True)
        # Exit non-zero so a batch runner cannot mistake an empty run for a
        # finished one. A seed whose control run silently produced nothing
        # cannot be decomposed, and that must not pass unnoticed.
        sys.exit(1)


if __name__ == "__main__":
    main()

