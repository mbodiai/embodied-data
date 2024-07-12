"""This module demonstrates the usage of the embdata library for processing and analyzing complex datasets.

It showcases how to load a dataset, create samples, episodes, and trajectories, and perform various
operations on them such as flattening, plotting, and transforming.

Example:
    To run this demo:

    ```
    python demo.py
    ```

    This will generate several PNG files with visualizations of the processed data.
"""

from typing import TYPE_CHECKING

from datasets import get_dataset_config_info, get_dataset_config_names, load_dataset

from embdata.describe import describe
from embdata.episode import Episode
from embdata.sample import Sample
from embdata.schema.oxe import get_hf_dataset, get_oxe_metadata, get_state

if TYPE_CHECKING:
    from embdata.trajectory import Trajectory


def load_and_process_dataset() -> tuple[Sample, Sample, Sample]:
    """Load a dataset, create a sample, and process it into actions, observations, and states.

    Returns:
        tuple: A tuple containing actions, observations, and states as flattened samples.

    Example:
        >>> actions, observations, states = load_and_process_dataset()
        >>> print(type(actions), type(observations), type(states))
        <class 'embdata.sample.Sample'> <class 'embdata.sample.Sample'> <class 'embdata.sample.Sample'>
    """
    ds = load_dataset("mbodiai/oxe_bridge_v2")
    describe(ds)

    s = Sample(ds["shard_0"])
    describe(s)

    actions = s.flatten(to="action")
    observations = s.flatten(to="observation")
    states = s.flatten(to="state")

    describe(actions)
    describe(observations)

    return actions, observations, states

def create_and_analyze_episode(observations, actions, states) -> Episode:
    """Create an episode from observations, actions, and states, and perform various analyses.

    Args:
        observations (Sample): Flattened observations.
        actions (Sample): Flattened actions.
        states (Sample): Flattened states.

    Returns:
        Episode: The created and analyzed episode.

    Example:
        >>> actions, observations, states = load_and_process_dataset()
        >>> episode = create_and_analyze_episode(observations, actions, states)
        >>> print(type(episode))
        <class 'embdata.episode.Episode'>
    """
    e = Episode(zip(observations, actions, states, strict=False), metadata={"freq_hz": 5})
    describe(e.unpack())
    describe(e)

    e.trajectory(freq_hz=1).plot().save("trajectory.png")
    e.trajectory().resample(target_hz=50).plot().save("resampled_trajectory.png")

    t: Trajectory = e.trajectory().transform("minmax", min=0, max=255).plot().save("minmaxed_trajectory.png")
    t.transform("unminmax", orig_min=e.trajectory().min(), orig_max=e.trajectory().max()).plot()

    e.trajectory().frequencies().plot().save("frequencies.png")

    return e

if __name__ == "__main__":
    # actions, observations, states = load_and_process_dataset()
    # episode = create_and_analyze_episode(observations, actions, states)
    from rich import print
    cn = get_dataset_config_names("jxu124/OpenX-Embodiment")
    for c in cn:
        print(get_dataset_config_info("jxu124/OpenX-Embodiment", c))
        describe(load_dataset("jxu124/OpenX-Embodiment", c))

    ds = get_hf_dataset("taco_play")


