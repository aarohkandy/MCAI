from __future__ import annotations

import asyncio
import concurrent.futures
import copy
import threading
import time

import torch

import combat_ai.service as service_module
from combat_ai.config import PPOConfig, ServiceConfig
from combat_ai.service import PolicyService
from fixtures import observation


def _message(tick: int, execution: dict | None = None) -> dict:
    step = {
        "agent_id": "fighter-a",
        "observation": observation(tick=tick),
        "reward": 0.0,
        "terminated": False,
        "truncated": False,
        "info": {},
    }
    if execution is not None:
        step["execution"] = execution
    return {
        "schema_version": 1,
        "type": "step_batch",
        "sequence": tick,
        "steps": [step],
    }


async def _service_with_one_transition(tmp_path) -> PolicyService:
    service = PolicyService(
        PPOConfig(rollout_agent_ticks=16, recurrent_sequence_length=1),
        ServiceConfig(
            checkpoint_dir=tmp_path,
            deterministic_inference=True,
            cpu_threads=1,
        ),
    )
    # Population behavior is orthogonal to actor/learner publication and can
    # otherwise assign the sole test fighter as its own scripted opponent.
    service.league.assign_batch = lambda _steps: None
    service.league.assignment_for = lambda _episode: None
    first = (await service.handle_message(_message(1)))["actions"][0]
    await service.handle_message(_message(2, {
        "source": "policy",
        "action_id": first["action_id"],
        "action": first["action"],
    }))
    assert len(service.buffer) == 1
    return service


def _successful_job(entered: threading.Event, release: threading.Event):
    def run(
        actor_state, optimizer_state, _transitions, _config,
        _imitation_records, _online_imitation_records, _elite_replay_records,
        _imitation_weight, reward_metrics,
        _checkpoint_directory, _snapshot_interval, proposed_state,
        _low_kl_updates,
    ):
        entered.set()
        if not release.wait(timeout=10):
            raise TimeoutError("test did not release learner")
        learned_state = copy.deepcopy(actor_state)
        parameter_name = next(
            name for name, value in learned_state.items()
            if torch.is_floating_point(value)
        )
        learned_state[parameter_name].add_(1.0)
        metrics = {"valid_samples": 1, "learning_rate": 1e-4}
        stage = _checkpoint_directory / "latest.test.stage"
        torch.save({
            "policy": learned_state,
            "optimizer": copy.deepcopy(optimizer_state),
            "policy_version": proposed_state.policy_version,
            "rollout_generation": proposed_state.rollout_generation,
        }, stage)
        return (
            metrics, reward_metrics, proposed_state, 2,
            str(stage), None, None,
        )

    return run


def test_blocked_learner_does_not_block_actions_and_publishes_atomically(
    monkeypatch, tmp_path,
):
    async def scenario() -> None:
        service = await _service_with_one_transition(tmp_path)
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        service.learner_executor = executor
        entered = threading.Event()
        release = threading.Event()
        monkeypatch.setattr(
            service_module, "_learn_generation_job",
            _successful_job(entered, release),
        )
        monkeypatch.setattr(service.checkpoints, "promote_staged", lambda *_args: None)
        old_version = service.state.policy_version
        parameter_name, old_parameter = next(iter(service.policy.state_dict().items()))
        old_parameter = old_parameter.detach().clone()

        try:
            service._start_background_update()
            assert await asyncio.to_thread(entered.wait, 2.0)

            # The learner is deliberately stalled, yet the old actor must
            # still complete an ordinary websocket decision promptly.
            response = await asyncio.wait_for(
                service.handle_message(_message(3)), timeout=2.0,
            )
            assert response["actions"]
            assert response["policy_version"] == old_version
            assert torch.equal(service.policy.state_dict()[parameter_name], old_parameter)

            release.set()
            assert service.training_task is not None
            await asyncio.wait_for(asyncio.shield(service.training_task), timeout=3.0)
            await service._publish_background_generation()

            assert service.state.policy_version == old_version + 1
            assert service.trainer.low_kl_updates == 2
            assert not torch.equal(service.policy.state_dict()[parameter_name], old_parameter)
            assert service.training_task is None
        finally:
            release.set()
            executor.shutdown(wait=True, cancel_futures=True)

    asyncio.run(scenario())


def test_failed_learner_keeps_actor_and_restores_drained_rollout(monkeypatch, tmp_path):
    async def scenario() -> None:
        service = await _service_with_one_transition(tmp_path)
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        service.learner_executor = executor
        old_version = service.state.policy_version
        old_parameters = {
            name: value.detach().clone()
            for name, value in service.policy.state_dict().items()
        }

        def fail(*_args):
            raise RuntimeError("synthetic learner failure")

        monkeypatch.setattr(service_module, "_learn_generation_job", fail)
        try:
            service._start_background_update()
            assert service.training_task is not None
            while not service.training_task.done():
                await asyncio.sleep(0)
            await service._publish_background_generation()

            assert service.training_task is None
            assert service.state.policy_version == old_version
            assert len(service.buffer) == 1
            assert service.training_transitions == []
            assert all(
                torch.equal(service.policy.state_dict()[name], value)
                for name, value in old_parameters.items()
            )
            response = await asyncio.wait_for(
                service.handle_message(_message(3)), timeout=2.0,
            )
            assert response["actions"]
            assert response["policy_version"] == old_version
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

    asyncio.run(scenario())


def test_publish_does_not_wait_for_large_elite_replay_sidecar(monkeypatch, tmp_path):
    async def scenario() -> None:
        service = await _service_with_one_transition(tmp_path)
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        service.learner_executor = executor
        learner_entered = threading.Event()
        release_learner = threading.Event()
        sidecar_entered = threading.Event()
        release_sidecar = threading.Event()
        monkeypatch.setattr(
            service_module, "_learn_generation_job",
            _successful_job(learner_entered, release_learner),
        )

        def blocked_sidecar(_directory, records, version):
            sidecar_entered.set()
            if not release_sidecar.wait(timeout=10):
                raise TimeoutError("test did not release sidecar")
            return int(version), len(records)

        monkeypatch.setattr(
            service_module, "_save_elite_replay_sidecar_job", blocked_sidecar,
        )
        monkeypatch.setattr(service.checkpoints, "promote_staged", lambda *_args: None)

        try:
            service._start_background_update()
            assert await asyncio.to_thread(learner_entered.wait, 2.0)
            release_learner.set()
            assert service.training_task is not None
            await asyncio.wait_for(asyncio.shield(service.training_task), timeout=3.0)

            # Publication must only queue the expensive replay write. The
            # actor remains responsive while that write is deliberately stuck.
            await asyncio.wait_for(service._publish_background_generation(), timeout=1.0)
            assert await asyncio.to_thread(sidecar_entered.wait, 2.0)
            response = await asyncio.wait_for(
                service.handle_message(_message(3)), timeout=2.0,
            )
            assert response["actions"]
            assert service.sidecar_tasks
        finally:
            release_learner.set()
            release_sidecar.set()
            if service.sidecar_tasks:
                await asyncio.gather(*tuple(service.sidecar_tasks))
            executor.shutdown(wait=True, cancel_futures=True)

    asyncio.run(scenario())


def test_reward_profile_cutover_keeps_inference_live_but_quarantines_ppo(tmp_path):
    async def scenario() -> None:
        service = await _service_with_one_transition(tmp_path)
        assert len(service.buffer) == 1

        service._begin_adaptive_reward_quarantine(4)
        assert len(service.buffer) == 0
        assert service.pending == {}

        during = (await service.handle_message(_message(3)))["actions"][0]
        assert during["action"]
        await service.handle_message(_message(4, {
            "source": "policy",
            "action_id": during["action_id"],
            "action": during["action"],
        }))
        assert len(service.buffer) == 0
        assert service.adaptive_reward_quarantined_steps >= 2

        service.adaptive_reward_quarantine_until = time.monotonic() - 1.0
        after = (await service.handle_message(_message(5)))["actions"][0]
        await service.handle_message(_message(6, {
            "source": "policy",
            "action_id": after["action_id"],
            "action": after["action"],
        }))

        assert len(service.buffer) == 1
        assert not service.adaptive_reward_cutover_telemetry()["active"]

    asyncio.run(scenario())


def test_adaptive_reward_rollout_metrics_difference_server_counters(tmp_path):
    async def scenario() -> None:
        service = await _service_with_one_transition(tmp_path)
        first = {
            "stats": {
                "damage_taken": 5.0,
                "self_damage": 2.0,
                "inaction_penalty_ticks": 7,
                "policy_crystal_chains_started": 3,
                "policy_crystal_chains_damaging": 1,
                "execution": {"policy": {
                    "damage_dealt": 8.0,
                    "hits_landed": 4,
                    "blocks_placed": 2,
                    "blocks_mined": 1,
                }},
            }
        }
        second = copy.deepcopy(first)
        second["stats"].update({
            "damage_taken": 7.0,
            "self_damage": 3.0,
            "inaction_penalty_ticks": 10,
            "policy_crystal_chains_started": 4,
            "policy_crystal_chains_damaging": 2,
        })
        second["stats"]["execution"]["policy"].update({
            "damage_dealt": 11.0,
            "hits_landed": 6,
            "blocks_placed": 3,
            "blocks_mined": 3,
        })

        service._record_policy_crystal_counters("fighter-a", "episode-1", first)
        service._record_policy_crystal_counters("fighter-a", "episode-1", second)
        metrics = service.reward_telemetry.metrics()

        assert metrics["server_policy_damage_dealt"] == 11.0
        assert metrics["server_policy_hits_landed"] == 6.0
        assert metrics["server_damage_taken"] == 7.0
        assert metrics["server_self_damage"] == 3.0
        assert metrics["server_inaction_penalty_ticks"] == 10.0
        assert metrics["server_policy_blocks_placed"] == 3.0
        assert metrics["server_policy_blocks_mined"] == 3.0
        assert metrics["server_policy_crystal_chains_started_events"] == 4
        assert metrics["server_policy_crystal_chains_damaging_events"] == 2

    asyncio.run(scenario())
