import torch
import numpy as np
import time

# import gymnasium
# import fire
from tqdm.notebook import tqdm

from controller.mppi import MPPI
from controller.cem import CEM
from envs.navigation_2d import Navigation2DEnv


def main(traj, save_mode: bool = True):
    env = Navigation2DEnv()

    # solver
    # solver = MPPI(
    #     horizon=30,
    #     num_samples=3000,
    #     dim_state=3,
    #     dim_control=2,
    #     dynamics=env.dynamics,
    #     cost_func=env.cost_function,
    #     u_min=env.u_min,
    #     u_max=env.u_max,
    #     sigmas=torch.tensor([0.5, 0.5]),
    #     lambda_=1.0,
    #     auto_lambda=False,
    # )
    solver = CEM(
        horizon=30,
        num_samples=300,
        dim_state=3,
        dim_control=2,
        dynamics=env.dynamics,
        cost_func=env.cost_function,
        u_min=env.u_min,
        u_max=env.u_max,
        sigmas=torch.tensor([0.5, 0.5]),
        lambda_=1.0,
        auto_lambda=False,
        iters=3,
        elite_ratio=0.1,
        min_std=1e-3,
    )

    state = env.reset()
    traj.append(state[:2].cpu().numpy())
    
    max_steps = 500
    total_time = 0.0
    step_count = 0
    for i in range(max_steps):
        start = time.time()
        action_seq, state_seq = solver.forward(state=state)
        end = time.time()
        total_time += end - start
        step_count += 1

        state, is_goal_reached = env.step(action_seq[0, :])
        traj.append(state[:2].cpu().numpy())

        is_collisions = env.collision_check(state=state_seq)

        try:
            top_samples, top_weights = solver.get_top_samples(num_samples=50)
        except:
            top_samples, top_weights = solver.get_top_samples(num_samples=solver._num_samples)

        if save_mode:
            env.render(
                predicted_trajectory=state_seq,
                is_collisions=is_collisions,
                top_samples=(top_samples, top_weights),
                mode="rgb_array",
            )
            # progress bar
            if i == 0:
                pbar = tqdm(total=max_steps, desc="recording video")
            pbar.update(1)

        else:
            env.render(
                predicted_trajectory=state_seq,
                is_collisions=is_collisions,
                top_samples=(top_samples, top_weights),
                mode="human",
            )
        if is_goal_reached:
            print("Goal Reached!")
            break

    average_time = total_time / step_count
    print("average solve time: {:.3f} ms".format(average_time * 1000))
    env.close()  # close window and save video if save_mode is True
    
    return np.stack(traj, axis=0)


if __name__ == "__main__":
    main()
