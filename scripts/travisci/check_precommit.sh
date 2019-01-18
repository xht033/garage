#!/usr/bin/env bash
status=0

pylint --rcfile=setup.cfg garage/misc/logger/singleton_tabular.py tests/fixtures/tf/instrumented_npo.py garage/misc/prog_bar_counter.py garage/theano/algos/npo.py garage/tf/algos/npo.py garage/tf/regressors/bernoulli_mlp_regressor.py garage/misc/autoargs.py tests/garage/tf/algos/test_ddpg.py garage/misc/tabulate.py scripts/run_experiment.py garage/theano/algos/reps.py tests/garage/tf/algos/test_vpg.py garage/algos/cem.py tests/garage/tf/algos/test_trpo.py garage/envs/mujoco/maze/maze_env.py garage/envs/mujoco/gather/gather_env.py garage/misc/snapshotter.py garage/tf/algos/batch_polopt.py tests/garage/tf/policies/test_gaussian_policies.py garage/theano/regressors/gaussian_mlp_regressor.py garage/misc/logger/logger_outputs.py garage/__init__.py garage/theano/regressors/gaussian_conv_regressor.py garage/config.py garage/envs/mujoco/maze/maze_env_utils.py garage/tf/regressors/categorical_mlp_regressor.py garage/envs/mujoco/simple_humanoid_env.py garage/tf/policies/gaussian_mlp_policy.py tests/fixtures/theano/instrumented_batch_polopt.py tests/fixtures/tf/instrumented_batch_polopt.py tests/benchmarks/test_benchmark_trpo.py garage/envs/mujoco/walker2d_env.py garage/algos/batch_polopt.py tests/garage/tf/algos/test_tnpg.py

if [[ "${TRAVIS_PULL_REQUEST}" != "false" && "${TRAVIS}" == "true" ]]; then
  pre-commit run --source "${TRAVIS_COMMIT_RANGE%...*}" \
                 --origin "${TRAVIS_COMMIT_RANGE#*...}"
  status="$((${status} | ${?}))"

  # Check commit messages
  while read commit; do
    commit_msg="$(mktemp)"
    git log --format=%B -n 1 "${commit}" > "${commit_msg}"
    pre-commit run --hook-stage commit-msg --commit-msg-file="${commit_msg}"
    pass=$?
    status="$((${status} | ${pass}))"

    # Print message if it fails
    if [[ "${pass}" -ne 0 ]]; then
      echo "Failing commit message:"
      cat "${commit_msg}"
    fi

  done < <(git rev-list "${TRAVIS_COMMIT_RANGE}")
else
  git remote set-branches --add origin master
  git fetch
  pre-commit run --source origin/master --origin ${TRAVIS_BRANCH}
  status="$((${status} | ${?}))"

  # Check commit messages
  while read commit; do
    commit_msg="$(mktemp)"
    git log --format=%B -n 1 "${commit}" > "${commit_msg}"
    pre-commit run --hook-stage commit-msg --commit-msg-file="${commit_msg}"
    pass=$?
    status="$((${status} | ${pass}))"

    # Print message if it fails
    if [[ "${pass}" -ne 0 ]]; then
      echo "Failing commit message:"
      cat "${commit_msg}"
    fi

  done < <(git rev-list origin/master..."${TRAVIS_BRANCH}")
fi

exit "${status}"
