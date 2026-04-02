# Next Steps

- Check Beinborn mail to see what I planned to do next (LR re-warming and replay -> How did I intend to make it work with stacking?)
- I think next steps are
  - Refactor Curriculum Learning so that
    - Stages are distinct
    - Get rid of re-assessing the difficulty for stage-wise curriculum
    - Make stages compatible with stacking
  - LR re-warming
  - Data Replay

# TODOs Later

- Remove local in requirements, remove local setup
- Adjust local setting up script similar to the one on the cluster
- Adjust setting up environment docu for new cluster script
  - Add info for the datasets download previously
  - Add info for downloading evaluation data
- Test setting up cluster from scratch
  - Especially huggingface and wandb login