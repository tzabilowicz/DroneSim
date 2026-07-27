# DoneSim
DroneSim is a realistic simulation and training environment for RL models.

Assets: https://github.com/google-deepmind/mujoco_menagerie?tab=License-1-ov-file <br>
The drone asset for this simulation is from pre-developed Google Deepmind robotics assets (with modifications).

## Getting Started
#### Dependencies
```bash
pip install -r requirements.txt
```

#### Setting Up Assets
To get started, run
```bash
py clone_assets.py
```
This will clone the assets from the GitHub directory into the `\assets` folder. For additional setup steps and modifications, see the README.md in `\assets`. This is a walk-through of additional asset modifications and additions for the simulation.

#### Environment Testing
To test that everything is setup correctly, run
```bash
py main.py
```
You should see the MuJoCo viewer appear with a drone that takes off and hovers at a pre-defined altitude. You should also see a green sphere (target) rendered at a random location in the scene. Note that you may have to zoom out to see the target.

#### Training
The reinforcement learning (RL) model uses a PPO (proximal policy optimization) for its policy. To train the model, run
```bash
py train.py
```
This kicks off the training. To view the training in tensorboard, run
```bash
tensorboard --logdir model/tensorboard
```

#### Validation
Once the model has been trained, the model's performance and policy can be inspected.
```bash
py validate.py
```
This will run 10 simulations using the best trained model to navigate to the target. Additionally, the MuJoCo viewer will open and step through the simulation in real-time. Once the validation is complete, the tradjectory plot will be compluted and the mean reward and distance from episodes will be displayed.

The purpose of the validation is to show that the model learned how to navigate to the target.