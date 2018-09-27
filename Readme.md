RunFaithRun
---

An autopilot system using Computer Vision in Mirror's Edge Catalyst (a FPS game). [Homepage](http://shaofanlai.com/exp/6)

A library that can:
- Record and synchronize frames, key press and mouse controls from the game.
- Capture frames, do inference with TensorFlow and send control signal back to the game.
- Hack the memory of the game to get the position, velocity and orientation.

Algorithms:
- Supervised learning framework that can redo and generalize my action in game.
[![Demo on Supervised Learning](http://img.youtube.com/vi/EtMgox--Ofc/0.jpg)](http://www.youtube.com/watch?v=EtMgox--Ofc)
- A reinforcement learning (DQL) framework that tries to maximize the velocity of the character.
