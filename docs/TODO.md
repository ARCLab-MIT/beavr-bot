# 📋 TODOs

This file tracks pending tasks, improvements, and ideas.
Use `- [ ]` for open tasks and `- [x]` for completed ones.

---

## 🔧 Code Improvements
- [ ] Organize beavr/teleop/configs better
- [ ] Allow the program to algorithmically assign port numbers so we never have to think about port numbers for messaging
- [ ] Simplify the messaging API and maybe make it a PyPI package at some point in the future
- [ ] Abstract out the core logic for all components so that we don't have to carry a unique operator and interface for each robot.
      The idea here is to make it as easy as possible to add a new robot
- [ ] Create/find/use a sim environment for all robots (XArm, LEAP, etc.)
- [ ] Find a better solution for LeRobot integration. We will likely have it as a submodule/use the package and maintain a thin wrapper that allows us to use it
- [ ] The operator code is too long and complicated can we simplify it?
- [ ] We need better handling of IK for LEAP hand. Can we make the hand more dexterous?
- [ ] We also need better control for the xArm robot. Should we add a different IK solver (pink, ikpy, etc...) or different control (calculate direction and velocity and use velocity cartesian control (mode 4)).

---
## 🖥️ Interface
- [ ] Beavr app version 2.0 is out. Can we improve UX and interface?

---

## 📝 Documentation
- [ ] We need better documentation for all components
- [ ] We need a good tutorial about how to add a robot
- [ ] We need a good tutorial about how BEAVR works. This can be a Jupyter notebook or a video

---

## ✅ Testing
- [ ] We need unit tests for all components (these should be really simple)
- [ ] We need unit tests for all common items, folder: beavr/common
- [ ] Add integration tests for teleop

---

## 🚀 Features
- [ ] Implement joystick input adapter (this could be a really cool side project)
- [ ] Implement control of a LeRobot robot (Koch arms, SO101, etc)
- [ ] Support sim teleoperation mode

---

## 🧹 Cleanup / Technical Debt
- [ ] Review all `FIXME` and `TODO` comments in code

---

## 📌 Notes
- Link related issues like `#12` for visibility.
- Use labels: `priority-high`, `help wanted`, `good first issue`.
