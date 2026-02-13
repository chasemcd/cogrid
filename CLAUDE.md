# Overview

This repository contains CoGrid, a library for creating multi-agent grid-world environments for reinforcement learning. The goal is to have a highly modular library that allows users to easily create and customize enviroments 
by accessing the core components of the environment individually: Rewards, Features, and Agents. 


## Structure

The core environment components live in cogrid/core. This defines all the generalized logic. All environment-specific logic (e.g., Overcooked, SearchRescue) lives in its own directory and subclasses core modules or registers new objects to be utilized in custom environments. 


## Critical

The most important consideration is to make the code as simple, concise, and easy to understand as possible. We must have the minimal number of code paths and it must be readable instantly to someone who is not familiar with the project.

No environment-specific logic should ever go into `cogrid/cogrid_env.py` or any file under `cogrid/core/`. Environment-specific behavior (e.g., Overcooked pot mechanics, SearchRescue scoring) must be implemented through the component API: `@register_object_type` classmethods (`build_tick_fn`, `build_interaction_fn`, `build_render_sync_fn`, etc.) and `@register_reward_type` subclasses. The autowire system composes these into scope_config hooks that the generic engine calls. 