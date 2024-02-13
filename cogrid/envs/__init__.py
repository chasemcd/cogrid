from cogrid.envs import registry
from cogrid.envs.goal_seeking import goal_seeking
from cogrid.envs.search_rescue import search_rescue
from cogrid.envs.simple_cooking import simple_cooking

registry.register("goal_seeking", goal_seeking.GoalSeeking)
registry.register("search_rescue", search_rescue.SearchRescue)
registry.register("simple_cooking", simple_cooking.SimpleCooking)
