from cogrid.envs import registry
from cogrid.envs.search_rescue import search_rescue, search_rescue_grid_objects
from cogrid.envs.overcooked import overcooked, overcooked_grid_objects
from cogrid.core import grid_object

registry.register("search_rescue", search_rescue.SearchRescue)
registry.register("overcooked", overcooked.Overcooked)