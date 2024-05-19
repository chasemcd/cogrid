from pettingzoo import test as pettingzoo_test
from cogrid import cogrid_env
from cogrid.envs import registry
from cogrid.envs import overcooked
from cogrid.envs.overcooked import rewards

import unittest


class TestPettingZooAPI(unittest.TestCase):

    def test_overcooked_pettingzoo(self):
        env = registry.make(
            "Overcooked-CrampedRoom-V0",
        )
        pettingzoo_test.parallel_api_test(env, num_cycles=1000)


if __name__ == "__main__":
    unittest.main()
