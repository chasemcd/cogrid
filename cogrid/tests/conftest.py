"""Shared test fixtures and setup for cogrid tests.

V2-specific tests (test_v2_observation_channels, test_order_observation) live
in cogrid/envs/overcooked/ and run in a separate pytest invocation to avoid
polluting the type registry for core tests.
"""
