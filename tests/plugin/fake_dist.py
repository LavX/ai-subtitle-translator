"""Entry points for a fake extension distribution, so scope tests need no installed package."""

from subtitle_translator.core.backends import BUILTIN_DISTRIBUTION
from tests.core.registry_fakes import EchoBackend, FakeEntryPoint, SignInBackend, entry_points_from

HARNESS = "lavx-translator-harness"


def points(explode_harness=False):
    return entry_points_from(
        FakeEntryPoint("echo", EchoBackend, dist=BUILTIN_DISTRIBUTION),
        FakeEntryPoint("signin", SignInBackend, dist=HARNESS, explode=explode_harness),
    )
