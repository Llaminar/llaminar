## @file driver_evidence.py
# @brief One metadata contract for exact-owned Docker control/process retirement.
"""One metadata contract for exact-owned Docker control/process retirement.

Numerical panels and OpenCode share the same outer driver lifecycle. This
reader grants no work or cleanup authority; it revalidates the retained daemon
observations before either phase can qualify a release.
"""
import re


def validate_retirement(cleanup: dict) -> dict:
    """Reject incomplete ownership, unexpected live work or an unretired driver."""
    driver = cleanup['driver_retirement']
    if (cleanup.get('complete') is not True or cleanup.get('passed') is not True
            or cleanup.get('errors') != [] or cleanup.get('unexpected_live') != []
            or not re.fullmatch(r'llaminar-suite-driver-[0-9a-f]{32}', cleanup.get('owner', ''))
            or driver.get('Running') is not False or type(driver.get('Pid')) is not int or driver['Pid'] != 0
            or driver.get('Status') != 'absent'
            or any(not re.fullmatch(r'[0-9a-f]{64}', identity)
                   or state.get('Running') is not False or type(state.get('Pid')) is not int or state['Pid'] != 0
                   or state.get('Status') not in {'exited', 'created'}
                   for identity, state in cleanup['retired'].items())):
        raise ValueError('outer driver did not prove complete native retirement')
    return cleanup
