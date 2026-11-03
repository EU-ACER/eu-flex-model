# Contributors to EU-Wide Flexibility Needs Assessment Model of the Future European Energy System
"""
Log formatting shared between the workflow and any logging extension.

Kept deliberately free of project imports. A logging extension is loaded from
inside :func:`helpers.configure_logging`, so an extension that needed the
formatter from ``helpers`` would be importing a module that is still being
imported. Keeping the formatter here means neither side can end up in a cycle.
"""

import logging


class RuleTagFormatter(logging.Formatter):
    """
    Formatter that embeds ``[rule=<name>]`` in each log line.

    Produces lines of the form ``LEVEL:logger:[rule=<name>] message``, which
    preserves the standard level and logger name while embedding the rule tag.
    Snakemake interleaves the output of rules running in parallel, so the tag
    is what lets a reader, or a log-consuming extension, attribute a line to
    the rule that emitted it.

    Parameters
    ----------
    rule : str
        Name of the Snakemake rule whose output is being formatted.
    """

    def __init__(self, rule: str) -> None:
        super().__init__(f"%(levelname)s:%(name)s:[rule={rule}] %(message)s")
