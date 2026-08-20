"""
Factorial plan: 5 families x 4 depths x 5 widths x 4 resolutions = 400 configs.

No a-priori filtering — every cell is attempted and logged.
"""

from typing import Iterator, TypedDict

FAMILIES = ["sequential", "residual", "dense", "branched_2way", "branched_4way"]
DEPTHS = [4, 8, 16, 32]
WIDTHS = [16, 32, 64, 128, 256]
RESOLUTIONS = [96, 160, 224, 384]


class Config(TypedDict):
    """One cell of the factorial design: family, depth, base width, resolution."""
    family: str
    depth: int
    base_width: int
    resolution: int
    tag: str


def make_tag(family: str, depth: int, base_width: int, resolution: int) -> str:
    """Build the canonical tag that identifies a configuration everywhere downstream."""
    return f"{family}_d{depth}_w{base_width}_r{resolution}"


def all_configs() -> list[Config]:
    """Return all 400 configurations of the factorial design.

    No cell is filtered out in advance. Configurations that fail to build are logged
    as failures rather than skipped, because where the build stops IS a result: it
    maps the empirical boundary of what this toolchain can handle.
    """
    out: list[Config] = []
    for family in FAMILIES:
        for depth in DEPTHS:
            for width in WIDTHS:
                for res in RESOLUTIONS:
                    out.append(
                        {
                            "family": family,
                            "depth": depth,
                            "base_width": width,
                            "resolution": res,
                            "tag": make_tag(family, depth, width, res),
                        }
                    )
    return out


def iter_configs() -> Iterator[Config]:
    """Iterate over the 400 configurations."""
    yield from all_configs()


if __name__ == "__main__":
    cfgs = all_configs()
    print(f"Total configurations: {len(cfgs)}")
    for c in cfgs[:5]:
        print(f"  {c['tag']}")
    print(f"  ... ({len(cfgs) - 5} more)")
